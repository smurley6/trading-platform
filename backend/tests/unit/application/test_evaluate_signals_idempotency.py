"""Order idempotency + open-order awareness + evaluate serialization (review F5 + F6).

A cron places a DAY order pre-open; it queues UNFILLED until 09:30 ET. Any second evaluation in
that window (a crash-retry, POST /ingest, /proposals/execute, the scheduler) used to rebuild the
portfolio from ``list_positions()`` only — held=0 — re-propose the same delta, mint a FRESH
``ord-{uuid4}`` so broker-side idempotency could never fire, and submit again -> double exposure.

These tests pin the three fixes:
- **F5 deterministic COID:** a retry re-uses ``ord-{sleeve}-{symbol}-{side}-{cycle-date}`` so the
  broker's own idempotency collapses it to ONE order.
- **F5 open-order awareness:** a queued/unfilled open order is counted as held, so the same delta
  is not re-proposed within the window.
- **F6 serialization:** the per-sleeve ``asyncio.Lock`` prevents two overlapping ``execute()``
  coroutines from both snapshotting before either places.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from decimal import Decimal

from tests.conftest import FixedClock, SignalFactory
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.place_order import PlaceOrder
from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderNotFoundError,
)
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.domain.events import EventType
from trading.domain.orders import OrderSide, OrderStatus, OrderType
from trading.domain.provenance import Provenance
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.strategies.trivial import TrivialStrategy

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("50"),
    max_position_concentration_pct=Decimal("90"),
    max_drawdown_pct=Decimal("90"),
)
_SLEEVE = Sleeve(
    id="sleeve-political",
    strategy_id="trivial",
    allocated_capital=Decimal("100000"),
    mode=TradingMode.PAPER,
    status=SleeveStatus.RUNNING,
)


class _QueuingBroker:
    """A broker that QUEUES orders unfilled — like a pre-open DAY order that sits until the open.

    ``submit_order`` records a SUBMITTED (unfilled) order and NEVER updates positions, so
    ``list_positions()`` stays empty while the order is in flight. Idempotent on
    ``client_order_id`` exactly like the paper and Alpaca adapters. ``visible_orders`` toggles
    whether ``list_orders()`` reports the queued orders, so a test can isolate the deterministic-
    COID layer (False) from the open-order-awareness layer (True).
    """

    def __init__(
        self,
        *,
        cash: Decimal,
        visible_orders: bool = True,
        seed_orders: Sequence[BrokerOrder] = (),
    ) -> None:
        self._cash = cash
        self._visible = visible_orders
        self._orders: dict[str, BrokerOrder] = {o.client_order_id: o for o in seed_orders}
        self.new_submissions = 0
        self._counter = 0

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        existing = self._orders.get(client_order_id)
        if existing is not None:
            return existing  # idempotent resubmission (FR-EXE-3)
        self.new_submissions += 1
        self._counter += 1
        order = BrokerOrder(
            client_order_id=client_order_id,
            broker_order_id=f"q-{self._counter}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=OrderStatus.SUBMITTED,
            fills=(),
        )
        self._orders[client_order_id] = order
        return order

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        order = self._orders.get(client_order_id)
        if order is None:
            raise OrderNotFoundError(client_order_id)
        return order

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return list(self._orders.values()) if self._visible else []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return []

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=self._cash, equity=self._cash, buying_power=self._cash)


def _evaluator(
    broker: _QueuingBroker, store: InMemoryEventStore, clock: FixedClock
) -> EvaluateSignals:
    return EvaluateSignals(
        strategy=TrivialStrategy(position_size=Decimal("5")),
        sleeve=_SLEEVE,
        risk_gate=RiskGate(_LIMITS, KillSwitch()),
        place_order=PlaceOrder(broker=broker, event_store=store, clock=clock),
        broker=broker,
        market_data=StaticMarketData({"AAPL": Decimal("100")}, clock),
        event_store=store,
        clock=clock,
    )


def _expected_coid(clock: FixedClock) -> str:
    return f"ord-{_SLEEVE.id}-AAPL-buy-{clock.now().date().isoformat()}"


async def test_retry_reuses_deterministic_coid_and_does_not_double_submit(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # visible_orders=False hides the queued order from sizing, so the SECOND run re-proposes the
    # same delta (the F5 bug scenario). The deterministic COID must still collapse it to ONE order.
    broker = _QueuingBroker(cash=Decimal("100000"), visible_orders=False)
    store = InMemoryEventStore(clock)
    evaluator = _evaluator(broker, store, clock)
    signals = [make_signal(ticker="AAPL", sentiment=0.8, provenance=Provenance.REAL)]

    first = await evaluator.execute(signals)
    second = await evaluator.execute(signals)  # crash-retry in the same pre-open window

    # Both runs computed the SAME deterministic key; the broker deduped to a single new submission.
    assert first.orders[0].client_order_id == _expected_coid(clock)
    assert second.orders[0].client_order_id == _expected_coid(clock)
    assert broker.new_submissions == 1

    intents = [
        s.event for s in await store.read_all() if s.event.event_type == EventType.ORDER_INTENT
    ]
    # Both intents write to the SAME order stream (deterministic COID), not two uuid4 streams.
    assert {e.stream_id for e in intents} == {f"order-{_expected_coid(clock)}"}


async def test_queued_open_order_counts_as_held_and_is_not_reproposed(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # A prior run already placed a BUY for the full target (5 shares) that is still queued unfilled.
    queued = BrokerOrder(
        client_order_id=_expected_coid(clock),
        broker_order_id="q-seed",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        order_type=OrderType.MARKET,
        status=OrderStatus.SUBMITTED,
        fills=(),
    )
    broker = _QueuingBroker(cash=Decimal("100000"), visible_orders=True, seed_orders=[queued])
    store = InMemoryEventStore(clock)
    evaluator = _evaluator(broker, store, clock)

    result = await evaluator.execute(
        [make_signal(ticker="AAPL", sentiment=0.8, provenance=Provenance.REAL)]
    )

    # target=5, position=0, pending BUY=5 -> current=5 -> delta=0 -> nothing proposed or placed.
    assert result.orders == ()
    assert broker.new_submissions == 0
    decisions = [
        s.event for s in await store.read_all() if s.event.event_type == EventType.DECISION_MADE
    ]
    assert decisions == []


async def test_partially_filled_open_order_only_counts_the_unfilled_remainder(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # A prior 5-share BUY is partially filled to 2 (2 already in positions... modeled here as the
    # remainder only): remainder pending = 3, so target 5 - 3 = 2 more should still be proposed.
    from trading.domain.orders import Fill

    partial = BrokerOrder(
        client_order_id="ord-prior",
        broker_order_id="q-seed",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        order_type=OrderType.MARKET,
        status=OrderStatus.PARTIALLY_FILLED,
        fills=(
            Fill(
                client_order_id="ord-prior",
                symbol="AAPL",
                side=OrderSide.BUY,
                price=Decimal("100"),
                quantity=Decimal("2"),
                fees=Decimal("0"),
                filled_at=clock.now(),
            ),
        ),
    )
    broker = _QueuingBroker(cash=Decimal("100000"), visible_orders=True, seed_orders=[partial])
    store = InMemoryEventStore(clock)
    evaluator = _evaluator(broker, store, clock)

    result = await evaluator.execute(
        [make_signal(ticker="AAPL", sentiment=0.8, provenance=Provenance.REAL)]
    )

    # pending remainder = 5 - 2 = 3; target 5 - 3 = 2 still to buy.
    assert [(o.symbol, o.quantity) for o in result.orders] == [("AAPL", Decimal("2"))]
    assert broker.new_submissions == 1


class _ConcurrencyProbeBroker(_QueuingBroker):
    """Records the max number of evaluations whose critical section overlapped.

    ``get_account`` bumps a counter and yields to the event loop; if the per-sleeve lock did NOT
    serialize ``execute()``, a second concurrent evaluation would enter here during the first's
    suspension and drive ``max_concurrent`` above 1.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._active = 0
        self.max_concurrent = 0

    async def get_account(self) -> BrokerAccount:
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        await asyncio.sleep(0)  # yield: an unserialized peer could interleave right here
        self._active -= 1
        return await super().get_account()


async def test_overlapping_executions_are_serialized_and_place_one_order(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    broker = _ConcurrencyProbeBroker(cash=Decimal("100000"), visible_orders=True)
    store = InMemoryEventStore(clock)
    evaluator = _evaluator(broker, store, clock)
    signals = [make_signal(ticker="AAPL", sentiment=0.8, provenance=Provenance.REAL)]

    # Two overlapping execute() calls for the same (sleeve, symbol, side, cycle-date).
    await asyncio.gather(evaluator.execute(signals), evaluator.execute(signals))

    # The asyncio.Lock kept the critical sections from ever overlapping...
    assert broker.max_concurrent == 1
    # ...so the second run saw the first's queued order as held and placed nothing new.
    assert broker.new_submissions == 1


class _RecordingCrossProcessLock:
    """Fake CrossProcessLockPort that records guard enter/exit, proving release on ALL paths."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    @asynccontextmanager
    async def guard(self, key: str) -> AsyncIterator[None]:
        self.events.append(("enter", key))
        try:
            yield
        finally:
            self.events.append(("exit", key))


class _ExplodingBroker(_QueuingBroker):
    async def get_account(self) -> BrokerAccount:
        raise RuntimeError("mid-evaluation failure")


async def test_cross_process_guard_wraps_execute_and_releases_on_exception(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    lock = _RecordingCrossProcessLock()
    broker = _ExplodingBroker(cash=Decimal("100000"))
    store = InMemoryEventStore(clock)
    evaluator = EvaluateSignals(
        strategy=TrivialStrategy(position_size=Decimal("5")),
        sleeve=_SLEEVE,
        risk_gate=RiskGate(_LIMITS, KillSwitch()),
        place_order=PlaceOrder(broker=broker, event_store=store, clock=clock),
        broker=broker,
        market_data=StaticMarketData({"AAPL": Decimal("100")}, clock),
        event_store=store,
        clock=clock,
        cross_process_lock=lock,
    )

    with contextlib.suppress(RuntimeError):
        await evaluator.execute(
            [make_signal(ticker="AAPL", sentiment=0.8, provenance=Provenance.REAL)]
        )

    # The advisory-lock guard was entered for THIS sleeve and released despite the mid-loop failure.
    assert lock.events == [
        ("enter", f"evaluate-{_SLEEVE.id}"),
        ("exit", f"evaluate-{_SLEEVE.id}"),
    ]
