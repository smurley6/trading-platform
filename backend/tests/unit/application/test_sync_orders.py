"""SyncOrders: ingest the broker's async fills/status changes the local log is missing.

The Alpaca venue fills AFTER submission, so PlaceOrder never sees the fill. SyncOrders pulls
the broker's orders and appends the missing `order.filled` / `order.status_changed` events,
idempotently (rebuilds the local picture from the log every run), and only for orders the
local log already knows about.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.sync_orders import SyncOrders
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType
from trading.infrastructure.persistence.memory import InMemoryEventStore

_SLEEVE = "sleeve-test"


def _fill(symbol: str = "AAPL", quantity: str = "5") -> Fill:
    return Fill(
        client_order_id="ord-1",
        symbol=symbol,
        side=OrderSide.BUY,
        price=Decimal("100"),
        quantity=Decimal(quantity),
        fees=Decimal("0"),
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )


def _broker_order(
    *,
    client_order_id: str = "ord-1",
    status: OrderStatus,
    fills: tuple[Fill, ...] = (),
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="b-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        order_type=OrderType.MARKET,
        status=status,
        fills=fills,
    )


class _FakeBroker:
    """Minimal BrokerPort returning a fixed list of orders."""

    def __init__(self, orders: Sequence[BrokerOrder]) -> None:
        self._orders = list(orders)

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return self._orders

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return []

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"))


async def _seed_submitted(store: InMemoryEventStore, clock: FixedClock) -> None:
    """Make the local log aware of `ord-1` at status `submitted` (as PlaceOrder would)."""
    await store.append(
        DomainEvent(
            stream_id="order-ord-1",
            event_type=EventType.ORDER_SUBMITTED,
            payload={"client_order_id": "ord-1", "sleeve_id": _SLEEVE},
            occurred_at=clock.now(),
            correlation_id="corr-seed",
        )
    )


def _sync(broker: _FakeBroker, store: InMemoryEventStore, clock: FixedClock) -> SyncOrders:
    return SyncOrders(broker=broker, event_store=store, clock=clock, sleeve_id=_SLEEVE)


async def _order_events(store: InMemoryEventStore) -> list[str]:
    stored = await store.read_stream("order-ord-1")
    return [s.event.event_type for s in stored]


async def test_emits_missing_fill(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    broker = _FakeBroker([_broker_order(status=OrderStatus.FILLED, fills=(_fill(),))])

    report = await _sync(broker, store, clock).execute()

    assert report.fills_emitted == 1
    # Filled is also a status change vs the local `submitted`.
    assert report.status_changes_emitted == 1
    assert report.checked_orders == 1
    types = await _order_events(store)
    assert types.count(EventType.ORDER_FILLED) == 1
    assert types.count(EventType.ORDER_STATUS_CHANGED) == 1
    # The fill carries the place_order payload shape (used by the positions projection).
    fill_event = next(
        s
        for s in await store.read_stream("order-ord-1")
        if s.event.event_type == EventType.ORDER_FILLED
    )
    payload = fill_event.event.payload
    assert payload["sleeve_id"] == _SLEEVE
    assert payload["symbol"] == "AAPL"
    assert payload["quantity"] == "5"
    assert payload["price"] == "100"


async def test_idempotent_second_run_emits_nothing(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    broker = _FakeBroker([_broker_order(status=OrderStatus.FILLED, fills=(_fill(),))])
    sync = _sync(broker, store, clock)

    first = await sync.execute()
    assert not first.emitted_nothing

    second = await sync.execute()

    assert second.emitted_nothing
    assert second.fills_emitted == 0
    assert second.status_changes_emitted == 0
    # Still exactly one fill + one status change recorded — no duplicates.
    types = await _order_events(store)
    assert types.count(EventType.ORDER_FILLED) == 1
    assert types.count(EventType.ORDER_STATUS_CHANGED) == 1


async def test_status_only_change_is_emitted(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    # Canceled at the broker, no fills.
    broker = _FakeBroker([_broker_order(status=OrderStatus.CANCELED)])

    report = await _sync(broker, store, clock).execute()

    assert report.fills_emitted == 0
    assert report.status_changes_emitted == 1
    types = await _order_events(store)
    assert EventType.ORDER_FILLED not in types
    assert types.count(EventType.ORDER_STATUS_CHANGED) == 1
    status_event = next(
        s
        for s in await store.read_stream("order-ord-1")
        if s.event.event_type == EventType.ORDER_STATUS_CHANGED
    )
    assert status_event.event.payload["status"] == OrderStatus.CANCELED.value


async def test_unknown_order_is_ignored(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    # Local log knows nothing about this order; the broker reports a filled account order.
    broker = _FakeBroker(
        [
            _broker_order(
                client_order_id="ord-foreign",
                status=OrderStatus.FILLED,
                fills=(_fill(),),
            )
        ]
    )

    report = await _sync(broker, store, clock).execute()

    assert report.emitted_nothing
    assert report.checked_orders == 0
    assert list(await store.read_all()) == []


async def test_status_matching_local_is_not_re_emitted(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    # Broker still reports `submitted` — the baseline PlaceOrder already recorded.
    broker = _FakeBroker([_broker_order(status=OrderStatus.SUBMITTED)])

    report = await _sync(broker, store, clock).execute()

    assert report.checked_orders == 1
    assert report.emitted_nothing
    assert await _order_events(store) == [EventType.ORDER_SUBMITTED]
