from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.place_order import PlaceOrder, PlaceOrderRequest
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.reconcile import Reconcile
from trading.application.sync_orders import SyncOrders
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.persistence.memory import InMemoryEventStore


async def test_clean_reconciliation_after_paper_fill(
    broker: PaperBroker,
    event_store: InMemoryEventStore,
    place_order: PlaceOrder,
    reconcile: Reconcile,
    clock: FixedClock,
) -> None:
    await place_order.execute(
        PlaceOrderRequest(
            sleeve_id="sleeve-test",
            decision_id="dec-1",
            client_order_id="ord-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("5"),
            correlation_id="corr-1",
        )
    )

    report = await reconcile.execute()

    assert report.clean
    types = [s.event.event_type for s in await event_store.read_stream("reconciliation")]
    assert types == [EventType.RECONCILIATION_COMPLETED]


class EmptyBroker:
    """A broker that knows nothing — simulates a crash between intent and submission."""

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
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [BrokerPosition("TSLA", Decimal("3"), Decimal("250"))]

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("750"))


async def test_drift_is_detected_and_recorded(clock: FixedClock) -> None:
    event_store = InMemoryEventStore(clock)
    # Local log shows an order intent the broker never saw...
    await event_store.append(
        DomainEvent(
            stream_id="order-ord-lost",
            event_type=EventType.ORDER_INTENT,
            payload={"client_order_id": "ord-lost", "symbol": "AAPL"},
            occurred_at=clock.now(),
            correlation_id="corr-x",
        )
    )
    # ...and the broker holds a position the local log never recorded.
    reconcile = Reconcile(broker=EmptyBroker(), event_store=event_store, clock=clock)

    report = await reconcile.execute()

    assert not report.clean
    drift_kinds = {(d.kind, d.key) for d in report.drifts}
    assert ("order", "ord-lost") in drift_kinds
    assert ("position", "TSLA") in drift_kinds

    recorded = [s.event.event_type for s in await event_store.read_stream("reconciliation")]
    assert recorded.count(EventType.RECONCILIATION_DRIFT) == 2
    assert recorded[-1] == EventType.RECONCILIATION_COMPLETED


class _PositionOnlyBroker:
    """Holds one position and no orders — a standing position drift vs an empty local log."""

    def __init__(self, quantity: Decimal) -> None:
        self._quantity = quantity

    async def submit_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [BrokerPosition("TSLA", self._quantity, Decimal("250"))]

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"))


async def test_repeated_identical_drift_is_deduped(clock: FixedClock) -> None:
    # Review F14: the same drift recurred every run (fill ingestion off), appending an identical
    # event each time. It must now be recorded ONCE while every report still reflects it.
    store = InMemoryEventStore(clock)
    reconcile = Reconcile(broker=_PositionOnlyBroker(Decimal("3")), event_store=store, clock=clock)

    first = await reconcile.execute()
    second = await reconcile.execute()

    assert not first.clean and not second.clean  # drift still reported both runs
    assert ("position", "TSLA") in {(d.kind, d.key) for d in first.drifts}
    drift_events = [
        s
        for s in await store.read_stream("reconciliation")
        if s.event.event_type == EventType.RECONCILIATION_DRIFT
    ]
    assert len(drift_events) == 1  # not re-appended on the identical second run


async def test_changed_drift_is_recorded_again(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await Reconcile(
        broker=_PositionOnlyBroker(Decimal("3")), event_store=store, clock=clock
    ).execute()
    # Broker quantity changed 3 -> 5: a DIFFERENT drift value, so it must be recorded anew.
    await Reconcile(
        broker=_PositionOnlyBroker(Decimal("5")), event_store=store, clock=clock
    ).execute()

    drift_events = [
        s.event.payload["broker"]
        for s in await store.read_stream("reconciliation")
        if s.event.event_type == EventType.RECONCILIATION_DRIFT
    ]
    assert drift_events == ["3", "5"]


class _SyncedBroker:
    """One filled order plus the matching position — what the broker looks like after a
    cron-placed order fills at the open."""

    async def submit_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        fill = Fill(
            client_order_id="ord-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=Decimal("100"),
            quantity=Decimal("5"),
            fees=Decimal("0"),
            filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
        )
        return [
            BrokerOrder(
                client_order_id="ord-1",
                broker_order_id="b-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=Decimal("5"),
                order_type=OrderType.MARKET,
                status=OrderStatus.FILLED,
                fills=(fill,),
            )
        ]

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [BrokerPosition("AAPL", Decimal("5"), Decimal("100"))]

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("500"))


async def test_sync_then_reconcile_shows_no_drift(clock: FixedClock) -> None:
    # The cron path (review F14): a cron-placed order fills async; SyncOrders records the fill and
    # a following reconcile then finds the projection matches the broker — no drift.
    store = InMemoryEventStore(clock)
    await store.append(
        DomainEvent(
            stream_id="order-ord-1",
            event_type=EventType.ORDER_SUBMITTED,
            payload={"client_order_id": "ord-1", "sleeve_id": "sleeve-political"},
            occurred_at=clock.now(),
            correlation_id="corr-1",
        )
    )
    broker = _SyncedBroker()

    sync = SyncOrders(
        broker=broker,
        event_store=store,
        clock=clock,
        sleeve_id="sleeve-political",
    )
    assert (await sync.execute()).fills_emitted == 1

    reconcile = Reconcile(broker=broker, event_store=store, clock=clock)
    report = await reconcile.execute()
    assert report.clean
