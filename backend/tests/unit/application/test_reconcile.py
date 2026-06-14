from collections.abc import Sequence
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.place_order import PlaceOrder, PlaceOrderRequest
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.reconcile import Reconcile
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide, OrderType
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
