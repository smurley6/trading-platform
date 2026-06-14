"""PlaceOrder: write-ahead-of-action, event emission, and failure handling."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tests.conftest import FixedClock
from trading.application.place_order import OrderSubmissionError, PlaceOrder, PlaceOrderRequest
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.domain.events import EventType
from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType
from trading.infrastructure.persistence.memory import InMemoryEventStore


def _request() -> PlaceOrderRequest:
    return PlaceOrderRequest(
        sleeve_id="sleeve-test",
        decision_id="dec-1",
        client_order_id="ord-dec-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        correlation_id="corr-1",
    )


class IntentCheckingBroker:
    """Asserts the order intent event was durably recorded BEFORE the broker call."""

    def __init__(self, event_store: InMemoryEventStore) -> None:
        self._event_store = event_store
        self.intent_recorded_before_submit: bool | None = None

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        stream = await self._event_store.read_stream(f"order-{client_order_id}")
        self.intent_recorded_before_submit = any(
            s.event.event_type == EventType.ORDER_INTENT for s in stream
        )
        fill = Fill(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            price=Decimal("100"),
            quantity=quantity,
            fees=Decimal("0"),
            filled_at=datetime(2026, 6, 12, tzinfo=UTC),
        )
        return BrokerOrder(
            client_order_id=client_order_id,
            broker_order_id="b-1",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=OrderStatus.FILLED,
            fills=(fill,),
        )

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return []

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"))


class FailingBroker(IntentCheckingBroker):
    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        raise ConnectionError("broker unreachable")


async def test_intent_is_persisted_before_broker_call(clock: FixedClock) -> None:
    event_store = InMemoryEventStore(clock)
    broker = IntentCheckingBroker(event_store)
    place_order = PlaceOrder(broker=broker, event_store=event_store, clock=clock)

    await place_order.execute(_request())

    assert broker.intent_recorded_before_submit is True


async def test_successful_order_emits_full_event_chain(clock: FixedClock) -> None:
    event_store = InMemoryEventStore(clock)
    place_order = PlaceOrder(
        broker=IntentCheckingBroker(event_store), event_store=event_store, clock=clock
    )

    await place_order.execute(_request())

    types = [s.event.event_type for s in await event_store.read_stream("order-ord-dec-1")]
    assert types == [
        EventType.ORDER_INTENT,
        EventType.ORDER_SUBMITTED,
        EventType.ORDER_FILLED,
        EventType.ORDER_STATUS_CHANGED,
    ]


async def test_broker_failure_is_recorded_and_reraised(clock: FixedClock) -> None:
    event_store = InMemoryEventStore(clock)
    place_order = PlaceOrder(
        broker=FailingBroker(event_store), event_store=event_store, clock=clock
    )

    with pytest.raises(OrderSubmissionError):
        await place_order.execute(_request())

    types = [s.event.event_type for s in await event_store.read_stream("order-ord-dec-1")]
    assert types == [EventType.ORDER_INTENT, EventType.ORDER_SUBMIT_FAILED]
