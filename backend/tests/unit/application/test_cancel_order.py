"""CancelOrder: delegates to the right sleeve's broker and records the audit trail."""

from collections.abc import Callable, Sequence
from decimal import Decimal

import pytest

from tests.conftest import FixedClock
from trading.application.cancel_order import CancelFailedError, CancelOrder
from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderNotFoundError,
)
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide, OrderStatus, OrderType
from trading.infrastructure.persistence.memory import InMemoryEventStore


def _order(status: OrderStatus) -> BrokerOrder:
    return BrokerOrder(
        client_order_id="ord-1",
        broker_order_id="b-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
        status=status,
    )


class _FakeBroker:
    """Minimal BrokerPort whose cancel behavior is injected per test."""

    def __init__(self, on_cancel: Callable[[str], BrokerOrder]) -> None:
        self._on_cancel = on_cancel
        self.cancel_calls = 0

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
        self.cancel_calls += 1
        return self._on_cancel(client_order_id)

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return []

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"))


def _cancel(broker: _FakeBroker, clock: FixedClock) -> tuple[CancelOrder, InMemoryEventStore]:
    store = InMemoryEventStore(clock)
    use_case = CancelOrder(brokers={"sleeve-x": broker}, event_store=store, clock=clock)
    return use_case, store


async def _record_order(store: InMemoryEventStore, clock: FixedClock, *, sleeve_id: str) -> None:
    await store.append(
        DomainEvent(
            stream_id="order-ord-1",
            event_type=EventType.ORDER_SUBMITTED,
            payload={"client_order_id": "ord-1", "sleeve_id": sleeve_id},
            occurred_at=clock.now(),
            correlation_id="c1",
        )
    )


async def test_cancel_records_user_action_and_status_change(clock: FixedClock) -> None:
    use_case, store = _cancel(_FakeBroker(lambda _id: _order(OrderStatus.CANCELED)), clock)
    await _record_order(store, clock, sleeve_id="sleeve-x")

    result = await use_case.execute(client_order_id="ord-1")

    assert result.sleeve_id == "sleeve-x"
    assert result.order.status is OrderStatus.CANCELED
    events = await store.read_all()
    types = [e.event.event_type for e in events]
    assert EventType.USER_ACTION in types
    assert EventType.ORDER_STATUS_CHANGED in types
    user_action = next(e for e in events if e.event.event_type == EventType.USER_ACTION)
    assert user_action.event.payload["action"] == "cancel_order"


async def test_not_found_propagates(clock: FixedClock) -> None:
    def _raise(_id: str) -> BrokerOrder:
        raise OrderNotFoundError("nope")

    use_case, store = _cancel(_FakeBroker(_raise), clock)
    await _record_order(store, clock, sleeve_id="sleeve-x")
    with pytest.raises(OrderNotFoundError):
        await use_case.execute(client_order_id="ord-1")


async def test_broker_failure_is_wrapped(clock: FixedClock) -> None:
    def _boom(_id: str) -> BrokerOrder:
        raise RuntimeError("alpaca 500")

    use_case, store = _cancel(_FakeBroker(_boom), clock)
    await _record_order(store, clock, sleeve_id="sleeve-x")
    with pytest.raises(CancelFailedError):
        await use_case.execute(client_order_id="ord-1")


async def test_resolves_sleeve_from_event_log(clock: FixedClock) -> None:
    seen: dict[str, str] = {}

    def _record(client_order_id: str) -> BrokerOrder:
        seen["client_order_id"] = client_order_id
        return _order(OrderStatus.CANCELED)

    use_case, store = _cancel(_FakeBroker(_record), clock)
    await _record_order(store, clock, sleeve_id="sleeve-x")

    result = await use_case.execute(client_order_id="ord-1")

    assert result.sleeve_id == "sleeve-x"  # resolved from the log
    assert seen["client_order_id"] == "ord-1"


async def test_unknown_order_without_sleeve_is_not_found_and_never_adopted(
    clock: FixedClock,
) -> None:
    """F27: cancelling an order the platform never placed must 404, not silently adopt it.

    A manual Alpaca-dashboard order (no local lifecycle events, no explicit sleeve param) must
    NOT be stamped into any sleeve's ledger — the broker is never even called for it.
    """
    broker = _FakeBroker(lambda _id: _order(OrderStatus.CANCELED))
    use_case, store = _cancel(broker, clock)  # no events recorded for ord-1

    with pytest.raises(OrderNotFoundError):
        await use_case.execute(client_order_id="ord-1")

    assert broker.cancel_calls == 0  # never adopted / never acted on
    # No sleeve-scoped events were written for the foreign order.
    events = await store.read_all()
    assert not [e for e in events if e.event.payload.get("client_order_id") == "ord-1"]


async def test_explicit_sleeve_param_adopts_unknown_order(clock: FixedClock) -> None:
    """F27: an unknown order CAN be cancelled, but only with an EXPLICIT sleeve param."""
    broker = _FakeBroker(lambda _id: _order(OrderStatus.CANCELED))
    use_case, _ = _cancel(broker, clock)  # no recorded events for ord-1

    result = await use_case.execute(client_order_id="ord-1", sleeve_id="sleeve-x")

    assert result.sleeve_id == "sleeve-x"
    assert broker.cancel_calls == 1
