from decimal import Decimal

import pytest

from trading.domain.orders import (
    InvalidOrderTransitionError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    can_transition,
)


def _order(status: OrderStatus = OrderStatus.INTENT) -> Order:
    return Order(
        client_order_id="ord-1",
        decision_id="dec-1",
        sleeve_id="sleeve-test",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        order_type=OrderType.MARKET,
        status=status,
    )


def test_full_lifecycle() -> None:
    order = _order()
    order = order.with_status(OrderStatus.SUBMITTED, broker_order_id="b-1")
    order = order.with_status(OrderStatus.ACKNOWLEDGED)
    order = order.with_status(OrderStatus.PARTIALLY_FILLED)
    order = order.with_status(OrderStatus.FILLED)
    assert order.status is OrderStatus.FILLED
    assert order.broker_order_id == "b-1"


def test_terminal_states_admit_no_transitions() -> None:
    for terminal in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
        for target in OrderStatus:
            assert not can_transition(terminal, target)


def test_invalid_transition_raises() -> None:
    order = _order(OrderStatus.INTENT)
    with pytest.raises(InvalidOrderTransitionError):
        order.with_status(OrderStatus.ACKNOWLEDGED)  # must be submitted first


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Order(
            client_order_id="ord-1",
            decision_id="dec-1",
            sleeve_id="sleeve-test",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("0"),
            order_type=OrderType.MARKET,
            status=OrderStatus.INTENT,
        )
