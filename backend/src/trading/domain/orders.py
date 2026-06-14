"""Order lifecycle: intent -> submitted -> acknowledged -> (partially) filled -> final."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    INTENT = "intent"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


TERMINAL_STATUSES = frozenset({OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED})

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.INTENT: frozenset({OrderStatus.SUBMITTED, OrderStatus.REJECTED}),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.ACKNOWLEDGED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED}
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


class InvalidOrderTransitionError(Exception):
    def __init__(self, current: OrderStatus, attempted: OrderStatus) -> None:
        super().__init__(f"invalid order transition: {current} -> {attempted}")
        self.current = current
        self.attempted = attempted


def can_transition(current: OrderStatus, new: OrderStatus) -> bool:
    return new in _ALLOWED_TRANSITIONS[current]


@dataclass(frozen=True, slots=True)
class Order:
    client_order_id: str
    decision_id: str
    sleeve_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    status: OrderStatus
    broker_order_id: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")

    def with_status(self, new: OrderStatus, broker_order_id: str | None = None) -> "Order":
        if not can_transition(self.status, new):
            raise InvalidOrderTransitionError(self.status, new)
        return replace(self, status=new, broker_order_id=broker_order_id or self.broker_order_id)


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fees: Decimal
    filled_at: datetime
