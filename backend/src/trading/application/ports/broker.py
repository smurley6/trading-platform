"""Broker port: the single execution boundary (FR-EXE). Paper and live differ only here."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    status: OrderStatus
    fills: tuple[Fill, ...] = ()


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal


@dataclass(frozen=True, slots=True)
class BrokerAccount:
    cash: Decimal
    equity: Decimal
    # Spendable capital right now — the shared-pool headline (real accounts allow margin, so this
    # can exceed cash). Defaults to cash so pre-existing constructions (and cash-only simulators)
    # stay correct without change.
    buying_power: Decimal = Decimal("0")


class OrderNotFoundError(Exception):
    """No order exists for the given client_order_id at the broker."""


class BrokerPort(Protocol):
    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        """Submit an order. MUST be idempotent on `client_order_id` (FR-EXE-3)."""
        ...

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        """Cancel an open order by its idempotency key, returning its resulting state.

        Raises `OrderNotFoundError` if no such order exists. MUST be safe to call on an
        already-terminal order (filled/canceled/rejected): it is a no-op that returns the
        current state rather than erroring.
        """
        ...

    async def list_orders(self) -> Sequence[BrokerOrder]: ...

    async def list_positions(self) -> Sequence[BrokerPosition]: ...

    async def get_account(self) -> BrokerAccount: ...
