"""Trade-updates stream port: real-time broker order/fill events (FR-REC, FR-EXE, spec §11).

The broker (Alpaca paper) fills ASYNCHRONOUSLY after submission. The 30s polling order-sync
closes the loop on an interval; this port is the push-based complement — a stream of
``TradeUpdate``s the broker emits the moment an order changes (fill / partial fill / status
change), so positions and P&L reflect reality in seconds, not on the next poll.

Like ``QuoteStreamPort`` it is allowed to be a disabled no-op so the app always has something
injected and opens no socket by default. ``run`` is the only method that touches the network;
it is started as a background task on app startup and is cancel-safe.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from trading.domain.orders import OrderSide, OrderStatus


@dataclass(frozen=True, slots=True)
class TradeUpdate:
    """One broker trade-update: a status change, optionally carrying a fill.

    ``event`` is the raw broker event name (e.g. ``fill``/``partial_fill``/``canceled``) kept
    for the audit trail; the mapped lifecycle ``status`` drives our projections. Fill fields
    are populated only on (partial) fills.
    """

    client_order_id: str
    broker_order_id: str
    event: str
    status: OrderStatus
    symbol: str
    side: OrderSide
    occurred_at: datetime
    fill_price: Decimal | None = None
    fill_quantity: Decimal | None = None
    filled_at: datetime | None = None

    @property
    def is_fill(self) -> bool:
        return self.fill_price is not None and self.fill_quantity is not None


class TradeUpdatesStreamPort(Protocol):
    """Real-time, push-based source of broker trade updates.

    ``enabled`` lets the composition root report a disabled stream (no socket, no updates).
    ``run`` connects, authenticates, subscribes to ``trade_updates`` and dispatches each
    decoded ``TradeUpdate`` to ``handler`` until cancelled; the disabled implementation
    returns immediately.
    """

    @property
    def enabled(self) -> bool: ...

    async def run(self, handler: "TradeUpdateHandler") -> None: ...


class TradeUpdateHandler(Protocol):
    """Consumes one decoded trade update (the apply-trade-update use case implements this)."""

    async def __call__(self, update: TradeUpdate, *, sleeve_id: str) -> None: ...
