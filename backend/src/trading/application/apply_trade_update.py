"""ApplyTradeUpdate use case: record a real-time broker trade update as events (FR-REC, §11).

The push-based complement to ``SyncOrders``. Where the 30s poller diffs the broker's whole
order list on an interval, this records a single ``TradeUpdate`` the instant the broker emits
it, appending the SAME ``order.filled`` / ``order.status_changed`` events the poller does so
every downstream projection (positions, P&L, order status) is identical regardless of which
path observed the change.

Idempotency (do NOT double-record), against both repeats on the stream and the poller writing
the same fact:
  * A fill is recorded only if no ``order.filled`` for this ``client_order_id`` already covers
    the broker's cumulative filled quantity. A re-delivered fill, or the poller having already
    recorded it, is therefore a no-op. This supports partial fills (each new increment of
    cumulative filled quantity is recorded once).
  * A status change is recorded only when the broker's mapped status differs from the latest
    locally known status (identical rule to ``SyncOrders``), and a redundant ``submitted`` is
    never re-emitted.

Scoping: only orders whose ``client_order_id`` already appears in this sleeve's local log are
acted on, so pre-existing account orders the platform never placed are ignored.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.trade_updates import TradeUpdate
from trading.application.projections import order_statuses_from_events
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.domain.orders import OrderStatus


@dataclass(frozen=True, slots=True)
class ApplyResult:
    recorded_fill: bool
    recorded_status_change: bool
    skipped: bool  # the order is unknown to this sleeve's local log


def _recorded_fill_quantity(events: Sequence[StoredEvent], client_order_id: str) -> Decimal:
    """Total quantity already recorded as filled for an order across all `order.filled`."""
    total = Decimal("0")
    for stored in events:
        event = stored.event
        if event.event_type != EventType.ORDER_FILLED:
            continue
        if event.payload.get("client_order_id") != client_order_id:
            continue
        total += Decimal(str(event.payload.get("quantity", "0")))
    return total


class ApplyTradeUpdate:
    """Records one trade update for one sleeve. Constructed per Alpaca-backed sleeve and
    invoked by the stream adapter as updates arrive."""

    def __init__(
        self,
        *,
        event_store: EventStorePort,
        clock: ClockPort,
        sleeve_id: str,
    ) -> None:
        self._event_store = event_store
        self._clock = clock
        self._sleeve_id = sleeve_id

    async def __call__(self, update: TradeUpdate, *, sleeve_id: str | None = None) -> None:
        # TradeUpdateHandler entry point (the stream calls this). The bound `sleeve_id` from
        # construction is authoritative; the stream passes its own which we ignore. The
        # `ApplyResult` is for tests/observability — discarded on the hot path.
        await self.execute(update)

    async def execute(self, update: TradeUpdate) -> ApplyResult:
        correlation_id = str(uuid4())
        events = await self._event_store.read_all()
        local_statuses = order_statuses_from_events(events, sleeve_id=self._sleeve_id)
        if update.client_order_id not in local_statuses:
            # Not an order this sleeve placed (or not yet recorded) — ignore.
            return ApplyResult(recorded_fill=False, recorded_status_change=False, skipped=True)

        recorded_fill = await self._maybe_record_fill(update, events, correlation_id)
        recorded_status = await self._maybe_record_status(update, local_statuses, correlation_id)
        return ApplyResult(
            recorded_fill=recorded_fill,
            recorded_status_change=recorded_status,
            skipped=False,
        )

    async def _maybe_record_fill(
        self, update: TradeUpdate, events: Sequence[StoredEvent], correlation_id: str
    ) -> bool:
        if not update.is_fill:
            return False
        assert update.fill_price is not None and update.fill_quantity is not None
        # Cumulative-quantity idempotency: the update's filled qty is the broker's running
        # total for the order; only the unrecorded increment is appended. <= 0 means the
        # poller (or an earlier stream frame) already recorded this fill.
        already = _recorded_fill_quantity(events, update.client_order_id)
        increment = update.fill_quantity - already
        if increment <= 0:
            return False
        await self._append(
            f"order-{update.client_order_id}",
            EventType.ORDER_FILLED,
            {
                "client_order_id": update.client_order_id,
                "sleeve_id": self._sleeve_id,
                "symbol": update.symbol,
                "side": update.side.value,
                "price": str(update.fill_price),
                "quantity": str(increment),
                "fees": "0",  # Alpaca equities are commission-free
                "filled_at": (update.filled_at or update.occurred_at).isoformat(),
                "source": "trade_updates_stream",
            },
            correlation_id,
        )
        return True

    async def _maybe_record_status(
        self, update: TradeUpdate, local_statuses: dict[str, str], correlation_id: str
    ) -> bool:
        broker_status = update.status.value
        local_status = local_statuses.get(update.client_order_id)
        if broker_status == local_status:
            return False
        if broker_status == OrderStatus.SUBMITTED.value and local_status in (
            None,
            OrderStatus.SUBMITTED.value,
        ):
            return False
        await self._append(
            f"order-{update.client_order_id}",
            EventType.ORDER_STATUS_CHANGED,
            {
                "client_order_id": update.client_order_id,
                "sleeve_id": self._sleeve_id,
                "status": broker_status,
                "broker_order_id": update.broker_order_id,
                "source": "trade_updates_stream",
            },
            correlation_id,
        )
        return True

    async def _append(
        self, stream_id: str, event_type: str, payload: dict[str, Any], correlation_id: str
    ) -> None:
        await self._event_store.append(
            DomainEvent(
                stream_id=stream_id,
                event_type=event_type,
                payload=payload,
                occurred_at=self._clock.now(),
                correlation_id=correlation_id,
            )
        )
