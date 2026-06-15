"""SyncOrders use case: pull the broker's order state and emit the lifecycle events the local
log is missing (FR-REC, FR-EXE).

The in-process PaperBroker fills synchronously at submission, so PlaceOrder records the fill
immediately. The real AlpacaBroker fills ASYNCHRONOUSLY after submission, so a later fill is
never recorded locally — it only ever surfaces as reconciliation drift. This use case closes
that loop: per Alpaca-backed sleeve it diffs the broker's orders against the local event log
and appends the missing `order.filled` / `order.status_changed` events so the read models
(positions, P&L, order status) reflect what actually executed.

Idempotency: the local picture is rebuilt from the event log on every run. An `order.filled`
is emitted only for an order with no fill already recorded locally; an `order.status_changed`
is emitted only when the broker's status differs from the latest local status. A second run
therefore finds the gap already closed and emits nothing.

Scoping: only orders whose `client_order_id` already appears in this sleeve's local log are
acted on, so pre-existing account orders the platform never placed are never imported.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from trading.application.ports.broker import BrokerOrder, BrokerPort
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.projections import order_statuses_from_events
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.domain.orders import OrderStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyncReport:
    correlation_id: str
    sleeve_id: str
    fills_emitted: int
    status_changes_emitted: int
    checked_orders: int

    @property
    def emitted_nothing(self) -> bool:
        return self.fills_emitted == 0 and self.status_changes_emitted == 0


def _filled_client_order_ids(events: Sequence[StoredEvent], sleeve_id: str) -> set[str]:
    """Client order ids that already have an `order.filled` recorded for this sleeve."""
    filled: set[str] = set()
    for stored in events:
        event = stored.event
        if event.event_type != EventType.ORDER_FILLED:
            continue
        if event.payload.get("sleeve_id") != sleeve_id:
            continue
        client_order_id = event.payload.get("client_order_id")
        if client_order_id is not None:
            filled.add(str(client_order_id))
    return filled


class SyncOrders:
    def __init__(
        self,
        *,
        broker: BrokerPort,
        event_store: EventStorePort,
        clock: ClockPort,
        sleeve_id: str,
    ) -> None:
        self._broker = broker
        self._event_store = event_store
        self._clock = clock
        self._sleeve_id = sleeve_id

    async def execute(self) -> SyncReport:
        correlation_id = str(uuid4())
        events = await self._event_store.read_all()
        # Local picture, scoped to this sleeve: latest known status per client_order_id and
        # which orders already have a fill recorded. Rebuilt every run -> idempotent.
        local_statuses = order_statuses_from_events(events, sleeve_id=self._sleeve_id)
        already_filled = _filled_client_order_ids(events, self._sleeve_id)
        # Only act on orders the local log already knows about (avoid importing unrelated
        # pre-existing account orders).
        known = set(local_statuses)

        fills_emitted = 0
        status_changes_emitted = 0
        checked = 0
        for order in await self._broker.list_orders():
            if order.client_order_id not in known:
                continue
            checked += 1
            fills_emitted += await self._sync_fills(order, already_filled, correlation_id)
            status_changes_emitted += await self._sync_status(order, local_statuses, correlation_id)

        return SyncReport(
            correlation_id=correlation_id,
            sleeve_id=self._sleeve_id,
            fills_emitted=fills_emitted,
            status_changes_emitted=status_changes_emitted,
            checked_orders=checked,
        )

    async def _sync_fills(
        self, order: BrokerOrder, already_filled: set[str], correlation_id: str
    ) -> int:
        if not order.fills or order.client_order_id in already_filled:
            return 0
        for fill in order.fills:
            await self._append(
                f"order-{order.client_order_id}",
                EventType.ORDER_FILLED,
                {
                    "client_order_id": order.client_order_id,
                    "sleeve_id": self._sleeve_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "price": str(fill.price),
                    "quantity": str(fill.quantity),
                    "fees": str(fill.fees),
                    "filled_at": fill.filled_at.isoformat(),
                },
                correlation_id,
            )
        return len(order.fills)

    async def _sync_status(
        self, order: BrokerOrder, local_statuses: dict[str, str], correlation_id: str
    ) -> int:
        broker_status = order.status.value
        local_status = local_statuses.get(order.client_order_id)
        # `submitted` is the baseline PlaceOrder records; the broker reporting `submitted`
        # back is not a change worth re-emitting.
        if broker_status == local_status:
            return 0
        if broker_status == OrderStatus.SUBMITTED.value and local_status in (
            None,
            OrderStatus.SUBMITTED.value,
        ):
            return 0
        await self._append(
            f"order-{order.client_order_id}",
            EventType.ORDER_STATUS_CHANGED,
            {
                "client_order_id": order.client_order_id,
                "sleeve_id": self._sleeve_id,
                "status": broker_status,
                "broker_order_id": order.broker_order_id,
            },
            correlation_id,
        )
        return 1

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
