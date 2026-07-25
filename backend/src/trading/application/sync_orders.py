"""SyncOrders use case: pull the broker's order state and emit the lifecycle events the local
log is missing (FR-REC, FR-EXE).

The in-process PaperBroker fills synchronously at submission, so PlaceOrder records the fill
immediately. The real AlpacaBroker fills ASYNCHRONOUSLY after submission, so a later fill is
never recorded locally — it only ever surfaces as reconciliation drift. This use case closes
that loop: per Alpaca-backed sleeve it diffs the broker's orders against the local event log
and appends the missing `order.filled` / `order.status_changed` events so the read models
(positions, P&L, order status) reflect what actually executed.

Idempotency (CUMULATIVE, matching `ApplyTradeUpdate`): the local picture is rebuilt from the
event log on every run. The broker reports a running TOTAL filled quantity, so each run records
only `broker_filled_qty - locally_recorded_qty` when positive — a partial that later completes
records the remaining increment (review F8), and a second run with nothing new emits nothing.
An `order.status_changed` is emitted only when the broker's status differs from the latest local
status. Fills are appended under a compare-and-set (`expected_version`) so a concurrent real-time
stream write for the same fill is detected and never doubles the position (review F9).

Ordering & isolation: orders are applied oldest-first (`submitted_at` ascending) so a SELL is
never processed before its BUY on a newest-first broker list, and each order is wrapped so one
bad order cannot wedge the whole sync loop (review F7).

Scoping: only orders whose `client_order_id` already appears in this sleeve's local log are
acted on, so pre-existing account orders the platform never placed are never imported.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from trading.application.ports.broker import BrokerOrder, BrokerPort
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort, ExpectedVersionError
from trading.application.projections import order_statuses_from_events, stream_version
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.domain.orders import OrderStatus

logger = logging.getLogger(__name__)

# Bound on the compare-and-set fill re-read loop (review F9). A conflict means another writer
# (the real-time stream, or an overlapping poll) advanced the order stream; we re-read and
# re-decide. A tiny handful of attempts is plenty — real contention is one racing writer.
_FILL_CAS_ATTEMPTS = 5

# Stable sort key: submitted_at ascending, Nones (paper / older payloads) sorted first so their
# original list order is preserved (Python's sort is stable). Any tz-aware minimum works.
_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SyncReport:
    correlation_id: str
    sleeve_id: str
    fills_emitted: int
    status_changes_emitted: int
    checked_orders: int
    failed_orders: int = 0

    @property
    def emitted_nothing(self) -> bool:
        return self.fills_emitted == 0 and self.status_changes_emitted == 0


def _recorded_fill_quantity(
    events: Sequence[StoredEvent], client_order_id: str, sleeve_id: str
) -> Decimal:
    """Total quantity already recorded as filled for an order across all `order.filled`.

    This is the CUMULATIVE model (matches `ApplyTradeUpdate`): the broker reports a running
    total filled quantity, so only the unrecorded increment should be appended. The previous
    boolean `client_order_id in already_filled` gate dropped every partial after the first
    (review F8), permanently understating positions.
    """
    total = Decimal("0")
    for stored in events:
        event = stored.event
        if event.event_type != EventType.ORDER_FILLED:
            continue
        payload = event.payload
        if payload.get("sleeve_id") != sleeve_id:
            continue
        if payload.get("client_order_id") != client_order_id:
            continue
        total += Decimal(str(payload.get("quantity", "0")))
    return total


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
        # Local picture, scoped to this sleeve: latest known status per client_order_id. Rebuilt
        # every run -> idempotent. Only act on orders the local log already knows about (avoid
        # importing unrelated pre-existing account orders).
        local_statuses = order_statuses_from_events(events, sleeve_id=self._sleeve_id)
        known = set(local_statuses)

        # Apply oldest-first: a newest-first broker list (Alpaca) would otherwise process a SELL
        # before its BUY (review F7). Stable sort keeps insertion order for orders without a
        # submitted_at (the paper simulator is already insertion-ordered).
        orders = sorted(await self._broker.list_orders(), key=lambda o: o.submitted_at or _EPOCH)

        fills_emitted = 0
        status_changes_emitted = 0
        checked = 0
        failed = 0
        for order in orders:
            if order.client_order_id not in known:
                continue
            checked += 1
            # Per-order isolation: one malformed/oversell order must NOT wedge the whole loop
            # (and thereby every following order) for the sleeve (review F7). Log and move on.
            try:
                fills_emitted += await self._sync_fills(order, correlation_id)
                status_changes_emitted += await self._sync_status(
                    order, local_statuses, correlation_id
                )
            except Exception:
                failed += 1
                logger.exception(
                    "order sync failed for client_order_id=%s sleeve=%s (skipping this order)",
                    order.client_order_id,
                    self._sleeve_id,
                )

        return SyncReport(
            correlation_id=correlation_id,
            sleeve_id=self._sleeve_id,
            fills_emitted=fills_emitted,
            status_changes_emitted=status_changes_emitted,
            checked_orders=checked,
            failed_orders=failed,
        )

    async def _sync_fills(self, order: BrokerOrder, correlation_id: str) -> int:
        """Record the unrecorded cumulative-fill increment for one order, exactly once.

        The broker reports a running total filled quantity (Alpaca collapses it to a single
        aggregate `Fill` at the average price). We append `broker_filled_qty -
        locally_recorded_qty` when positive (review F8), under a compare-and-set append so a
        concurrent real-time-stream write for the same fill cannot double the position (review
        F9): on a version conflict we re-read and re-decide (the increment is then <= 0).
        """
        if not order.fills:
            return 0
        broker_qty = sum((f.quantity for f in order.fills), Decimal("0"))
        stream_id = f"order-{order.client_order_id}"
        representative = order.fills[-1]
        for _ in range(_FILL_CAS_ATTEMPTS):
            events = await self._event_store.read_all()
            recorded = _recorded_fill_quantity(events, order.client_order_id, self._sleeve_id)
            increment = broker_qty - recorded
            if increment <= 0:
                return 0
            try:
                await self._append(
                    stream_id,
                    EventType.ORDER_FILLED,
                    {
                        "client_order_id": order.client_order_id,
                        "sleeve_id": self._sleeve_id,
                        "symbol": representative.symbol,
                        "side": representative.side.value,
                        "price": str(representative.price),
                        "quantity": str(increment),
                        # Alpaca equities are commission-free; the aggregate fill carries fees=0.
                        "fees": str(representative.fees),
                        "filled_at": representative.filled_at.isoformat(),
                        "source": "order_sync",
                    },
                    correlation_id,
                    expected_version=stream_version(events, stream_id),
                )
                return 1
            except ExpectedVersionError:
                continue  # a concurrent writer advanced the stream; re-read and re-decide
        return 0

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
        self,
        stream_id: str,
        event_type: str,
        payload: dict[str, Any],
        correlation_id: str,
        *,
        expected_version: int | None = None,
    ) -> None:
        await self._event_store.append(
            DomainEvent(
                stream_id=stream_id,
                event_type=event_type,
                payload=payload,
                occurred_at=self._clock.now(),
                correlation_id=correlation_id,
            ),
            expected_version=expected_version,
        )
