"""CancelOrder use case: a manual override that cancels an open order at the broker.

The broker is the source of execution truth (FR-REC), so cancellation is delegated to it and
its resulting state is recorded. Every cancellation is auditable: the manual request is logged
as a user action and the outcome as an order status change (spec section 9).
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from trading.application.ports.broker import BrokerOrder, BrokerPort, OrderNotFoundError
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.projections import order_sleeve
from trading.domain.events import DomainEvent, EventType

logger = logging.getLogger(__name__)

_USER_ACTION_STREAM = "user-actions"


class CancelFailedError(Exception):
    """The broker refused or failed the cancellation (distinct from not-found)."""


@dataclass(frozen=True, slots=True)
class CancelResult:
    correlation_id: str
    order: BrokerOrder
    sleeve_id: str


class CancelOrder:
    def __init__(
        self,
        *,
        brokers: Mapping[str, BrokerPort],
        event_store: EventStorePort,
        clock: ClockPort,
    ) -> None:
        self._brokers = brokers
        self._event_store = event_store
        self._clock = clock

    async def execute(self, *, client_order_id: str, sleeve_id: str | None = None) -> CancelResult:
        correlation_id = str(uuid4())
        # Resolve the owning sleeve from the order's own recorded lifecycle. Do NOT silently
        # default an unknown order into a sleeve's ledger (review F27): a manual Alpaca-dashboard
        # order the platform never placed would otherwise be stamped into sleeve-political's log,
        # after which SyncOrders' "known" scoping admits its fills and attributes a human trade to
        # the algo (and a foreign partial SELL can wedge order sync, review F7). Adopting an
        # unknown order requires an EXPLICIT sleeve param; without one, this is a 404.
        resolved = sleeve_id or await self._resolve_sleeve(client_order_id)
        if resolved is None:
            raise OrderNotFoundError(
                f"no local record of order {client_order_id}; pass an explicit sleeve_id to "
                "adopt an order the platform did not place"
            )
        broker = self._brokers.get(resolved)
        if broker is None:
            raise OrderNotFoundError(f"unknown sleeve '{resolved}' for order {client_order_id}")

        # Record the manual cancel request before acting (spec section 9: every user action).
        await self._append(
            _USER_ACTION_STREAM,
            EventType.USER_ACTION,
            {"action": "cancel_order", "client_order_id": client_order_id, "sleeve_id": resolved},
            correlation_id,
        )

        try:
            order = await broker.cancel_order(client_order_id=client_order_id)
        except OrderNotFoundError:
            raise
        except Exception as exc:  # broker/transport failure — surface it, never swallow
            logger.exception(
                "order cancellation failed",
                extra={"correlation_id": correlation_id, "client_order_id": client_order_id},
            )
            raise CancelFailedError(f"cancel failed for {client_order_id}: {exc}") from exc

        await self._append(
            f"order-{client_order_id}",
            EventType.ORDER_STATUS_CHANGED,
            {
                "client_order_id": client_order_id,
                "sleeve_id": resolved,
                "status": order.status.value,
                "broker_order_id": order.broker_order_id,
            },
            correlation_id,
        )
        return CancelResult(correlation_id=correlation_id, order=order, sleeve_id=resolved)

    async def _resolve_sleeve(self, client_order_id: str) -> str | None:
        events = await self._event_store.read_all()
        return order_sleeve(events, client_order_id)

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
