"""PlaceOrder use case.

Write-ahead-of-action (spec section 9): the order intent is durably persisted as an event
BEFORE the broker call, so a crash leaves a recoverable record, not a gap. Reconciliation
detects intents that never reached the broker.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading.application.ports.broker import BrokerOrder, BrokerPort
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class OrderSubmissionError(Exception):
    """Broker submission failed after the intent was durably recorded."""


@dataclass(frozen=True, slots=True)
class PlaceOrderRequest:
    sleeve_id: str
    decision_id: str
    client_order_id: str  # idempotency key (FR-EXE-3)
    symbol: str
    side: OrderSide
    quantity: Decimal
    correlation_id: str
    order_type: OrderType = OrderType.MARKET


class PlaceOrder:
    def __init__(
        self, *, broker: BrokerPort, event_store: EventStorePort, clock: ClockPort
    ) -> None:
        self._broker = broker
        self._event_store = event_store
        self._clock = clock

    async def execute(self, request: PlaceOrderRequest) -> BrokerOrder:
        stream_id = f"order-{request.client_order_id}"
        base: dict[str, Any] = {
            "client_order_id": request.client_order_id,
            "decision_id": request.decision_id,
            "sleeve_id": request.sleeve_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": str(request.quantity),
            "order_type": request.order_type.value,
        }

        await self._append(stream_id, EventType.ORDER_INTENT, base, request.correlation_id)

        try:
            broker_order = await self._broker.submit_order(
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                order_type=request.order_type,
            )
        except Exception as exc:
            logger.exception(
                "order submission failed",
                extra={"correlation_id": request.correlation_id},
            )
            await self._append(
                stream_id,
                EventType.ORDER_SUBMIT_FAILED,
                {**base, "error": str(exc)},
                request.correlation_id,
            )
            raise OrderSubmissionError(
                f"submission failed for {request.client_order_id}: {exc}"
            ) from exc

        await self._append(
            stream_id,
            EventType.ORDER_SUBMITTED,
            {**base, "broker_order_id": broker_order.broker_order_id},
            request.correlation_id,
        )
        for fill in broker_order.fills:
            await self._append(
                stream_id,
                EventType.ORDER_FILLED,
                {
                    "client_order_id": request.client_order_id,
                    "sleeve_id": request.sleeve_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "price": str(fill.price),
                    "quantity": str(fill.quantity),
                    "fees": str(fill.fees),
                    "filled_at": fill.filled_at.isoformat(),
                },
                request.correlation_id,
            )
        if broker_order.status is not OrderStatus.SUBMITTED:
            await self._append(
                stream_id,
                EventType.ORDER_STATUS_CHANGED,
                {
                    "client_order_id": request.client_order_id,
                    "sleeve_id": request.sleeve_id,
                    "status": broker_order.status.value,
                    "broker_order_id": broker_order.broker_order_id,
                },
                request.correlation_id,
            )
        return broker_order

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
