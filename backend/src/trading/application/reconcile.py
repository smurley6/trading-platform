"""Reconciliation (FR-REC-1): the broker is the source of execution truth.

Diffs local event-log projections against the broker's orders and positions; any drift is
emitted as reconciliation events and returned for the UI.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from trading.application.ports.broker import BrokerPort
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.projections import order_statuses_from_events, positions_from_events
from trading.domain.events import DomainEvent, EventType, StoredEvent

_STREAM_ID = "reconciliation"


def _latest_recorded_drifts(
    events: Sequence[StoredEvent], sleeve_id: str | None
) -> dict[tuple[str, str], tuple[str, str]]:
    """The most recently recorded (local, broker) values per (kind, key) drift, this sleeve only.

    Lets `Reconcile` suppress re-appending a drift identical to the one already on the log
    (review F14) while still reporting the current drift set.
    """
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    for stored in events:
        event = stored.event
        if event.event_type != EventType.RECONCILIATION_DRIFT:
            continue
        payload = event.payload
        if payload.get("sleeve_id") != sleeve_id:
            continue
        key = (str(payload.get("kind")), str(payload.get("key")))
        latest[key] = (str(payload.get("local")), str(payload.get("broker")))
    return latest


@dataclass(frozen=True, slots=True)
class Drift:
    kind: str  # "position" | "order"
    key: str  # symbol or client_order_id
    local: str
    broker: str


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    correlation_id: str
    sleeve_id: str | None
    drifts: tuple[Drift, ...]
    checked_positions: int
    checked_orders: int

    @property
    def clean(self) -> bool:
        return not self.drifts


class Reconcile:
    def __init__(
        self,
        *,
        broker: BrokerPort,
        event_store: EventStorePort,
        clock: ClockPort,
        sleeve_id: str | None = None,
    ) -> None:
        self._broker = broker
        self._event_store = event_store
        self._clock = clock
        self._sleeve_id = sleeve_id

    async def execute(self) -> ReconciliationReport:
        correlation_id = str(uuid4())
        events = await self._event_store.read_all()
        local_positions = positions_from_events(events, sleeve_id=self._sleeve_id)
        local_orders = order_statuses_from_events(events, sleeve_id=self._sleeve_id)

        broker_positions = {p.symbol: p.quantity for p in await self._broker.list_positions()}
        broker_orders = {
            o.client_order_id: o.status.value for o in await self._broker.list_orders()
        }

        drifts: list[Drift] = []
        for symbol in sorted(set(local_positions) | set(broker_positions)):
            local_qty = local_positions.get(symbol, Decimal("0"))
            broker_qty = broker_positions.get(symbol, Decimal("0"))
            if local_qty != broker_qty:
                drifts.append(Drift("position", symbol, str(local_qty), str(broker_qty)))

        for client_order_id in sorted(set(local_orders) | set(broker_orders)):
            local_status = local_orders.get(client_order_id, "missing")
            broker_status = broker_orders.get(client_order_id, "missing")
            if local_status == "submit_failed" and broker_status == "missing":
                continue  # already recorded as a failure, nothing to reconcile
            if local_status != broker_status:
                drifts.append(Drift("order", client_order_id, local_status, broker_status))

        # Dedupe repeated identical drift: without fill ingestion the same drift recurred every
        # run, appending an identical event each time and burying real changes (review F14). Only
        # append a drift that is NEW or CHANGED vs the last one recorded for this (kind, key); the
        # returned report still reflects the CURRENT drift, so readiness/`clean` is unaffected.
        last_recorded = _latest_recorded_drifts(events, self._sleeve_id)
        for drift in drifts:
            if last_recorded.get((drift.kind, drift.key)) == (drift.local, drift.broker):
                continue
            await self._append(
                EventType.RECONCILIATION_DRIFT,
                {
                    "sleeve_id": self._sleeve_id,
                    "kind": drift.kind,
                    "key": drift.key,
                    "local": drift.local,
                    "broker": drift.broker,
                },
                correlation_id,
            )
        await self._append(
            EventType.RECONCILIATION_COMPLETED,
            {
                "sleeve_id": self._sleeve_id,
                "drift_count": len(drifts),
                "checked_positions": len(set(local_positions) | set(broker_positions)),
                "checked_orders": len(set(local_orders) | set(broker_orders)),
            },
            correlation_id,
        )
        return ReconciliationReport(
            correlation_id=correlation_id,
            sleeve_id=self._sleeve_id,
            drifts=tuple(drifts),
            checked_positions=len(set(local_positions) | set(broker_positions)),
            checked_orders=len(set(local_orders) | set(broker_orders)),
        )

    async def _append(self, event_type: str, payload: dict[str, Any], correlation_id: str) -> None:
        stream_id = _STREAM_ID if self._sleeve_id is None else f"{_STREAM_ID}-{self._sleeve_id}"
        await self._event_store.append(
            DomainEvent(
                stream_id=stream_id,
                event_type=event_type,
                payload=payload,
                occurred_at=self._clock.now(),
                correlation_id=correlation_id,
            )
        )
