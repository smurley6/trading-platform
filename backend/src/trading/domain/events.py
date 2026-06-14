"""Immutable domain events — the append-only system of record (spec section 9)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


class EventType:
    """Well-known event type names. New types may be added; existing ones never change."""

    SIGNAL_RECEIVED = "signal.received"
    DECISION_MADE = "decision.made"
    RISK_EVALUATED = "risk.evaluated"
    ORDER_INTENT = "order.intent"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_SUBMIT_FAILED = "order.submit_failed"
    ORDER_STATUS_CHANGED = "order.status_changed"
    ORDER_FILLED = "order.filled"
    PORTFOLIO_SNAPSHOT = "portfolio.snapshot"
    RECONCILIATION_DRIFT = "reconciliation.drift"
    RECONCILIATION_COMPLETED = "reconciliation.completed"
    KILL_SWITCH_CHANGED = "risk.kill_switch_changed"
    USER_ACTION = "user.action"
    LIVE_READINESS_ASSESSED = "live.readiness_assessed"
    SLEEVE_MODE_CHANGED = "sleeve.mode_changed"


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """A fact that happened. Events are appended, never updated or deleted."""

    stream_id: str
    event_type: str
    payload: dict[str, Any]
    occurred_at: datetime
    correlation_id: str
    schema_version: int = 1
    event_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """An event as persisted: assigned a gap-free per-stream sequence number."""

    event: DomainEvent
    seq: int
    recorded_at: datetime
