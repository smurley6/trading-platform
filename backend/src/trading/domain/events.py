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
    # Scheduled-cycle trade mode (auto/manual) arm/disarm, appended by POST /cycle/mode so the
    # operator's choice is DURABLE: a fresh process (the daily cron, any web redeploy/restart)
    # replays the latest one at startup instead of re-arming the CYCLE_TRADE_MODE env default
    # (review F16). Paired with a USER_ACTION event, mirroring the kill-switch idiom.
    CYCLE_MODE_CHANGED = "cycle.mode_changed"
    # Per-source ingest cursor watermark, appended after a successful ingest so a fresh
    # process (the daily cron) resumes where the last run left off instead of re-scanning the
    # whole lookback window (review F1). Replayed at ingest time to rehydrate the cursor.
    INGEST_CURSOR_ADVANCED = "ingest.cursor_advanced"


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
