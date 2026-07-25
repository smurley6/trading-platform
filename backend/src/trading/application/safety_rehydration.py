"""Rehydrate process-local safety state from the append-only event log at startup (review F2/F3).

The kill switch and the drawdown circuit breaker both live in per-process memory. A fresh
process — the daily 13:00 UTC cron, or any web redeploy/restart — used to boot with an
*empty* safety state: the kill switch defaulted to disengaged (so an operator's emergency stop
never reached the process that trades, F2) and peak equity defaulted to ``0`` (so drawdown was
always computed as ``0`` and the ``max_drawdown_pct`` breaker could never trip, F3).

This module derives both facts by replaying existing event types — it appends nothing, reads no
new schema, and is point-in-time correct: the kill switch takes the LATEST
``KILL_SWITCH_CHANGED`` state and peak equity takes the MAXIMUM equity ever recorded in a
``PORTFOLIO_SNAPSHOT``. The composition root seeds the ``KillSwitch`` and each ``EvaluateSignals``
from the result; both then keep updating in-process as before.
"""

from dataclasses import dataclass
from decimal import Decimal

from trading.application.ports.event_store import EventStorePort
from trading.domain.events import EventType


@dataclass(frozen=True, slots=True)
class SafetyState:
    """Process-local safety state recovered from the durable log."""

    kill_switch_engaged: bool
    peak_equity: Decimal


async def rehydrate_safety_state(event_store: EventStorePort) -> SafetyState:
    """Replay the log for the latest kill-switch state and the high-water equity mark.

    ``read_all`` yields events in recording order, so the last ``KILL_SWITCH_CHANGED`` wins
    (the operator's most recent decision). Peak equity is the running maximum across every
    ``PORTFOLIO_SNAPSHOT``, matching the in-process ``max(peak, current)`` update it seeds.
    """
    engaged = False
    peak_equity = Decimal("0")
    for stored in await event_store.read_all():
        event_type = stored.event.event_type
        if event_type == EventType.KILL_SWITCH_CHANGED:
            engaged = bool(stored.event.payload["engaged"])
        elif event_type == EventType.PORTFOLIO_SNAPSHOT:
            equity = Decimal(str(stored.event.payload["equity"]))
            if equity > peak_equity:
                peak_equity = equity
    return SafetyState(kill_switch_engaged=engaged, peak_equity=peak_equity)
