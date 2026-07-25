"""Rehydrating process-local safety state from the append-only log (review F2/F3).

These exercise the derivation directly on a shared ``InMemoryEventStore`` (the durable log a
fresh process reads); the composition-root wiring and cross-process behaviour are covered in the
Postgres integration suite.
"""

from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.safety_rehydration import rehydrate_safety_state
from trading.domain.events import DomainEvent, EventType


def _kill_switch(clock: FixedClock, *, engaged: bool) -> DomainEvent:
    return DomainEvent(
        stream_id="risk",
        event_type=EventType.KILL_SWITCH_CHANGED,
        payload={"engaged": engaged},
        occurred_at=clock.now(),
        correlation_id="corr",
    )


def _snapshot(clock: FixedClock, *, equity: str) -> DomainEvent:
    return DomainEvent(
        stream_id="sleeve-political",
        event_type=EventType.PORTFOLIO_SNAPSHOT,
        payload={
            "sleeve_id": "sleeve-political",
            "cash": equity,
            "equity": equity,
            "positions": {},
        },
        occurred_at=clock.now(),
        correlation_id="corr",
    )


async def test_empty_log_yields_defaults(event_store) -> None:  # type: ignore[no-untyped-def]
    state = await rehydrate_safety_state(event_store)
    assert state.kill_switch_engaged is False
    assert state.peak_equity == Decimal("0")


async def test_latest_kill_switch_event_wins(clock: FixedClock, event_store) -> None:  # type: ignore[no-untyped-def]
    # Engage, disengage, re-engage: the operator's MOST RECENT decision must be the one restored.
    await event_store.append(_kill_switch(clock, engaged=True))
    await event_store.append(_kill_switch(clock, engaged=False))
    await event_store.append(_kill_switch(clock, engaged=True))

    state = await rehydrate_safety_state(event_store)
    assert state.kill_switch_engaged is True


async def test_disengaged_last_stays_disengaged(clock: FixedClock, event_store) -> None:  # type: ignore[no-untyped-def]
    await event_store.append(_kill_switch(clock, engaged=True))
    await event_store.append(_kill_switch(clock, engaged=False))

    state = await rehydrate_safety_state(event_store)
    assert state.kill_switch_engaged is False


async def test_peak_equity_is_the_max_snapshot(clock: FixedClock, event_store) -> None:  # type: ignore[no-untyped-def]
    # Equity climbs then falls; the high-water mark (not the latest value) is the peak.
    await event_store.append(_snapshot(clock, equity="8000"))
    await event_store.append(_snapshot(clock, equity="12500.50"))
    await event_store.append(_snapshot(clock, equity="9000"))

    state = await rehydrate_safety_state(event_store)
    assert state.peak_equity == Decimal("12500.50")


async def test_both_are_recovered_together(clock: FixedClock, event_store) -> None:  # type: ignore[no-untyped-def]
    await event_store.append(_snapshot(clock, equity="10000"))
    await event_store.append(_kill_switch(clock, engaged=True))
    await event_store.append(_snapshot(clock, equity="7000"))

    state = await rehydrate_safety_state(event_store)
    assert state.kill_switch_engaged is True
    assert state.peak_equity == Decimal("10000")
