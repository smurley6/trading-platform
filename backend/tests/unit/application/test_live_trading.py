"""Live readiness preflight + per-sleeve opt-in (spec section 11)."""

from decimal import Decimal

import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.live_trading import (
    AssessLiveReadiness,
    LiveTradingError,
    RequestLiveTrading,
)
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch, RiskLimits
from trading.application.signal_recording import record_signals
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.events import EventType
from trading.domain.provenance import Provenance
from trading.domain.sleeves import TradingMode
from trading.infrastructure.persistence.memory import InMemoryEventStore

_TIGHT = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("5"),
    max_position_concentration_pct=Decimal("15"),
    max_drawdown_pct=Decimal("10"),
)
_CEILINGS = _TIGHT


def _assessor(
    *,
    allow_live: bool,
    live_configured: bool,
    configured: RiskLimits,
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
) -> AssessLiveReadiness:
    return AssessLiveReadiness(
        allow_live=allow_live,
        live_broker_configured=live_configured,
        configured_limits=configured,
        live_ceilings=_CEILINGS,
        kill_switch=kill_switch,
        reconcilers={"sleeve-test": reconcile},
        event_store=event_store,
        clock=clock,
    )


async def test_ready_when_all_checks_pass(
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
) -> None:
    assessor = _assessor(
        allow_live=True,
        live_configured=True,
        configured=_TIGHT,
        kill_switch=kill_switch,
        reconcile=reconcile,
        event_store=event_store,
        clock=clock,
    )

    report = await assessor.assess("sleeve-test")

    assert report.ready
    assert {c.name for c in report.checks} == {
        "master_switch",
        "live_credentials",
        "risk_caps_hardened",
        "kill_switch_operational",
        "reconciliation_clean",
        "data_provenance_real",
    }
    # The kill-switch self-test restored the original (disengaged) state.
    assert kill_switch.engaged is False
    # The assessment is recorded for audit.
    recorded = [
        s
        for s in await event_store.read_all()
        if s.event.event_type == EventType.LIVE_READINESS_ASSESSED
    ]
    assert recorded and recorded[-1].event.payload["ready"] is True


async def test_not_ready_lists_failures(
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
) -> None:
    loose = RiskLimits(
        max_trade_pct_of_sleeve=Decimal("50"),  # exceeds ceiling
        max_position_concentration_pct=Decimal("90"),
        max_drawdown_pct=Decimal("90"),
    )
    assessor = _assessor(
        allow_live=False,
        live_configured=False,
        configured=loose,
        kill_switch=kill_switch,
        reconcile=reconcile,
        event_store=event_store,
        clock=clock,
    )

    report = await assessor.assess("sleeve-test")

    assert not report.ready
    failed = {c.name for c in report.checks if not c.passed}
    assert {"master_switch", "live_credentials", "risk_caps_hardened"} <= failed


async def test_not_ready_when_synthetic_signal_in_window(
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
    make_signal: SignalFactory,
) -> None:
    # A SYNTHETIC/placeholder signal in the corroboration window must block go-live, even when
    # every other check passes.
    await record_signals(
        event_store=event_store,
        clock=clock,
        signals=[make_signal(provenance=Provenance.SYNTHETIC)],
        correlation_id="seed",
    )
    assessor = _assessor(
        allow_live=True,
        live_configured=True,
        configured=_TIGHT,
        kill_switch=kill_switch,
        reconcile=reconcile,
        event_store=event_store,
        clock=clock,
    )

    report = await assessor.assess("sleeve-test")

    assert not report.ready
    provenance_check = next(c for c in report.checks if c.name == "data_provenance_real")
    assert not provenance_check.passed


async def test_enable_requires_confirmation(
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
    sleeve_registry: SleeveRegistry,
) -> None:
    request = RequestLiveTrading(
        registry=sleeve_registry,
        readiness=_assessor(
            allow_live=True,
            live_configured=True,
            configured=_TIGHT,
            kill_switch=kill_switch,
            reconcile=reconcile,
            event_store=event_store,
            clock=clock,
        ),
        event_store=event_store,
        clock=clock,
    )

    with pytest.raises(LiveTradingError, match="confirmation"):
        await request.enable("sleeve-test", confirm=False)
    assert sleeve_registry.mode("sleeve-test") is TradingMode.PAPER


async def test_enable_blocked_when_not_ready(
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
    sleeve_registry: SleeveRegistry,
) -> None:
    request = RequestLiveTrading(
        registry=sleeve_registry,
        readiness=_assessor(
            allow_live=False,  # master off -> not ready
            live_configured=False,
            configured=_TIGHT,
            kill_switch=kill_switch,
            reconcile=reconcile,
            event_store=event_store,
            clock=clock,
        ),
        event_store=event_store,
        clock=clock,
    )

    with pytest.raises(LiveTradingError, match="not ready"):
        await request.enable("sleeve-test", confirm=True)
    assert sleeve_registry.mode("sleeve-test") is TradingMode.PAPER


async def test_enable_and_disable_flip_mode_and_record(
    kill_switch: KillSwitch,
    reconcile: Reconcile,
    event_store: InMemoryEventStore,
    clock: FixedClock,
    sleeve_registry: SleeveRegistry,
) -> None:
    request = RequestLiveTrading(
        registry=sleeve_registry,
        readiness=_assessor(
            allow_live=True,
            live_configured=True,
            configured=_TIGHT,
            kill_switch=kill_switch,
            reconcile=reconcile,
            event_store=event_store,
            clock=clock,
        ),
        event_store=event_store,
        clock=clock,
    )

    report = await request.enable("sleeve-test", confirm=True)
    assert report.ready
    assert sleeve_registry.mode("sleeve-test") is TradingMode.LIVE

    await request.disable("sleeve-test")
    assert sleeve_registry.mode("sleeve-test") is TradingMode.PAPER

    mode_events = [
        s.event.payload["mode"]
        for s in await event_store.read_all()
        if s.event.event_type == EventType.SLEEVE_MODE_CHANGED
    ]
    assert mode_events == ["live", "paper"]
