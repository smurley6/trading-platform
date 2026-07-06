"""Live-trading readiness preflight and per-sleeve opt-in (spec section 11).

Going live is gated, explicit, and audited. `AssessLiveReadiness` runs the §11 checklist —
master switch, live credentials, hardened risk caps, a functional kill-switch test, and a
clean reconciliation — and records the assessment. `RequestLiveTrading` flips a sleeve to
live ONLY when every check passes and the operator explicitly confirms; the mode change and
the user action are both recorded as immutable events. Returning to paper is always allowed.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from trading.application.data_integrity import execution_barred
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.projections import signals_from_events
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch, RiskContext, RiskGate, RiskLimits
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.decisions import RiskVerdict
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide
from trading.domain.sleeves import TradingMode


@dataclass(frozen=True, slots=True)
class LiveCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class LiveReadinessReport:
    sleeve_id: str
    ready: bool
    checks: tuple[LiveCheck, ...]


class LiveTradingError(Exception):
    pass


def _caps_within(configured: RiskLimits, ceilings: RiskLimits) -> bool:
    return (
        configured.max_trade_pct_of_sleeve <= ceilings.max_trade_pct_of_sleeve
        and configured.max_position_concentration_pct <= ceilings.max_position_concentration_pct
        and configured.max_drawdown_pct <= ceilings.max_drawdown_pct
    )


class AssessLiveReadiness:
    def __init__(
        self,
        *,
        allow_live: bool,
        live_broker_configured: bool,
        configured_limits: RiskLimits,
        live_ceilings: RiskLimits,
        kill_switch: KillSwitch,
        reconcilers: dict[str, Reconcile],
        event_store: EventStorePort,
        clock: ClockPort,
        window_days: int = 21,
    ) -> None:
        self._allow_live = allow_live
        self._live_broker_configured = live_broker_configured
        self._configured_limits = configured_limits
        self._live_ceilings = live_ceilings
        self._kill_switch = kill_switch
        self._reconcilers = reconcilers
        self._event_store = event_store
        self._clock = clock
        self._window_days = window_days

    async def assess(self, sleeve_id: str) -> LiveReadinessReport:
        checks: list[LiveCheck] = [
            LiveCheck(
                "master_switch",
                self._allow_live,
                "ALLOW_LIVE enabled" if self._allow_live else "ALLOW_LIVE is off",
            ),
            LiveCheck(
                "live_credentials",
                self._live_broker_configured,
                "live broker configured"
                if self._live_broker_configured
                else "no live Alpaca credentials configured",
            ),
            self._caps_check(),
            self._kill_switch_check(),
            await self._reconciliation_check(sleeve_id),
            await self._data_provenance_check(),
        ]
        ready = all(c.passed for c in checks)
        await self._record(sleeve_id, ready, checks)
        return LiveReadinessReport(sleeve_id=sleeve_id, ready=ready, checks=tuple(checks))

    def _caps_check(self) -> LiveCheck:
        ok = _caps_within(self._configured_limits, self._live_ceilings)
        return LiveCheck(
            "risk_caps_hardened",
            ok,
            "risk caps are within live ceilings"
            if ok
            else (
                f"caps exceed live ceilings (trade≤{self._live_ceilings.max_trade_pct_of_sleeve}%, "
                f"concentration≤{self._live_ceilings.max_position_concentration_pct}%, "
                f"drawdown≤{self._live_ceilings.max_drawdown_pct}%)"
            ),
        )

    def _kill_switch_check(self) -> LiveCheck:
        # Functionally verify the kill switch halts the risk gate, then restore prior state.
        original = self._kill_switch.engaged
        self._kill_switch.engage()
        probe = RiskGate(self._live_ceilings, self._kill_switch).evaluate(
            OrderSide.BUY,
            Decimal("1"),
            RiskContext(
                sleeve_equity=Decimal("1"),
                peak_equity=Decimal("1"),
                current_position_value=Decimal("0"),
                price=Decimal("1"),
            ),
        )
        if not original:
            self._kill_switch.disengage()
        halts = probe.verdict is RiskVerdict.REJECTED
        return LiveCheck(
            "kill_switch_operational",
            halts,
            "kill switch halts the risk gate" if halts else "kill switch did NOT halt orders",
        )

    async def _reconciliation_check(self, sleeve_id: str) -> LiveCheck:
        reconciler = self._reconcilers.get(sleeve_id)
        if reconciler is None:
            return LiveCheck("reconciliation_clean", False, f"no reconciler for {sleeve_id}")
        report = await reconciler.execute()
        return LiveCheck(
            "reconciliation_clean",
            report.clean,
            "no drift vs broker" if report.clean else f"{len(report.drifts)} drift(s) detected",
        )

    async def _data_provenance_check(self) -> LiveCheck:
        # Anti-placeholder: no SYNTHETIC/placeholder or BACKFILLED signal may taint the live
        # decision path (the execution wall, data_integrity.execution_barred). An empty window
        # passes (nothing barred present) — the trade-time guard still blocks any fake that
        # arrives, and the startup preflight already requires real market data.
        events = await self._event_store.read_all()
        window = signals_from_events(events, now=self._clock.now(), window_days=self._window_days)
        fake = execution_barred(window)
        if fake:
            return LiveCheck(
                "data_provenance_real",
                False,
                f"{len(fake)} SYNTHETIC/placeholder/BACKFILLED signal(s) in the window — "
                "refusing live",
            )
        detail = (
            f"all {len(window)} in-window signal(s) are REAL"
            if window
            else "no signals in the window (nothing synthetic present)"
        )
        return LiveCheck("data_provenance_real", True, detail)

    async def _record(self, sleeve_id: str, ready: bool, checks: list[LiveCheck]) -> None:
        await self._event_store.append(
            DomainEvent(
                stream_id=f"sleeve-{sleeve_id}",
                event_type=EventType.LIVE_READINESS_ASSESSED,
                payload={
                    "sleeve_id": sleeve_id,
                    "ready": ready,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks
                    ],
                },
                occurred_at=self._clock.now(),
                correlation_id=str(uuid4()),
            )
        )


class RequestLiveTrading:
    def __init__(
        self,
        *,
        registry: SleeveRegistry,
        readiness: AssessLiveReadiness,
        event_store: EventStorePort,
        clock: ClockPort,
    ) -> None:
        self._registry = registry
        self._readiness = readiness
        self._event_store = event_store
        self._clock = clock

    async def enable(self, sleeve_id: str, *, confirm: bool) -> LiveReadinessReport:
        report = await self._readiness.assess(sleeve_id)
        if not confirm:
            raise LiveTradingError("live trading requires explicit confirmation")
        if not report.ready:
            failed = [c.name for c in report.checks if not c.passed]
            raise LiveTradingError(f"sleeve not ready for live: {', '.join(failed)}")
        self._registry.set_mode(sleeve_id, TradingMode.LIVE)
        await self._record_mode_change(sleeve_id, TradingMode.LIVE)
        return report

    async def disable(self, sleeve_id: str) -> None:
        self._registry.set_mode(sleeve_id, TradingMode.PAPER)
        await self._record_mode_change(sleeve_id, TradingMode.PAPER)

    async def _record_mode_change(self, sleeve_id: str, mode: TradingMode) -> None:
        correlation_id = str(uuid4())
        for stream_id, event_type, payload in (
            (
                "user-actions",
                EventType.USER_ACTION,
                {"action": "set_sleeve_mode", "sleeve_id": sleeve_id, "mode": mode.value},
            ),
            (
                f"sleeve-{sleeve_id}",
                EventType.SLEEVE_MODE_CHANGED,
                {"sleeve_id": sleeve_id, "mode": mode.value},
            ),
        ):
            await self._append(stream_id, event_type, payload, correlation_id)

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
