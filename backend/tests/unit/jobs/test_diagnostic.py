"""Diagnostic job CLI, driven against fake/in-memory deps (never a real DB or network).

`build_dependencies` is monkeypatched to return a lightweight fake deps carrying: a fake
`ReadModelPort` (an explicit signals batch), an `InMemoryOutcomeLabels` store (real port impl,
seeded directly), a real `InMemoryEventStore` (to assert the single audit event), and a fixed
clock. Covers: latest-run selection (default), an explicit `--run-id`, "no run found" (exit 1),
JSON to stdout vs `--out <path>`, the human-readable table, the single `user.action` audit event,
and `main()` exit-code mapping.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading.application.ports.outcome_labels import OutcomeLabel
from trading.application.ports.read_model import PortfolioSnapshotRecord, SignalRecord
from trading.config import Settings
from trading.domain.conviction import ConvictionThresholds
from trading.domain.events import EventType
from trading.domain.positions import HeldPosition
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_outcome_labels import InMemoryOutcomeLabels
from trading.jobs import diagnostic

_AS_OF = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
_SETTINGS = Settings(
    _env_file=None,
    max_trade_pct_of_sleeve="10",
    max_position_concentration_pct="25",
    max_drawdown_pct="20",
)


class _FixedClock:
    def now(self) -> datetime:
        return _AS_OF


def _signal(signal_id: str, ticker: str | None = "AAPL") -> Signal:
    return Signal(
        id=signal_id,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=None,
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.75,
        provenance=Provenance.BACKFILLED,
    )


def _label(run_id: str, signal_id: str = "sec-form4-1") -> OutcomeLabel:
    return OutcomeLabel(
        signal_id=signal_id,
        ticker="AAPL",
        horizon_days=1,
        abnormal_return=0.05,
        benchmark="market",
        status="ok",
        reason=None,
        label_run_id=run_id,
    )


class _FakeReadModel:
    def __init__(self, signals: Sequence[Signal]) -> None:
        self._signals = signals

    async def held_positions(self) -> Sequence[HeldPosition]:  # pragma: no cover - unused
        return []

    async def order_statuses(self, *, sleeve_id: str | None = None) -> dict[str, str]:
        return {}  # pragma: no cover - unused

    async def signals_in_window(
        self, *, now: datetime, window_days: int
    ) -> Sequence[Signal]:  # pragma: no cover - unused
        return []

    async def recent_signals(self, *, limit: int) -> Sequence[SignalRecord]:  # pragma: no cover
        return []

    async def all_signals(self) -> Sequence[Signal]:
        return self._signals

    async def latest_portfolio_snapshots(
        self,
    ) -> Sequence[PortfolioSnapshotRecord]:  # pragma: no cover
        return []


@dataclass
class _FakeDeps:
    """The AppDeps surface the diagnostic job reads."""

    read_model: _FakeReadModel
    outcome_labels: InMemoryOutcomeLabels
    event_store: InMemoryEventStore
    clock: _FixedClock
    conviction_thresholds: ConvictionThresholds = field(
        default_factory=lambda: ConvictionThresholds(trade=70.0, watch=40.0)
    )
    conviction_window_days: int = 21
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


async def _seeded_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    signals: Sequence[Signal] = (),
    labels_by_run: dict[str, list[OutcomeLabel]] | None = None,
) -> _FakeDeps:
    labels_store = InMemoryOutcomeLabels()
    for run_id, labels in (labels_by_run or {}).items():
        await labels_store.replace_run(run_id, labels)
    clock = _FixedClock()
    deps = _FakeDeps(
        read_model=_FakeReadModel(signals),
        outcome_labels=labels_store,
        event_store=InMemoryEventStore(clock),
        clock=clock,
    )
    monkeypatch.setattr(diagnostic, "build_dependencies", lambda _settings: deps)
    monkeypatch.setattr(diagnostic, "Settings", lambda: _SETTINGS)
    monkeypatch.setattr(diagnostic, "configure_logging", lambda _level: None)
    return deps


async def test_defaults_to_latest_run_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await _seeded_deps(
        monkeypatch,
        signals=[_signal("sec-form4-1")],
        labels_by_run={
            "run-a": [_label("run-a")],
            "run-b": [_label("run-b")],  # written after run-a -> latest
        },
    )

    rc = await diagnostic.run_diagnostic(run_id=None, out=None)

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n\n", 1)[0])
    assert payload["label_run_id"] == "run-b"


async def test_explicit_run_id_overrides_latest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await _seeded_deps(
        monkeypatch,
        signals=[_signal("sec-form4-1")],
        labels_by_run={"run-a": [_label("run-a")], "run-b": [_label("run-b")]},
    )

    rc = await diagnostic.run_diagnostic(run_id="run-a", out=None)

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("\n\n", 1)[0])
    assert payload["label_run_id"] == "run-a"


async def test_no_run_found_returns_one_when_store_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await _seeded_deps(monkeypatch, signals=[], labels_by_run={})

    rc = await diagnostic.run_diagnostic(run_id=None, out=None)

    assert rc == 1
    assert "no label run found" in capsys.readouterr().out


async def test_explicit_run_id_not_found_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await _seeded_deps(
        monkeypatch, signals=[_signal("sec-form4-1")], labels_by_run={"run-a": [_label("run-a")]}
    )

    rc = await diagnostic.run_diagnostic(run_id="does-not-exist", out=None)

    assert rc == 1
    assert "no labels found" in capsys.readouterr().out


async def test_writes_json_to_out_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await _seeded_deps(
        monkeypatch, signals=[_signal("sec-form4-1")], labels_by_run={"run-a": [_label("run-a")]}
    )
    out_path = tmp_path / "report.json"

    rc = await diagnostic.run_diagnostic(run_id="run-a", out=str(out_path))

    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["label_run_id"] == "run-a"
    assert payload["signal_count"] == 1
    assert payload["label_count"] == 1


async def test_prints_human_readable_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await _seeded_deps(
        monkeypatch, signals=[_signal("sec-form4-1")], labels_by_run={"run-a": [_label("run-a")]}
    )

    rc = await diagnostic.run_diagnostic(run_id="run-a", out=None)

    assert rc == 0
    out = capsys.readouterr().out
    assert "=== Diagnostic report ===" in out
    assert "-- 1. Resolution by source --" in out
    assert "-- 2. Monthly disclosure-date histogram --" in out
    assert "-- 3. Abnormal return by source x horizon x benchmark class --" in out
    assert "-- 4. Abnormal return by source x horizon x conviction bucket --" in out
    assert "-- 5. Prior vs measured reliability --" in out
    assert "-- 6. Unresolvable-label reasons --" in out


async def test_appends_exactly_one_user_action_audit_event(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = await _seeded_deps(
        monkeypatch, signals=[_signal("sec-form4-1")], labels_by_run={"run-a": [_label("run-a")]}
    )

    rc = await diagnostic.run_diagnostic(run_id="run-a", out=None)

    assert rc == 0
    events = await deps.event_store.read_all()
    audit_events = [e for e in events if e.event.event_type == EventType.USER_ACTION]
    assert len(audit_events) == 1
    payload = audit_events[0].event.payload
    assert payload["action"] == "diagnostic"
    assert payload["label_run_id"] == "run-a"
    assert payload["signal_count"] == 1
    assert payload["label_count"] == 1
    assert payload["as_of"] == _AS_OF.isoformat()
    assert set(payload["sections"]) == {
        "resolution_by_source",
        "monthly_histogram",
        "horizon_benchmark_stats",
        "horizon_conviction_stats",
        "prior_vs_measured",
        "unresolvable_reasons",
    }


async def test_no_audit_event_when_no_run_found(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = await _seeded_deps(monkeypatch, signals=[], labels_by_run={})

    rc = await diagnostic.run_diagnostic(run_id=None, out=None)

    assert rc == 1
    events = await deps.event_store.read_all()
    assert events == []


def test_main_dry_success_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _seed() -> None:
        await _seeded_deps(
            monkeypatch,
            signals=[_signal("sec-form4-1")],
            labels_by_run={"run-a": [_label("run-a")]},
        )

    asyncio.run(_seed())
    assert diagnostic.main(["--run-id", "run-a"]) == 0


def test_main_unknown_flag_returns_one() -> None:
    # argparse exits 2 on a usage error; the entrypoint normalizes it to 1.
    assert diagnostic.main(["--bogus"]) == 1


def test_main_no_run_found_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _seed() -> None:
        await _seeded_deps(monkeypatch, signals=[], labels_by_run={})

    asyncio.run(_seed())
    assert diagnostic.main([]) == 1
