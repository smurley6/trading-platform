"""Backfill job end-to-end on the real composition root (in-memory store + read model).

Proves the two acceptance criteria from tasks/task-02-backfill.md against the ACTUAL wired
dependencies (`build_dependencies`), not fakes:
  * Idempotency at BOTH layers: running a window twice leaves the event count AND the read
    model's signal count identical after the second run (event-log + rm_signals dedupe).
  * Point-in-time timestamp: the emitted signal.received event's occurred_at equals the fake
    source's disclosure date — not the ingest wall-clock, not any later event/trade date.
Plus the BACKFILLED provenance + per-source pit_confidence stamping the labeler relies on.

No network, no real DB: a fake historical source is injected into the composition's (currently
empty) historical-source registry; DATABASE_URL is cleared so the in-memory read model stands in
for rm_signals (it maintains the same projections production's Postgres read model does).
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.interface.app import AppDeps
from trading.jobs import backfill as backfill_job
from trading.main import build_dependencies

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "DATABASE_URL",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "REQUIRE_REAL_DATA",
    "ENABLE_LIVE_QUOTES",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)

# Disclosure timestamps are historical (a real backfill window); the event's occurred_at must be
# these, never the ingest wall-clock the composition's SystemClock would produce.
_DISCLOSED_A = datetime(2024, 3, 1, 20, 30, tzinfo=UTC)
_DISCLOSED_B = datetime(2024, 3, 5, 18, 0, tzinfo=UTC)
_WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
_WINDOW_END = datetime(2024, 6, 1, tzinfo=UTC)


def _hist_signal(*, signal_id: str, source_uid: str, disclosed_at: datetime) -> Signal:
    return Signal(
        id=signal_id,
        created_at=disclosed_at,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker="AAPL",
        politician_id=None,
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.75,
        raw_payload={"source_uid": source_uid, "source": "fake-historical"},
        provenance=Provenance.REAL,  # the use case must re-stamp to BACKFILLED
    )


class _FakeHistoricalSource:
    def __init__(self, name: str, batch: Sequence[Signal]) -> None:
        self._name = name
        self._batch = list(batch)

    @property
    def name(self) -> str:
        return self._name

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        return list(self._batch)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BACKFILL_ENABLED", "true")


def _install(monkeypatch: pytest.MonkeyPatch) -> dict[str, AppDeps]:
    """Wire the real composition, injecting a fake sec_form4 historical source; capture the deps
    so the shared in-memory store/read model persist across the job's two runs."""
    source = _FakeHistoricalSource(
        "sec_form4",
        [
            _hist_signal(signal_id="s1", source_uid="acc-1", disclosed_at=_DISCLOSED_A),
            _hist_signal(signal_id="s2", source_uid="acc-2", disclosed_at=_DISCLOSED_B),
        ],
    )
    captured: dict[str, AppDeps] = {}
    real_build = build_dependencies

    def build_once(settings: Settings) -> AppDeps:
        if "deps" in captured:
            return captured["deps"]  # reuse across the two runs (shared in-memory state)
        deps = real_build(settings)
        deps.historical_sources[source.name] = source  # inject into the (empty) registry
        captured["deps"] = deps
        return deps

    monkeypatch.setattr(backfill_job, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(backfill_job, "build_dependencies", build_once)
    return captured


async def _signal_events(deps: AppDeps) -> int:
    return len(
        [
            e
            for e in await deps.event_store.read_all()
            if e.event.event_type == EventType.SIGNAL_RECEIVED
        ]
    )


async def test_rerun_is_idempotent_at_event_log_and_read_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    rc1 = await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    deps = captured["deps"]
    signal_events_after_1 = await _signal_events(deps)
    signals_after_1 = len(await deps.read_model.all_signals())

    rc2 = await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    signal_events_after_2 = await _signal_events(deps)
    signals_after_2 = len(await deps.read_model.all_signals())

    assert rc1 == 0 and rc2 == 0
    # Idempotency proven at BOTH layers: the re-run double-appends NO signal.received events and
    # duplicates NO rm_signals rows (only the per-run user.action audit event is added, by design).
    assert signal_events_after_1 == signal_events_after_2 == 2
    assert signals_after_1 == signals_after_2 == 2


async def test_emitted_occurred_at_is_disclosure_date_not_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    deps = captured["deps"]

    events = [
        e
        for e in await deps.event_store.read_all()
        if e.event.event_type == EventType.SIGNAL_RECEIVED
    ]
    occurred = {e.event.payload["signal_id"]: e.event.occurred_at for e in events}
    assert occurred["s1"] == _DISCLOSED_A
    assert occurred["s2"] == _DISCLOSED_B
    # The composition's SystemClock "now" is the run wall-clock; the disclosure dates are 2024.
    now = deps.clock.now()
    assert all(ts != now for ts in occurred.values())


async def test_backfilled_signals_carry_provenance_and_pit_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    deps = captured["deps"]

    signals = list(await deps.read_model.all_signals())
    assert len(signals) == 2
    assert all(s.provenance is Provenance.BACKFILLED for s in signals)
    # sec_form4 is a tier-1 "exact" source in the RESEARCH-PLAN §3 table.
    assert all(s.pit_confidence == "exact" for s in signals)
