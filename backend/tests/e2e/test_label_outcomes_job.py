"""Outcome-labeler job end-to-end on the real composition root (in-memory store + read model).

Records BACKFILLED signals via the real wiring, then runs the labeler job over them and asserts
labels land in the rebuildable outcome_labels projection (not the event log) with a per-run
user.action audit event. No network, no real DB: DATABASE_URL is cleared so the in-memory read
model + in-memory outcome-labels store + SyntheticHistoricalPrices stand in.

Because the synthetic price provider is wired (not the REAL Alpaca adapter), --confirm is refused
by the job guard — so the job path here is exercised as a dry-run, and the labeler use case is
driven directly (via deps.label_outcomes) for the write path, mirroring what --confirm does with a
real adapter.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from trading.application.outcome_labels import HORIZONS
from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.interface.app import AppDeps
from trading.jobs import label_outcomes as label_job
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
    "ENABLE_HISTORICAL_DATA",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)

# Old disclosures so the 20-day horizon resolves against synthetic bars (buffer extends past max
# disclosure by ~2x the max horizon in calendar days).
_DISCLOSED = datetime(2025, 1, 6, 20, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)


def _backfilled_signal(*, signal_id: str, ticker: str) -> Signal:
    return Signal(
        id=signal_id,
        created_at=_DISCLOSED,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=None,
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.75,
        raw_payload={"source_uid": signal_id},
        provenance=Provenance.BACKFILLED,
    )


async def _record(deps: AppDeps, signals: Sequence[Signal]) -> None:
    from trading.application.signal_recording import record_signals

    for signal in signals:
        await record_signals(
            event_store=deps.event_store,
            clock=deps.clock,
            signals=[signal],
            correlation_id=str(uuid4()),
            occurred_at=signal.created_at,
        )


async def test_labeler_writes_labels_from_read_model() -> None:
    deps = build_dependencies(Settings(_env_file=None))
    try:
        await _record(
            deps,
            [
                _backfilled_signal(signal_id="s1", ticker="AAPL"),
                _backfilled_signal(signal_id="s2", ticker="LMT"),
            ],
        )
        assert deps.label_outcomes is not None and deps.outcome_labels is not None

        run_id = str(uuid4())
        report = await deps.label_outcomes.execute(label_run_id=run_id, dry_run=False)

        # Two ticker-resolved BACKFILLED signals x 3 horizons = 6 label rows.
        assert report.signal_count == 2
        assert report.labels == 2 * len(HORIZONS)
        assert await deps.outcome_labels.count_for_run(run_id) == 2 * len(HORIZONS)

        # The labeler recorded exactly one user.action audit event; NO outcome data leaked into
        # the event log as signal-like events (labels live only in the projection).
        events = await deps.event_store.read_all()
        audits = [e for e in events if e.event.event_type == EventType.USER_ACTION]
        assert len(audits) == 1
        assert audits[0].event.payload["action"] == "label_outcomes"
    finally:
        for aclose in deps.aclosers:
            await aclose()


async def test_job_confirm_refused_without_real_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The synthetic price provider is wired (ENABLE_HISTORICAL_DATA cleared) -> --confirm refused.
    monkeypatch.setattr(label_job, "Settings", lambda: Settings(_env_file=None))
    rc = await label_job.run_labeling(confirm=True)
    assert rc == 1
    assert "REAL historical-bars adapter" in capsys.readouterr().out


async def test_job_dry_run_runs_against_synthetic_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(label_job, "Settings", lambda: Settings(_env_file=None))
    rc = await label_job.run_labeling(confirm=False)
    assert rc == 0
    assert "DRY-RUN" in capsys.readouterr().out
