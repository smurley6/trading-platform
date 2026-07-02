"""Backfill job end-to-end for sec_form4 specifically (Task 04a), through the REAL adapter.

Complements ``tests/e2e/test_backfill_job.py`` (which proves the job's generic idempotency/PIT
contract against a FAKE historical source). Here we inject the real ``SecForm4Source`` -- wired
via ``httpx.MockTransport`` against the checked-in fixtures under
``tests/unit/infrastructure/fixtures/sec/`` -- into the composition root's historical registry
and drive it through ``jobs.backfill`` DRY-RUN, proving: the CIK filter is ON by default and its
description reaches the run's ``user.action`` ``filters`` field on a --confirm run; ``--no-filter``
disables it; the emitted signals carry BACKFILLED provenance + ACCEPTANCE-DATETIME occurred_at
once recorded.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.provenance import Provenance
from trading.interface.app import AppDeps
from trading.jobs import backfill as backfill_job
from trading.main import build_dependencies

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent.parent / "unit" / "infrastructure" / "fixtures" / "sec"
_FIXTURE_BY_PATH: dict[str, str] = {
    "/Archives/edgar/full-index/2024/QTR1/form.idx": "form_2024QTR1.idx",
    "/Archives/edgar/data/320193/0000320193-24-000005.txt": "submission_apple.txt",
    "/Archives/edgar/data/789019/0000789019-24-000010.txt": "submission_microsoft.txt",
    "/files/company_tickers.json": "company_tickers.json",
}

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
    "SEC_FORM4_ENABLED",
    "API_KEY",
)

_WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
_WINDOW_END = datetime(2024, 3, 31, tzinfo=UTC)


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text()


def _handler(request: httpx.Request) -> httpx.Response:
    fixture = _FIXTURE_BY_PATH.get(request.url.path)
    if fixture is None:
        return httpx.Response(404, text="not found")
    return httpx.Response(200, text=_read(fixture))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SEC_FORM4_USER_AGENT", "trading-platform research test@example.com")
    monkeypatch.setenv("BACKFILL_ENABLED", "true")


def _install(monkeypatch: pytest.MonkeyPatch) -> dict[str, AppDeps]:
    """Wire the real composition (real SecForm4Source), but swap its httpx client for a mock
    transport bound to the checked-in fixtures -- no real network."""
    real_build = build_dependencies
    captured: dict[str, AppDeps] = {}

    def build_once(settings: Settings) -> AppDeps:
        if "deps" in captured:
            return captured["deps"]
        deps = real_build(settings)
        source = deps.historical_sources["sec_form4"]
        # Swap the adapter's real httpx client for one bound to our mock transport (offline).
        # SecForm4Source is a plain class (not frozen), so direct attribute assignment works.
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))  # type: ignore[attr-defined]
        captured["deps"] = deps
        return deps

    monkeypatch.setattr(backfill_job, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(backfill_job, "build_dependencies", build_once)
    return captured


async def test_dry_run_reports_fetched_signals_from_real_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch)

    rc = await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=False
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "fetched:   3" in out  # Apple P + Microsoft S + Microsoft P (award skipped)
    assert "DRY-RUN" in out
    assert "CIK filter" in out  # default-on filter shown in the summary


async def test_confirm_records_backfilled_signals_with_correct_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    rc = await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    assert rc == 0
    deps = captured["deps"]

    signals = list(await deps.read_model.all_signals())
    assert len(signals) == 3
    assert all(s.provenance is Provenance.BACKFILLED for s in signals)
    assert all(s.pit_confidence == "exact" for s in signals)

    aapl = next(s for s in signals if s.ticker == "AAPL")
    assert aapl.created_at == datetime(2024, 1, 15, 22, 30, 45, tzinfo=UTC)  # ACCEPTANCE-DATETIME


async def test_confirm_records_active_filter_in_user_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    await backfill_job.run_backfill(
        source_id="sec_form4", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    deps = captured["deps"]

    actions = [
        e.event.payload
        for e in await deps.event_store.read_all()
        if e.event.event_type == EventType.USER_ACTION
    ]
    assert len(actions) == 1
    filters = actions[0]["filters"]
    assert isinstance(filters, dict)
    assert "CIK filter" in filters.get("description", "")


async def test_no_filter_flag_widens_the_fetch_and_clears_filters_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    rc = await backfill_job.run_backfill(
        source_id="sec_form4",
        start=_WINDOW_START,
        end=_WINDOW_END,
        confirm=True,
        no_filter=True,
    )
    assert rc == 0
    deps = captured["deps"]

    actions = [
        e.event.payload
        for e in await deps.event_store.read_all()
        if e.event.event_type == EventType.USER_ACTION
    ]
    assert actions[0]["filters"] == {}
