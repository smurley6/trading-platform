"""Backfill job end-to-end for house_disclosures specifically (Task 04b), through the REAL
adapter wired by the composition root.

Complements ``tests/e2e/test_backfill_job.py`` (generic idempotency/PIT contract against a FAKE
historical source) and mirrors ``tests/e2e/test_backfill_job_sec_form4.py``'s pattern: inject the
real ``HouseDisclosuresSource`` -- wired via ``httpx.MockTransport`` against the checked-in
fixtures under ``tests/unit/infrastructure/fixtures/house/`` -- into the composition root's
historical registry, swap its lazy historical-roster loader for a fixture-backed one (so no real
congress-legislators YAML fetch happens), and drive it through ``jobs.backfill`` DRY-RUN and
--confirm, proving: house_disclosures is registered unconditionally (no env gate needed), the
emitted signals carry BACKFILLED provenance + the index FilingDate occurred_at once recorded, and
an unresolved-ticker signal (ticker=None) round-trips through the whole job.
"""

import io
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.config import Settings
from trading.domain.politicians import DEFAULT_BASE_WEIGHTS, Politician, PoliticianRole
from trading.domain.provenance import Provenance
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.interface.app import AppDeps
from trading.jobs import backfill as backfill_job
from trading.main import build_dependencies

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent.parent / "unit" / "infrastructure" / "fixtures" / "house"
_INDEX_2024 = _FIXTURES / "2024FD.xml"
_INDEX_2025 = _FIXTURES / "2025FD.xml"

_SINGLE_TXN_PDF = _FIXTURES / "20032062.pdf"
_SINGLE_TXN_DOC_ID = "20032062"
_SINGLE_TXN_MEMBER = "Robert B. Aderholt"

_MULTI_TXN_PDF = _FIXTURES / "20030502.pdf"
_MULTI_TXN_DOC_ID = "20030502"
_MULTI_TXN_MEMBER = "Richard W. Allen"

_UNRESOLVED_PDF = _FIXTURES / "20040001.pdf"
_UNRESOLVED_DOC_ID = "20040001"
_UNRESOLVED_MEMBER = "Retired Senior Member"

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
    "HOUSE_DISCLOSURES_ENABLED",
    "API_KEY",
)

_WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
_WINDOW_END = datetime(2025, 12, 31, tzinfo=UTC)


def _zip_index(xml_bytes: bytes, *, year: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{year}FD.xml", xml_bytes)
    return buffer.getvalue()


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    zips = {2024: _INDEX_2024, 2025: _INDEX_2025}
    for year, index_path in zips.items():
        if path.endswith(f"/{year}FD.ZIP") or path.endswith(f"/{year}FD.zip"):
            return httpx.Response(200, content=_zip_index(index_path.read_bytes(), year=year))
    pdfs = {
        _SINGLE_TXN_DOC_ID: _SINGLE_TXN_PDF,
        _MULTI_TXN_DOC_ID: _MULTI_TXN_PDF,
        _UNRESOLVED_DOC_ID: _UNRESOLVED_PDF,
    }
    for doc_id, pdf_path in pdfs.items():
        if path.endswith(f"/{doc_id}.pdf"):
            return httpx.Response(200, content=pdf_path.read_bytes())
    return httpx.Response(404)


def _politician(name: str, id_: str) -> Politician:
    return Politician(
        id=id_,
        name=name,
        role=PoliticianRole.MEMBER,
        committees=(),
        party="R",
        base_weight=DEFAULT_BASE_WEIGHTS[PoliticianRole.MEMBER],
        track_record_score=1.0,
        provenance=Provenance.REAL,
    )


async def _fixture_historical_roster() -> PoliticianRepositoryPort:
    return StaticPoliticianRepository(
        [
            _politician(_SINGLE_TXN_MEMBER, "pol-1"),
            _politician(_MULTI_TXN_MEMBER, "pol-2"),
            _politician(_UNRESOLVED_MEMBER, "pol-3"),
        ]
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BACKFILL_ENABLED", "true")


def _install(monkeypatch: pytest.MonkeyPatch) -> dict[str, AppDeps]:
    """Wire the real composition (real HouseDisclosuresSource), but swap its httpx client for a
    mock transport bound to the checked-in fixtures, and its lazy historical-roster loader for a
    fixture-backed one -- no real network anywhere in the run."""
    real_build = build_dependencies
    captured: dict[str, AppDeps] = {}

    def build_once(settings: Settings) -> AppDeps:
        if "deps" in captured:
            return captured["deps"]
        deps = real_build(settings)
        source = deps.historical_sources["house_disclosures"]
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))  # type: ignore[attr-defined]
        source._historical_politicians_loader = _fixture_historical_roster  # type: ignore[attr-defined]
        captured["deps"] = deps
        return deps

    monkeypatch.setattr(backfill_job, "Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(backfill_job, "build_dependencies", build_once)
    return captured


async def test_house_disclosures_registered_with_no_env_gate() -> None:
    """Unlike sec_form4 (gated on a user agent), house_disclosures registers unconditionally."""
    deps = build_dependencies(Settings(_env_file=None))
    try:
        assert "house_disclosures" in deps.historical_sources
    finally:
        for aclose in deps.aclosers:
            await aclose()


async def test_dry_run_reports_fetched_signals_from_real_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch)

    rc = await backfill_job.run_backfill(
        source_id="house_disclosures", start=_WINDOW_START, end=_WINDOW_END, confirm=False
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    # Single-txn (Aderholt/GSK) + multi-txn (Allen) + the unresolved-ticker fixture's one row.
    assert "fetched:" in out


async def test_confirm_records_backfilled_signals_with_filing_date_occurred_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install(monkeypatch)

    rc = await backfill_job.run_backfill(
        source_id="house_disclosures", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    assert rc == 0
    deps = captured["deps"]

    signals = list(await deps.read_model.all_signals())
    assert signals
    assert all(s.provenance is Provenance.BACKFILLED for s in signals)
    assert all(s.pit_confidence == "exact" for s in signals)

    gsk = next(s for s in signals if s.ticker == "GSK")
    # 2025FD.xml: Aderholt's FilingDate is 9/10/2025 -> midnight UTC (the PUBLIC disclosure
    # moment), never the underlying transaction date.
    assert gsk.created_at == datetime(2025, 9, 10, tzinfo=UTC)


async def test_confirm_records_unresolved_ticker_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unresolved-entity signal (ticker=None) is recorded like any other -- research data,
    never silently dropped -- and the read model round-trips it through Postgres-shaped columns
    (here the in-memory read model, since this test has no TEST_DATABASE_URL dependency)."""
    captured = _install(monkeypatch)

    rc = await backfill_job.run_backfill(
        source_id="house_disclosures", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    assert rc == 0
    deps = captured["deps"]

    signals = list(await deps.read_model.all_signals())
    unresolved = [s for s in signals if s.raw_payload.get("doc_id") == _UNRESOLVED_DOC_ID]
    assert len(unresolved) == 1
    assert unresolved[0].ticker is None
    assert unresolved[0].provenance is Provenance.BACKFILLED


async def test_rerun_is_idempotent_no_duplicate_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install(monkeypatch)

    first = await backfill_job.run_backfill(
        source_id="house_disclosures", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    second = await backfill_job.run_backfill(
        source_id="house_disclosures", start=_WINDOW_START, end=_WINDOW_END, confirm=True
    )
    assert first == 0
    assert second == 0
    deps = captured["deps"]

    signals = list(await deps.read_model.all_signals())
    signal_ids = [s.id for s in signals]
    assert len(signal_ids) == len(set(signal_ids))  # no duplicates from the re-run
