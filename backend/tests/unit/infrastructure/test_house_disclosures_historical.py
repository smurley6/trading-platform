"""House Clerk STOCK Act PTR HISTORICAL adapter (Task 04b, docs/RESEARCH-PLAN.md), no network.

Fixtures under ``fixtures/house/``: the real ``2025FD.xml`` index (single- and multi-transaction
e-filed PTRs already used by the live-path tests) plus a new ``2024FD.xml`` snippet (an e-filed
PTR for a filer who exists ONLY in the historical roster, a paper/scanned PTR to count, and a
non-PTR "O" filing type to ignore), the two committed real PTR PDFs, and a hand-built minimal PDF
(``20040001.pdf``) whose sole transaction line has no ticker in parens and names a fictional
company no gazetteer alias will ever match — the unresolved-ticker fixture.

The PIT rule this task closes: ``created_at`` on this path is the index's ``FilingDate``
(midnight UTC) — the SAME rule the (unchanged) live path already applies, just walked over a wide
``[start, end]`` window instead of a rolling ``since`` cursor, and resolved against a COMBINED
current + historical roster instead of current-only.
"""

import io
import zipfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.domain.politicians import DEFAULT_BASE_WEIGHTS, Politician, PoliticianRole
from trading.domain.provenance import Provenance
from trading.infrastructure.data_sources.house_disclosures import HouseDisclosuresSource
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.infrastructure.reference.sectors import StaticSectorMap

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "house"
_INDEX_2025 = _FIXTURES / "2025FD.xml"
_INDEX_2024 = _FIXTURES / "2024FD.xml"

_SINGLE_TXN_PDF = _FIXTURES / "20032062.pdf"
_SINGLE_TXN_DOC_ID = "20032062"
_SINGLE_TXN_MEMBER = "Robert B. Aderholt"

_MULTI_TXN_PDF = _FIXTURES / "20030502.pdf"
_MULTI_TXN_DOC_ID = "20030502"
_MULTI_TXN_MEMBER = "Richard W. Allen"

_UNRESOLVED_PDF = _FIXTURES / "20040001.pdf"
_UNRESOLVED_DOC_ID = "20040001"
_UNRESOLVED_MEMBER = "Retired Senior Member"

_WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
_WINDOW_END = datetime(2025, 12, 31, tzinfo=UTC)


def _zip_index(xml_bytes: bytes, *, year: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{year}FD.xml", xml_bytes)
    return buffer.getvalue()


def _politician(
    name: str, *, id_: str = "pol-house-1", provenance: Provenance = Provenance.REAL
) -> Politician:
    return Politician(
        id=id_,
        name=name,
        role=PoliticianRole.MEMBER,
        committees=(),
        party="R",
        base_weight=DEFAULT_BASE_WEIGHTS[PoliticianRole.MEMBER],
        track_record_score=1.0,
        provenance=provenance,
    )


def _combined_repo(*names: str) -> StaticPoliticianRepository:
    return StaticPoliticianRepository(
        [_politician(name, id_=f"pol-{i}") for i, name in enumerate(names)]
    )


def _handler() -> Handler:
    """Serve both year index zips and the three PTR PDFs; 404 anything unknown."""
    zips = {
        2024: _zip_index(_INDEX_2024.read_bytes(), year=2024),
        2025: _zip_index(_INDEX_2025.read_bytes(), year=2025),
    }
    pdfs = {
        _SINGLE_TXN_DOC_ID: _SINGLE_TXN_PDF.read_bytes(),
        _MULTI_TXN_DOC_ID: _MULTI_TXN_PDF.read_bytes(),
        _UNRESOLVED_DOC_ID: _UNRESOLVED_PDF.read_bytes(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for year, content in zips.items():
            if path.endswith(f"/{year}FD.ZIP") or path.endswith(f"/{year}FD.zip"):
                return httpx.Response(200, content=content)
        for doc_id, pdf_bytes in pdfs.items():
            if path.endswith(f"/{doc_id}.pdf"):
                return httpx.Response(200, content=pdf_bytes)
        return httpx.Response(404)

    return handler


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _historical_roster() -> PoliticianRepositoryPort:
    # The fake "lazy loader" this source calls on first fetch_historical: a roster including a
    # filer who (in real life) would exist ONLY in legislators-historical.
    return _combined_repo(_SINGLE_TXN_MEMBER, _MULTI_TXN_MEMBER, _UNRESOLVED_MEMBER)


RosterLoader = Callable[[], Awaitable[PoliticianRepositoryPort]]


def _source(
    handler: Handler,
    clock: FixedClock,
    *,
    historical_politicians: RosterLoader | None = None,
) -> tuple[HouseDisclosuresSource, httpx.AsyncClient]:
    client = _client(handler)
    source = HouseDisclosuresSource(
        client=client,
        politicians=StaticPoliticianRepository(),  # live-only roster; unused by fetch_historical
        sector_map=StaticSectorMap(),
        clock=clock,
        entity_resolver=GazetteerEntityResolver(),
        historical_politicians=historical_politicians or _historical_roster,
        sleep=_noop_sleep,
    )
    return source, client


async def _noop_sleep(_seconds: float) -> None:
    return None


# --- Fixture test: correct transaction rows, filing-date created_at, source_uid, e-filed-only,
#     paper filings counted -----------------------------------------------------------------


async def test_fetch_historical_returns_correct_transaction_rows(clock: FixedClock) -> None:
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    single = [s for s in signals if s.raw_payload["doc_id"] == _SINGLE_TXN_DOC_ID]
    multi = [s for s in signals if s.raw_payload["doc_id"] == _MULTI_TXN_DOC_ID]
    assert len(single) == 1
    assert single[0].ticker == "GSK"
    assert single[0].sentiment < 0  # the fixture PTR is a sale
    assert len(multi) > 1  # the multi-transaction PTR yields more than one signal


async def test_created_at_is_index_filing_date_midnight_utc(clock: FixedClock) -> None:
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    single = next(s for s in signals if s.raw_payload["doc_id"] == _SINGLE_TXN_DOC_ID)
    # 2025FD.xml: Aderholt's FilingDate is 9/10/2025 -> midnight UTC, NOT the transaction date
    # (07/28/2025, retained separately in raw_payload).
    assert single.created_at == datetime(2025, 9, 10, tzinfo=UTC)
    assert single.raw_payload["transaction_date"] == "2025-07-28"
    assert single.created_at.date().isoformat() != single.raw_payload["transaction_date"]


async def test_source_uid_is_doc_id_plus_row_index(clock: FixedClock) -> None:
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    multi = [s for s in signals if s.raw_payload["doc_id"] == _MULTI_TXN_DOC_ID]
    uids = [s.raw_payload["source_uid"] for s in multi]
    assert uids == [f"{_MULTI_TXN_DOC_ID}:{i}" for i in range(len(multi))]
    assert len(set(uids)) == len(uids)  # no collisions
    # The bare DocID is also present as its own field (per clarification 4).
    assert all(s.raw_payload["doc_id"] == _MULTI_TXN_DOC_ID for s in multi)


async def test_e_filed_only_constraint_still_applies(clock: FixedClock) -> None:
    """The 'O' (annual, non-PTR) filing in 2024FD.xml must never appear at all."""
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert all(s.raw_payload["doc_id"] != "20049999" for s in signals)


async def test_paper_filings_are_counted_and_run_continues(
    clock: FixedClock, caplog: pytest.LogCaptureFixture
) -> None:
    """2024FD.xml's DocID 10099999 is a PTR but NOT e-filed (starts with '1') -> excluded from
    the parseable entries entirely, but counted in the paper-filing tally, and the run still
    produces the other years' signals."""
    source, client = _source(_handler(), clock)
    async with client:
        with caplog.at_level("INFO"):
            signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert all(s.raw_payload["doc_id"] != "10099999" for s in signals)
    assert any("paper filing" in record.message for record in caplog.records)
    # The run continues: 2025's signals are still present despite 2024 containing a paper filing.
    assert any(s.raw_payload["doc_id"] == _SINGLE_TXN_DOC_ID for s in signals)


async def test_window_filters_by_filing_date(clock: FixedClock) -> None:
    """A window that excludes 2025 entirely must exclude the 2025 PTRs."""
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)
        )

    assert all(s.raw_payload["doc_id"] != _SINGLE_TXN_DOC_ID for s in signals)
    assert all(s.raw_payload["doc_id"] != _MULTI_TXN_DOC_ID for s in signals)


# --- Unparseable-PDF fixture -> logged, counted, run continues -------------------------------


async def test_unfetchable_pdf_is_logged_counted_and_run_continues(
    clock: FixedClock, caplog: pytest.LogCaptureFixture
) -> None:
    """20032062's PDF 404s (simulated) -> logged + counted, the OTHER filings still produce
    signals (the walk never aborts on one bad filing)."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(f"/{_SINGLE_TXN_DOC_ID}.pdf"):
            return httpx.Response(404)
        return _handler()(request)

    source, client = _source(handler, clock)
    async with client:
        with caplog.at_level("WARNING"):
            signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert all(s.raw_payload["doc_id"] != _SINGLE_TXN_DOC_ID for s in signals)
    assert any("unparseable/unfetchable PDF" in record.message for record in caplog.records)
    # The walk continues: the multi-transaction filing still produced signals.
    assert any(s.raw_payload["doc_id"] == _MULTI_TXN_DOC_ID for s in signals)


# --- Historical-roster test: a filer present ONLY in legislators-historical resolves ----------


async def test_filer_present_only_in_historical_roster_resolves(clock: FixedClock) -> None:
    """The lazy roster loader supplies a filer who would NOT be in the current-only roster; their
    PTR must still produce signals (this is the whole point of the historical roster load)."""
    current_only = StaticPoliticianRepository()  # empty: nobody resolves via current alone

    async def historical_loader() -> PoliticianRepositoryPort:
        return _combined_repo(_SINGLE_TXN_MEMBER, _MULTI_TXN_MEMBER, _UNRESOLVED_MEMBER)

    client = _client(_handler())
    source = HouseDisclosuresSource(
        client=client,
        politicians=current_only,
        sector_map=StaticSectorMap(),
        clock=clock,
        entity_resolver=GazetteerEntityResolver(),
        historical_politicians=historical_loader,
        sleep=_noop_sleep,
    )
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert any(s.raw_payload["doc_id"] == _UNRESOLVED_DOC_ID for s in signals)


async def test_unmatched_filer_is_skipped_and_counted(
    clock: FixedClock, caplog: pytest.LogCaptureFixture
) -> None:
    async def empty_roster() -> PoliticianRepositoryPort:
        return StaticPoliticianRepository()

    client = _client(_handler())
    source = HouseDisclosuresSource(
        client=client,
        politicians=StaticPoliticianRepository(),
        sector_map=StaticSectorMap(),
        clock=clock,
        entity_resolver=GazetteerEntityResolver(),
        historical_politicians=empty_roster,
        sleep=_noop_sleep,
    )
    async with client:
        with caplog.at_level("INFO"):
            signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert signals == []
    assert any("not in roster" in record.message for record in caplog.records)


async def test_historical_roster_loader_called_lazily_and_cached(clock: FixedClock) -> None:
    """The loader must not be invoked at construction time, and must be invoked at most once
    across multiple fetch_historical calls (cached on the instance)."""
    calls = 0

    async def counting_loader() -> PoliticianRepositoryPort:
        nonlocal calls
        calls += 1
        return await _historical_roster()

    client = _client(_handler())
    source = HouseDisclosuresSource(
        client=client,
        politicians=StaticPoliticianRepository(),
        sector_map=StaticSectorMap(),
        clock=clock,
        entity_resolver=GazetteerEntityResolver(),
        historical_politicians=counting_loader,
        sleep=_noop_sleep,
    )
    assert calls == 0  # construction alone must not touch the loader
    async with client:
        await source.fetch_historical(_WINDOW_START, _WINDOW_END)
        assert calls == 1
        await source.fetch_historical(_WINDOW_START, _WINDOW_END)
        assert calls == 1  # cached, not re-fetched


# --- Unresolved-ticker test: signal emitted with ticker=None and counted ---------------------


async def test_unresolved_ticker_emits_signal_with_none_and_counted(
    clock: FixedClock, caplog: pytest.LogCaptureFixture
) -> None:
    source, client = _source(_handler(), clock)
    async with client:
        with caplog.at_level("INFO"):
            signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    unresolved = [s for s in signals if s.raw_payload["doc_id"] == _UNRESOLVED_DOC_ID]
    assert len(unresolved) == 1
    assert unresolved[0].ticker is None
    assert unresolved[0].raw_payload["ticker"] is None
    assert any("unresolved ticker" in record.message for record in caplog.records)


async def test_unresolved_ticker_signal_still_carries_source_uid_and_pit_timestamp(
    clock: FixedClock,
) -> None:
    """An unresolved-ticker signal is not a second-class citizen: it still dedupes and PIT-stamps
    exactly like a resolved one."""
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    unresolved = next(s for s in signals if s.raw_payload["doc_id"] == _UNRESOLVED_DOC_ID)
    assert unresolved.raw_payload["source_uid"] == f"{_UNRESOLVED_DOC_ID}:0"
    # 2024FD.xml: FilingDate 3/25/2024 -> midnight UTC.
    assert unresolved.created_at == datetime(2024, 3, 25, tzinfo=UTC)


# --- Provenance: adapter always stamps REAL; BackfillSignals re-stamps regardless ------------


async def test_provenance_is_real_and_pit_confidence_exact(clock: FixedClock) -> None:
    source, client = _source(_handler(), clock)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert signals
    assert all(s.provenance is Provenance.REAL for s in signals)
    assert all(s.pit_confidence == "exact" for s in signals)


# --- Per-run summary counters: unmatched_filers (by reason), amount_parse_misses (Task 07) -----


async def test_source_implements_summarizing_historical_source_protocol() -> None:
    """``HouseDisclosuresSource`` is the first implementer of the optional
    ``SummarizingHistoricalSource`` capability (Task 07, docs/RESEARCH-PLAN.md)."""
    from trading.application.ports.data_source import SummarizingHistoricalSource

    client = _client(_handler())
    source = HouseDisclosuresSource(
        client=client,
        politicians=StaticPoliticianRepository(),
        sector_map=StaticSectorMap(),
        clock=FixedClock(datetime(2025, 12, 31, tzinfo=UTC)),
    )
    assert isinstance(source, SummarizingHistoricalSource)


async def test_last_run_summary_is_none_before_any_fetch(clock: FixedClock) -> None:
    source, _ = _source(_handler(), clock)
    assert source.last_run_summary() is None


async def test_last_run_summary_reports_unmatched_filers_by_reason(clock: FixedClock) -> None:
    """A roster with an AMBIGUOUS entry (two distinct politicians sharing normalized identity
    with the multi-txn filer "Richard W. Allen") plus a totally-unresolvable filer (the
    unresolved-ticker fixture's "Retired Senior Member" absent from the roster) drives BOTH
    unmatched_filers reasons in one run, asserted via the summary object the job/use case reads
    (not just the log line)."""

    async def ambiguous_roster() -> PoliticianRepositoryPort:
        # Two distinct roster entries both normalize to first="richard", last="allen" (the
        # middle initial "W." is an ignored middle token) -> the multi-txn filer "Hon. Richard
        # W. Allen" is genuinely ambiguous. Aderholt (single-txn) still resolves normally. The
        # unresolved-ticker fixture's member is entirely absent from this roster -> no_candidate.
        return _combined_repo(
            _SINGLE_TXN_MEMBER,
            _MULTI_TXN_MEMBER,
            "Richard Allen",
        )

    source, client = _source(_handler(), clock, historical_politicians=ambiguous_roster)
    async with client:
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    # The ambiguous filer's PTR produced no signals at all.
    assert all(s.raw_payload["doc_id"] != _MULTI_TXN_DOC_ID for s in signals)
    assert all(s.raw_payload["doc_id"] != _UNRESOLVED_DOC_ID for s in signals)

    summary = source.last_run_summary()
    assert summary is not None
    unmatched = summary["unmatched_filers"]
    assert isinstance(unmatched, dict)
    assert unmatched["ambiguous"] >= 1
    assert unmatched["no_candidate"] >= 1


async def test_last_run_summary_reports_amount_parse_misses(clock: FixedClock) -> None:
    """The unresolved-ticker fixture PDF (``20040001.pdf``) has no recognizable amount bracket,
    so it is a genuine amount-parse miss counted in the summary alongside the resolved fixtures'
    correctly-parsed brackets."""
    source, client = _source(_handler(), clock)
    async with client:
        await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    summary = source.last_run_summary()
    assert summary is not None
    assert isinstance(summary["amount_parse_misses"], int)
    assert isinstance(summary["signals"], int)
    assert isinstance(summary["unresolved_ticker"], int)
    assert isinstance(summary["unparseable_pdfs"], int)
    assert isinstance(summary["paper_filings_skipped"], int)


async def test_last_run_summary_reflects_the_most_recent_call(clock: FixedClock) -> None:
    """A second ``fetch_historical`` call over a narrower window (excluding the ambiguous-roster
    filer entirely) replaces the summary, rather than accumulating across calls."""

    async def empty_roster() -> PoliticianRepositoryPort:
        return StaticPoliticianRepository()

    source, client = _source(_handler(), clock, historical_politicians=empty_roster)
    async with client:
        await source.fetch_historical(_WINDOW_START, _WINDOW_END)
        first_summary = source.last_run_summary()
        assert first_summary is not None
        first_unmatched = first_summary["unmatched_filers"]
        assert isinstance(first_unmatched, dict)
        assert first_unmatched["no_candidate"] >= 1

        # A narrower window with no matching index entries at all -> zero unmatched this time.
        await source.fetch_historical(
            datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
        )
        second_summary = source.last_run_summary()
        assert second_summary is not None
        second_unmatched = second_summary["unmatched_filers"]
        assert isinstance(second_unmatched, dict)
        assert second_unmatched["no_candidate"] == 0
        assert second_unmatched["ambiguous"] == 0
        assert second_summary["signals"] == 0
