"""House Clerk STOCK Act PTR adapter — offline, committed fixtures + mock transport.

Drives :class:`HouseDisclosuresSource` entirely offline:

* The fetch path goes through ``httpx.MockTransport`` — NEVER the live network. The handler
  serves the committed real index zip and the committed real PTR PDFs by URL.
* The committed fixtures under ``fixtures/house/`` are REAL samples pulled from
  ``disclosures-clerk.house.gov`` (a single-transaction PTR and one or two multi-transaction
  PTRs), plus the real ``2025FD.xml`` index. Reading those committed files is fine; we just
  never reach out over the wire.

The seed roster does not contain the real members named in the PDFs, so each test injects a
``StaticPoliticianRepository`` seeded with a politician whose name matches the fixture member,
which also lets us assert provenance propagation (SYNTHETIC roster -> SYNTHETIC signal, REAL
roster member -> REAL signal), mirroring :mod:`congress_trades`.
"""

import io
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.politicians import DEFAULT_BASE_WEIGHTS, Politician, PoliticianRole
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.house_disclosures import HouseDisclosuresSource
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.infrastructure.reference.sectors import StaticSectorMap

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "house"
_INDEX_XML = _FIXTURES / "2025FD.xml"

# Committed real PTR PDFs. 20032062 is the single-transaction sample (Hon. Robert B. Aderholt,
# a GSK sale); the multi-transaction sample is an e-filed Allen Richard W. PTR.
_SINGLE_TXN_PDF = _FIXTURES / "20032062.pdf"
_SINGLE_TXN_DOC_ID = "20032062"
_SINGLE_TXN_MEMBER = "Robert B. Aderholt"

_MULTI_TXN_PDF = _FIXTURES / "20030502.pdf"
_MULTI_TXN_DOC_ID = "20030502"
_MULTI_TXN_MEMBER = "Richard W. Allen"


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _zip_index(xml_bytes: bytes, *, year: int = 2025) -> bytes:
    """Wrap index XML in a ``{year}FD.zip`` the way the Clerk site serves it."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{year}FD.xml", xml_bytes)
    return buffer.getvalue()


def _politician(name: str, *, provenance: Provenance = Provenance.SYNTHETIC) -> Politician:
    return Politician(
        id="pol-house-1",
        name=name,
        role=PoliticianRole.MEMBER,
        committees=(),
        party="R",
        base_weight=DEFAULT_BASE_WEIGHTS[PoliticianRole.MEMBER],
        track_record_score=1.0,
        provenance=provenance,
    )


def _repo(
    name: str, *, provenance: Provenance = Provenance.SYNTHETIC
) -> StaticPoliticianRepository:
    return StaticPoliticianRepository([_politician(name, provenance=provenance)])


def _handler_for(pdfs: dict[str, bytes], *, index_xml: bytes | None = None) -> Handler:
    """Serve the index zip and PTR PDFs by URL; 404 anything unknown."""
    xml = index_xml if index_xml is not None else _INDEX_XML.read_bytes()
    zip_bytes = _zip_index(xml)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("FD.ZIP") or path.endswith("FD.zip"):
            return httpx.Response(200, content=zip_bytes)
        for doc_id, pdf_bytes in pdfs.items():
            if path.endswith(f"/{doc_id}.pdf"):
                return httpx.Response(200, content=pdf_bytes)
        return httpx.Response(404)

    return handler


def _source(
    handler: Handler,
    repo: StaticPoliticianRepository,
    clock: FixedClock,
) -> tuple[HouseDisclosuresSource, httpx.AsyncClient]:
    client = _client(handler)
    source = HouseDisclosuresSource(
        client=client,
        politicians=repo,
        sector_map=StaticSectorMap(),
        clock=clock,
        years=(2025,),
    )
    return source, client


@pytest.fixture
def clock_2025() -> FixedClock:
    # Place "now" in 2025 so the single-year window (2025FD) is the natural one.
    return FixedClock(datetime(2025, 12, 31, tzinfo=UTC))


def _before_2025() -> datetime:
    return datetime(2025, 1, 1, tzinfo=UTC)


# --- Index parsing -------------------------------------------------------------------------


async def test_index_filters_to_efiled_ptrs(clock_2025: FixedClock) -> None:
    """Only e-filed PTRs (FilingType 'P', DocID starting '2', len >= 8) are fetched."""
    requested_pdf_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("FD.ZIP"):
            return httpx.Response(200, content=_zip_index(_INDEX_XML.read_bytes()))
        requested_pdf_paths.append(path)
        return httpx.Response(404)

    repo = _repo("Zzz Nobody Matches")  # match nothing -> no PDFs fetched
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    assert signals == []
    # Because no member matched, the adapter short-circuits before fetching any PDF.
    assert requested_pdf_paths == []


# --- Single-transaction PDF parsing --------------------------------------------------------


@pytest.mark.skipif(not _SINGLE_TXN_PDF.exists(), reason="single-transaction PTR fixture missing")
async def test_single_transaction_pdf_emits_one_signal(clock_2025: FixedClock) -> None:
    handler = _handler_for({_SINGLE_TXN_DOC_ID: _SINGLE_TXN_PDF.read_bytes()})
    repo = _repo(_SINGLE_TXN_MEMBER)
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    member_signals = [s for s in signals if s.raw_payload["doc_id"] == _SINGLE_TXN_DOC_ID]
    assert len(member_signals) == 1
    signal = member_signals[0]
    assert signal.source_type is SignalSource.DISCLOSED_TRADE
    assert signal.ticker == "GSK"
    assert signal.politician_id == "pol-house-1"
    assert signal.sentiment < 0  # the sample is a sale -> bearish
    assert signal.source_reliability == pytest.approx(0.7)
    assert signal.politician_weight >= 0.0
    # FilingDate is the available timestamp.
    assert signal.created_at.tzinfo is not None
    assert signal.raw_payload["ticker"] == "GSK"


@pytest.mark.skipif(not _SINGLE_TXN_PDF.exists(), reason="single-transaction PTR fixture missing")
async def test_synthetic_roster_member_yields_synthetic_signal(clock_2025: FixedClock) -> None:
    handler = _handler_for({_SINGLE_TXN_DOC_ID: _SINGLE_TXN_PDF.read_bytes()})
    repo = _repo(_SINGLE_TXN_MEMBER, provenance=Provenance.SYNTHETIC)
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    member_signals = [s for s in signals if s.raw_payload["doc_id"] == _SINGLE_TXN_DOC_ID]
    assert member_signals
    assert all(s.provenance is Provenance.SYNTHETIC for s in member_signals)


@pytest.mark.skipif(not _SINGLE_TXN_PDF.exists(), reason="single-transaction PTR fixture missing")
async def test_real_roster_member_yields_real_signal(clock_2025: FixedClock) -> None:
    handler = _handler_for({_SINGLE_TXN_DOC_ID: _SINGLE_TXN_PDF.read_bytes()})
    repo = _repo(_SINGLE_TXN_MEMBER, provenance=Provenance.REAL)
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    member_signals = [s for s in signals if s.raw_payload["doc_id"] == _SINGLE_TXN_DOC_ID]
    assert member_signals
    # Anti-placeholder guard: a REAL member on real disclosure data -> REAL signal.
    assert all(s.provenance is Provenance.REAL for s in member_signals)


@pytest.mark.skipif(not _SINGLE_TXN_PDF.exists(), reason="single-transaction PTR fixture missing")
async def test_unmatched_member_filing_is_skipped(clock_2025: FixedClock) -> None:
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("FD.ZIP"):
            return httpx.Response(200, content=_zip_index(_INDEX_XML.read_bytes()))
        fetched.append(path)
        return httpx.Response(200, content=_SINGLE_TXN_PDF.read_bytes())

    repo = _repo("Nobody In This Roster")
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    assert signals == []
    assert fetched == []  # short-circuit before fetching any PDF


# --- Multi-transaction PDF parsing ---------------------------------------------------------


@pytest.mark.skipif(not _MULTI_TXN_PDF.exists(), reason="multi-transaction PTR fixture missing")
async def test_multi_transaction_pdf_emits_multiple_signals(clock_2025: FixedClock) -> None:
    handler = _handler_for({_MULTI_TXN_DOC_ID: _MULTI_TXN_PDF.read_bytes()})
    repo = _repo(_MULTI_TXN_MEMBER)
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    member_signals = [s for s in signals if s.raw_payload["doc_id"] == _MULTI_TXN_DOC_ID]
    # A richer multi-row PTR yields more than one transaction signal.
    assert len(member_signals) > 1
    assert all(s.source_type is SignalSource.DISCLOSED_TRADE for s in member_signals)
    assert all(s.ticker == s.ticker.upper() for s in member_signals)


# --- Windowing -----------------------------------------------------------------------------


@pytest.mark.skipif(not _SINGLE_TXN_PDF.exists(), reason="single-transaction PTR fixture missing")
async def test_filings_at_or_before_cursor_are_excluded(clock_2025: FixedClock) -> None:
    handler = _handler_for({_SINGLE_TXN_DOC_ID: _SINGLE_TXN_PDF.read_bytes()})
    repo = _repo(_SINGLE_TXN_MEMBER)
    source, client = _source(handler, repo, clock_2025)
    async with client:
        # A cursor in the future relative to every 2025 FilingDate -> nothing fresh.
        signals = await source.fetch_signals(datetime(2026, 12, 31, tzinfo=UTC))

    assert signals == []


async def test_missing_year_index_is_tolerated(clock_2025: FixedClock) -> None:
    """A 404 on the year index zip must not crash the poll (returns no signals)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    repo = _repo(_SINGLE_TXN_MEMBER)
    source, client = _source(handler, repo, clock_2025)
    async with client:
        signals = await source.fetch_signals(_before_2025())

    assert signals == []
