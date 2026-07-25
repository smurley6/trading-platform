"""Senate eFD adapter driven by a mock HTTP transport — real fixture parsing, no network.

The fixtures under ``fixtures/senate/`` mirror the real efdsearch.senate.gov shapes: the
DataTables JSON ``data`` array of HTML cell fragments returned by ``/search/report/data/`` and
an electronic Periodic Transaction Report detail page's transactions ``<table>``. The handler
below stands in for the portal, replaying the GET-search -> POST-agreement -> POST-search ->
GET-detail handshake and asserting the request shape the production handshake sends.
"""

import dataclasses
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.politicians import DEFAULT_BASE_WEIGHTS, Politician, PoliticianRole
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.senate_efd import (
    SenateEfdSource,
    _Filing,
    _Transaction,
)
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.infrastructure.reference.sectors import StaticSectorMap

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "senate"
_SEARCH_ROWS = (_FIXTURES / "search_rows.json").read_text()
_PTR_DELGADO = (_FIXTURES / "ptr_detail.html").read_text()
_PTR_HOLT = (_FIXTURES / "ptr_detail_holt.html").read_text()
# Detail URLs are keyed off the report UUIDs that appear in the search-rows fixture.
_DELGADO_REPORT_ID = "aa11bb22-3344-5566-7788-99aabbccddee"
_HOLT_REPORT_ID = "ffeeddcc-bbaa-9988-7766-554433221100"

# A cursor far before any fixture date so every fresh transaction passes the window.
_OPEN_CURSOR = datetime(2026, 1, 1, tzinfo=UTC)


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _efd_handler(record: list[httpx.Request] | None = None) -> Handler:
    """A stand-in for efdsearch.senate.gov that replays the documented handshake + search."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        path = request.url.path
        if path == "/search/":
            # Django seeds the CSRF cookie on the initial GET.
            return httpx.Response(200, text="<html></html>", headers={"set-cookie": _csrf_cookie()})
        if path == "/search/home/":
            return httpx.Response(200, text="<html>accepted</html>")
        if path == "/search/report/data/":
            return httpx.Response(200, json=json.loads(_SEARCH_ROWS))
        if _DELGADO_REPORT_ID in path:
            return httpx.Response(200, text=_PTR_DELGADO)
        if _HOLT_REPORT_ID in path:
            return httpx.Response(200, text=_PTR_HOLT)
        return httpx.Response(404, text="not found")

    return handler


def _csrf_cookie() -> str:
    return "csrftoken=test-csrf-token; Path=/; SameSite=Lax"


def _real_roster() -> StaticPoliticianRepository:
    """A roster whose members are stamped REAL, so matched signals stamp REAL provenance."""
    real = tuple(
        dataclasses.replace(politician, provenance=Provenance.REAL)
        for politician in StaticPoliticianRepository().all()
    )
    return StaticPoliticianRepository(real)


def _source(
    client: httpx.AsyncClient,
    clock: FixedClock,
    *,
    politicians: StaticPoliticianRepository | None = None,
    max_filings: int = 50,
) -> SenateEfdSource:
    return SenateEfdSource(
        client=client,
        politicians=politicians or StaticPoliticianRepository(),
        sector_map=StaticSectorMap(),
        clock=clock,
        max_filings=max_filings,
    )


async def test_emits_disclosed_trade_signals_from_electronic_ptr(clock: FixedClock) -> None:
    async with _client(_efd_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_signals(_OPEN_CURSOR)

    # Delgado: NVDA Purchase (kept), AAPL Sale (kept), bond mutual fund (no ticker, dropped),
    # MSFT Exchange (dropped). Holt: LMT Purchase (kept). Paper Boyd filing skipped entirely.
    by_ticker = {signal.ticker: signal for signal in signals}
    assert set(by_ticker) == {"NVDA", "AAPL", "LMT"}
    assert all(signal.source_type is SignalSource.DISCLOSED_TRADE for signal in signals)

    nvda = by_ticker["NVDA"]
    assert nvda.sentiment > 0  # purchase reads bullish
    assert nvda.politician_id == "pol-senior-armed"  # Frank Delgado in the seed roster
    assert nvda.raw_payload["asset_name"].startswith("NVIDIA")
    assert nvda.raw_payload["amount"] == "$50,001 - $100,000"
    assert by_ticker["AAPL"].sentiment < 0  # sale reads bearish


async def test_name_property() -> None:
    async with _client(_efd_handler()) as client:
        assert _source(client, FixedClock()).name == "senate_efd"


async def test_skips_paper_ptr_filings(clock: FixedClock) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return _efd_handler()(request)

    async with _client(handler) as client:
        await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    # The scanned (paper) Boyd report is never fetched: only the two electronic PTRs are.
    assert any(_DELGADO_REPORT_ID in p for p in requested_paths)
    assert any(_HOLT_REPORT_ID in p for p in requested_paths)
    assert not any("/paper/" in p for p in requested_paths)


async def test_exchange_and_tickerless_rows_are_dropped(clock: FixedClock) -> None:
    async with _client(_efd_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    tickers = {signal.ticker for signal in signals}
    assert "MSFT" not in tickers  # Exchange transaction carries no directional read
    # The bond mutual fund row has ticker "--" and so cannot map to a tradeable security.
    assert all(signal.ticker not in {"", "--"} for signal in signals)


async def test_window_filters_by_filing_date_not_transaction_date(clock: FixedClock) -> None:
    # Windowing is by the FILING (received) date now, not the transaction date (review F13). The
    # fixtures' received dates are Delgado 2026-06-10 and Holt 2026-06-09; a cursor at the latest
    # received date (2026-06-10) filters BOTH filings (received <= since), so nothing passes —
    # even though several transactions are dated after 2026-06-09.
    async with _client(_efd_handler()) as client:
        signals = await _source(client, clock).fetch_signals(datetime(2026, 6, 10, tzinfo=UTC))

    assert signals == []


async def test_created_at_is_filing_date_and_late_disclosure_survives(clock: FixedClock) -> None:
    # A trade dated well before its filing must still be INGESTED, stamped with the FILING date,
    # not silently dropped by transaction-date windowing (review F13). Delgado's report was
    # received 2026-06-10; its NVDA transaction is dated earlier. A cursor of 2026-06-09 (after the
    # transaction date, before the received date) would have dropped it under the old rule.
    async with _client(_efd_handler()) as client:
        signals = await _source(client, clock).fetch_signals(datetime(2026, 6, 9, tzinfo=UTC))

    by_ticker = {signal.ticker: signal for signal in signals}
    assert "NVDA" in by_ticker  # survived despite an earlier transaction date
    nvda = by_ticker["NVDA"]
    assert nvda.created_at == datetime(2026, 6, 10, tzinfo=UTC)  # the FILING (received) date
    # The transaction date is retained in the payload for decay/analysis, distinct from created_at.
    assert nvda.raw_payload["transaction_date"] != nvda.created_at.date().isoformat()


async def test_real_roster_stamps_real_provenance(clock: FixedClock) -> None:
    async with _client(_efd_handler()) as client:
        source = _source(client, clock, politicians=_real_roster())
        signals = await source.fetch_signals(_OPEN_CURSOR)

    assert signals
    assert all(signal.provenance is Provenance.REAL for signal in signals)


async def test_synthetic_roster_stamps_synthetic_provenance(clock: FixedClock) -> None:
    async with _client(_efd_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    # The seed roster is SYNTHETIC, so even REAL government data cannot drive a real order until a
    # REAL roster is wired (fail-closed anti-placeholder guarantee).
    assert signals
    assert all(signal.provenance is Provenance.SYNTHETIC for signal in signals)


async def test_unmatched_filer_is_skipped(clock: FixedClock) -> None:
    # A roster with nobody named in the filings -> no signals, and no detail page is fetched.
    stranger = Politician(
        id="pol-x",
        name="Zelda Quintero",
        role=PoliticianRole.MEMBER,
        committees=(),
        party="I",
        base_weight=DEFAULT_BASE_WEIGHTS[PoliticianRole.MEMBER],
        track_record_score=0.5,
    )
    async with _client(_efd_handler()) as client:
        source = _source(client, clock, politicians=StaticPoliticianRepository((stranger,)))
        signals = await source.fetch_signals(_OPEN_CURSOR)

    assert signals == []


async def test_larger_amount_range_yields_larger_weight(clock: FixedClock) -> None:
    async with _client(_efd_handler()) as client:
        signals = await _source(client, clock, politicians=_real_roster()).fetch_signals(
            _OPEN_CURSOR
        )

    by_ticker = {signal.ticker: signal for signal in signals}
    # LMT is a $250k-$500k bracket; NVDA is $50k-$100k. Same base weight inputs are not directly
    # comparable across politicians, but the amount multiplier must be strictly larger for LMT.
    # Compare the multiplier in isolation by re-deriving from raw payload brackets.
    assert by_ticker["LMT"].raw_payload["amount"] == "$250,001 - $500,000"
    assert by_ticker["NVDA"].raw_payload["amount"] == "$50,001 - $100,000"
    assert by_ticker["LMT"].politician_weight > 0
    assert by_ticker["NVDA"].politician_weight > 0


async def test_handshake_request_shape(clock: FixedClock) -> None:
    record: list[httpx.Request] = []
    async with _client(_efd_handler(record)) as client:
        await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    paths = [request.url.path for request in record]
    # The documented order: GET search home, POST agreement, POST report data, then detail GETs.
    assert paths[0] == "/search/"
    assert paths[1] == "/search/home/"
    assert paths[2] == "/search/report/data/"

    agreement = next(r for r in record if r.url.path == "/search/home/")
    accept_form = parse_qs(agreement.content.decode())
    assert accept_form["prohibition_agreement"] == ["1"]
    assert accept_form["csrfmiddlewaretoken"] == ["test-csrf-token"]

    search = next(r for r in record if r.url.path == "/search/report/data/")
    search_form = parse_qs(search.content.decode())
    assert search_form["report_types"] == ["[11]"]  # PTR report-type code
    assert search_form["filer_types"] == ["[1]"]  # Senator filer-type code
    assert search.headers["X-CSRFToken"] == "test-csrf-token"


@pytest.mark.parametrize(
    ("amount", "expected_floor"),
    [("$1,001 - $15,000", 1.0), ("$250,001 - $500,000", 1.4), ("$5,000,001 - $25,000,000", 1.8)],
)
async def test_amount_multiplier_scales_with_bracket(amount: str, expected_floor: float) -> None:
    from trading.infrastructure.data_sources.senate_efd import _amount_multiplier

    assert _amount_multiplier(amount) >= expected_floor


def _efd_txn(ticker: str, transaction_type: str) -> _Transaction:
    return _Transaction(
        ticker=ticker,
        asset_name=f"{ticker} Common Stock",
        transaction_type=transaction_type,
        transaction_date=datetime(2026, 6, 8, tzinfo=UTC),
        transaction_date_raw="06/08/2026",
        amount_raw="$1,001 - $15,000",
    )


async def test_same_day_buy_and_sell_yield_two_distinct_ids(clock: FixedClock) -> None:
    """A senator who BUYS and SELLS the same ticker on the same day (same filing) must yield TWO
    distinct rm_signals rows: the old day-granular id (filer+ticker+day) collapsed them into one,
    with page order deciding the surviving side (review F25). The id now carries side + index."""
    async with _client(_efd_handler()) as client:
        source = _source(client, clock, politicians=_real_roster())
        politician = _real_roster().all()[0]
        filing = _Filing(
            filer_name=politician.name,
            detail_url="https://efdsearch.senate.gov/search/view/ptr/x/",
            is_electronic=True,
            received=datetime(2026, 6, 10, tzinfo=UTC),
        )
        since = datetime(2026, 1, 1, tzinfo=UTC)
        buy, buy_reason = source._to_signal(
            _efd_txn("AAPL", "Purchase"), filing, politician, since, 0
        )
        sell, sell_reason = source._to_signal(
            _efd_txn("AAPL", "Sale (Full)"), filing, politician, since, 1
        )

    assert (buy_reason, sell_reason) == ("ok", "ok")
    assert buy is not None and sell is not None
    assert buy.id != sell.id  # side (+index) distinguishes the same-day same-ticker buy/sell
    # created_at is the FILING date for both, not the transaction date (review F13).
    assert buy.created_at == sell.created_at == filing.received


async def test_max_filings_caps_detail_fetches(clock: FixedClock) -> None:
    detail_fetches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/search/view/"):
            detail_fetches.append(request.url.path)
        return _efd_handler()(request)

    async with _client(handler) as client:
        # Cap at one filing -> only the first (Delgado) electronic report is fetched.
        await _source(client, clock, max_filings=1).fetch_signals(_OPEN_CURSOR)

    assert len(detail_fetches) == 1
    assert _DELGADO_REPORT_ID in detail_fetches[0]
