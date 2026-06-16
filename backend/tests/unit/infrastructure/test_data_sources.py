"""Data-source adapters driven by a mock HTTP transport — real parsing, no network."""

import json
from collections.abc import Callable

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.congress_trades import CongressTradesSource
from trading.infrastructure.data_sources.news import NewsSource
from trading.infrastructure.data_sources.usaspending import UsaSpendingSource
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver
from trading.infrastructure.nlp.sentiment import LexiconSentiment
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.infrastructure.reference.sectors import StaticSectorMap

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _decode_body(request: httpx.Request) -> dict[str, object]:
    """Decode a JSON request body into a typed mapping (keeps the suite free of `Any`)."""
    decoded: object = json.loads(request.content)
    assert isinstance(decoded, dict)
    return decoded


def _requested_page(request: httpx.Request) -> int:
    """Read the 1-based `page` field out of a spending_by_award request body."""
    page = _decode_body(request)["page"]
    assert isinstance(page, int)
    return page


# "Margaret Holt" chairs Armed Services in the seed roster; LMT is a defense ticker.
_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Chair Margaret Holt backs new Lockheed Martin funding</title>
    <description>Strong support for the program, she said.</description>
    <link>http://example.test/1</link>
    <pubDate>Wed, 10 Jun 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Weather update</title>
    <description>Sunny in the capital.</description>
    <pubDate>Wed, 10 Jun 2026 12:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


@pytest.fixture
def politicians() -> StaticPoliticianRepository:
    return StaticPoliticianRepository()


async def test_news_source_emits_statement_signal(
    clock: FixedClock, politicians: StaticPoliticianRepository
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_RSS)

    async with _client(handler) as client:
        source = NewsSource(
            client=client,
            feed_urls=["http://example.test/feed.xml"],
            entity_resolver=GazetteerEntityResolver(),
            sentiment=LexiconSentiment(),
            politicians=politicians,
            sector_map=StaticSectorMap(),
            clock=clock,
        )
        signals = await source.fetch_signals(clock.now().replace(year=2026, month=1))

    assert len(signals) == 1
    signal = signals[0]
    assert signal.source_type is SignalSource.STATEMENT
    assert signal.ticker == "LMT"
    assert signal.politician_id == "pol-chair-armed"
    assert signal.sentiment > 0
    # Armed Services chair on a defense name => full topic relevance, weight > 1.
    assert signal.politician_weight > 1.0
    assert signal.raw_payload["title"].startswith("Chair Margaret Holt")


async def test_news_source_skips_items_before_cursor(
    clock: FixedClock, politicians: StaticPoliticianRepository
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_RSS)

    async with _client(handler) as client:
        source = NewsSource(
            client=client,
            feed_urls=["http://example.test/feed.xml"],
            entity_resolver=GazetteerEntityResolver(),
            sentiment=LexiconSentiment(),
            politicians=politicians,
            sector_map=StaticSectorMap(),
            clock=clock,
        )
        # Cursor in the future relative to the item's pubDate -> nothing fresh.
        signals = await source.fetch_signals(clock.now())

    assert signals == []


def _usaspending_page(
    results: list[dict[str, object]], *, has_next: bool, page: int = 1
) -> dict[str, object]:
    """A REAL-shaped spending_by_award response: rows under "results", paging metadata."""
    return {
        "results": results,
        "page_metadata": {"page": page, "hasNext": has_next, "hasPrevious": page > 1},
    }


# Real-shaped rows: uppercased legal recipient names, keyed by the requested display fields.
_LOCKHEED_ROW: dict[str, object] = {
    "Award ID": "C-1",
    "Recipient Name": "LOCKHEED MARTIN CORPORATION",
    "Award Amount": 5_000_000_000,
    "Action Date": "2026-06-09",
}
_BOOZ_ALLEN_ROW: dict[str, object] = {
    "Award ID": "C-2",
    "Recipient Name": "BOOZ ALLEN HAMILTON INC.",
    "Award Amount": 250_000_000,
    "Action Date": "2026-06-10",
}


async def test_usaspending_source_emits_contract_signal(clock: FixedClock) -> None:
    urls: list[str] = []
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        bodies.append(_decode_body(request))
        return httpx.Response(200, json=_usaspending_page([_LOCKHEED_ROW], has_next=False))

    async with _client(handler) as client:
        source = UsaSpendingSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock
        )
        signals = await source.fetch_signals(clock.now().replace(month=1))

    assert "api.usaspending.gov" in urls[0]
    # Request body matches the spending_by_award contract: contract award-type filter, the four
    # display fields (which become the row keys), amount-desc sort, and a 1-based page.
    body = bodies[0]
    filters = body["filters"]
    assert isinstance(filters, dict)
    assert filters["award_type_codes"] == ["A", "B", "C", "D"]
    assert body["fields"] == ["Award ID", "Recipient Name", "Award Amount", "Action Date"]
    assert body["sort"] == "Award Amount"
    assert body["order"] == "desc"
    assert body["page"] == 1

    assert len(signals) == 1
    assert signals[0].source_type is SignalSource.CONTRACT
    assert signals[0].ticker == "LMT"
    assert signals[0].politician_id is None
    assert signals[0].sentiment > 0
    assert signals[0].politician_weight > 1.0  # large award -> elevated weight
    # A federal contract is real data with no politician dependency -> REAL provenance, so it can
    # drive a real decision past the anti-placeholder guard.
    assert signals[0].provenance is Provenance.REAL


async def test_usaspending_resolves_multiple_real_contractors(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_usaspending_page([_LOCKHEED_ROW, _BOOZ_ALLEN_ROW], has_next=False)
        )

    async with _client(handler) as client:
        source = UsaSpendingSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock
        )
        signals = await source.fetch_signals(clock.now().replace(month=1))

    by_ticker = {signal.ticker: signal for signal in signals}
    assert set(by_ticker) == {"LMT", "BAH"}
    assert all(signal.source_type is SignalSource.CONTRACT for signal in signals)
    assert all(signal.provenance is Provenance.REAL for signal in signals)
    # Larger award -> strictly larger amount weight.
    assert by_ticker["LMT"].politician_weight > by_ticker["BAH"].politician_weight


async def test_usaspending_skips_rows_at_or_before_cursor(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_usaspending_page([_LOCKHEED_ROW], has_next=False))

    async with _client(handler) as client:
        source = UsaSpendingSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock
        )
        # Cursor after the award's action date -> nothing fresh.
        signals = await source.fetch_signals(clock.now())

    assert signals == []


async def test_usaspending_paginates_until_has_next_false(clock: FixedClock) -> None:
    pages_requested: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = _requested_page(request)
        pages_requested.append(page)
        # Page 1 advertises another page (a distinct award); page 2 is the last.
        if page == 1:
            body = _usaspending_page([_LOCKHEED_ROW], has_next=True, page=1)
        else:
            body = _usaspending_page([_BOOZ_ALLEN_ROW], has_next=False, page=2)
        return httpx.Response(200, json=body)

    async with _client(handler) as client:
        source = UsaSpendingSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock, max_pages=5
        )
        signals = await source.fetch_signals(clock.now().replace(month=1))

    # Walked exactly two pages, then stopped on hasNext=False (did not exhaust max_pages).
    assert pages_requested == [1, 2]
    assert {signal.ticker for signal in signals} == {"LMT", "BAH"}


async def test_usaspending_respects_max_pages_cap(clock: FixedClock) -> None:
    pages_requested: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = _requested_page(request)
        pages_requested.append(page)
        # Always advertise more pages; only the max_pages cap should stop the walk.
        return httpx.Response(
            200, json=_usaspending_page([_LOCKHEED_ROW], has_next=True, page=page)
        )

    async with _client(handler) as client:
        source = UsaSpendingSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock, max_pages=3
        )
        await source.fetch_signals(clock.now().replace(month=1))

    assert pages_requested == [1, 2, 3]


async def test_congress_trades_source_emits_disclosed_trade_signal(
    clock: FixedClock, politicians: StaticPoliticianRepository
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "NVDA",
                    "transactionDate": "2026-06-08",
                    "type": "Purchase",
                    "firstName": "Frank",
                    "lastName": "Delgado",
                }
            ],
        )

    async with _client(handler) as client:
        source = CongressTradesSource(
            client=client,
            endpoint="http://provider.test/senate-trading",
            api_key="test-key",
            politicians=politicians,
            sector_map=StaticSectorMap(),
            clock=clock,
        )
        signals = await source.fetch_signals(clock.now().replace(month=1))

    assert len(signals) == 1
    assert signals[0].source_type is SignalSource.DISCLOSED_TRADE
    assert signals[0].ticker == "NVDA"
    assert signals[0].sentiment > 0  # a purchase reads bullish
    assert signals[0].politician_id == "pol-senior-armed"


async def test_congress_trades_sale_is_bearish(
    clock: FixedClock, politicians: StaticPoliticianRepository
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "NVDA",
                    "transactionDate": "2026-06-08",
                    "type": "Sale (Full)",
                    "firstName": "Frank",
                    "lastName": "Delgado",
                }
            ],
        )

    async with _client(handler) as client:
        source = CongressTradesSource(
            client=client,
            endpoint="http://provider.test/senate-trading",
            api_key="test-key",
            politicians=politicians,
            sector_map=StaticSectorMap(),
            clock=clock,
        )
        signals = await source.fetch_signals(clock.now().replace(month=1))

    assert signals[0].sentiment < 0
