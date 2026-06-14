"""Data-source adapters driven by a mock HTTP transport — real parsing, no network."""

from collections.abc import Callable

import httpx
import pytest

from tests.conftest import FixedClock
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


async def test_usaspending_source_emits_contract_signal(clock: FixedClock) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "Award ID": "C-1",
                        "Recipient Name": "Lockheed Martin Corporation",
                        "Award Amount": 5_000_000_000,
                        "Action Date": "2026-06-09",
                    }
                ]
            },
        )

    async with _client(handler) as client:
        source = UsaSpendingSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock
        )
        signals = await source.fetch_signals(clock.now().replace(month=1))

    assert "api.usaspending.gov" in str(captured["url"])
    assert len(signals) == 1
    assert signals[0].source_type is SignalSource.CONTRACT
    assert signals[0].ticker == "LMT"
    assert signals[0].politician_id is None
    assert signals[0].sentiment > 0
    assert signals[0].politician_weight > 1.0  # large award -> elevated weight


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
