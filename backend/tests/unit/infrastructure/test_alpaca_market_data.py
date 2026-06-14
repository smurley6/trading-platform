"""AlpacaMarketData against a mock HTTP transport — request shaping + latest-quote parsing,
no network. Verifies the live feed satisfies MarketDataPort + QuoteFeedPort."""

from collections.abc import Callable
from decimal import Decimal

import httpx
import pytest

from trading.infrastructure.market_data.alpaca import (
    AlpacaMarketData,
    AlpacaMarketDataError,
)

Handler = Callable[[httpx.Request], httpx.Response]

_LATEST = {
    "quotes": {
        "AAPL": {"bp": 232.10, "ap": 232.14, "t": "2026-06-13T20:00:00.123456789Z"},
        "MSFT": {"bp": 410.00, "ap": 410.40, "t": "2026-06-13T20:00:00Z"},
    }
}


def _feed(handler: Handler) -> AlpacaMarketData:
    return AlpacaMarketData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
    )


async def test_get_quote_sends_auth_and_parses_single() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers.get("APCA-API-KEY-ID")
        captured["secret"] = request.headers.get("APCA-API-SECRET-KEY")
        captured["path"] = request.url.path
        captured["symbols"] = request.url.params.get("symbols")
        return httpx.Response(200, json=_LATEST)

    quote = await _feed(handler).get_quote("AAPL")

    assert captured["key"] == "k"
    assert captured["secret"] == "s"
    assert captured["path"] == "/v2/stocks/quotes/latest"
    assert captured["symbols"] == "AAPL"
    assert quote.symbol == "AAPL"
    assert quote.bid == Decimal("232.10")
    assert quote.ask == Decimal("232.14")
    assert quote.last == Decimal("232.12")  # midpoint of bid/ask
    assert quote.as_of.tzinfo is not None


async def test_get_quotes_batch_parses_all() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("symbols") == "AAPL,MSFT"
        return httpx.Response(200, json=_LATEST)

    quotes = await _feed(handler).get_quotes(["aapl", "msft"])  # also normalizes case

    assert {q.symbol for q in quotes} == {"AAPL", "MSFT"}
    by_symbol = {q.symbol: q for q in quotes}
    assert by_symbol["MSFT"].last == Decimal("410.20")


async def test_enabled_is_true() -> None:
    feed = _feed(lambda _r: httpx.Response(200, json={"quotes": {}}))
    assert feed.enabled is True


async def test_empty_symbols_makes_no_request() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be made for empty symbols")

    assert await _feed(handler).get_quotes([]) == []
    assert await _feed(handler).get_quotes(["  "]) == []


async def test_missing_symbol_is_skipped() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"quotes": {"AAPL": {"bp": 1.0, "ap": 2.0}}})

    quotes = await _feed(handler).get_quotes(["AAPL", "ZZZZ"])
    assert [q.symbol for q in quotes] == ["AAPL"]


async def test_get_quote_raises_when_symbol_absent() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"quotes": {}})

    with pytest.raises(AlpacaMarketDataError):
        await _feed(handler).get_quote("AAPL")


async def test_auth_failure_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    with pytest.raises(AlpacaMarketDataError, match="auth failed"):
        await _feed(handler).get_quotes(["AAPL"])


async def test_server_error_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(AlpacaMarketDataError, match="failed"):
        await _feed(handler).get_quotes(["AAPL"])


async def test_one_sided_quote_falls_back_to_present_side() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"quotes": {"AAPL": {"bp": 0, "ap": 232.14}}})

    quote = await _feed(handler).get_quote("AAPL")
    assert quote.last == Decimal("232.14")
