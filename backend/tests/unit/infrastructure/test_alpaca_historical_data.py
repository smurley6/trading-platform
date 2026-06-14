"""AlpacaHistoricalData against a mock HTTP transport — request shaping, daily-bar parsing
into a PriceSeries, pagination across next_page_token, and error handling. No network."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from trading.infrastructure.market_data.alpaca_history import (
    AlpacaHistoricalData,
    AlpacaHistoricalDataError,
)

Handler = Callable[[httpx.Request], httpx.Response]

_START = datetime(2026, 1, 1, tzinfo=UTC)
_END = datetime(2026, 1, 3, tzinfo=UTC)

_BARS = {
    "bars": {
        "AAPL": [
            {"t": "2026-01-01T05:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10},
            {"t": "2026-01-02T05:00:00Z", "o": 100.5, "h": 103, "l": 100, "c": 102.25, "v": 12},
        ],
        "MSFT": [
            {"t": "2026-01-01T05:00:00Z", "o": 400, "h": 405, "l": 399, "c": 404, "v": 8},
        ],
    },
    "next_page_token": None,
}


def _provider(handler: Handler) -> AlpacaHistoricalData:
    return AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
    )


async def test_sends_auth_and_params_and_parses_close() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers.get("APCA-API-KEY-ID")
        captured["secret"] = request.headers.get("APCA-API-SECRET-KEY")
        captured["path"] = request.url.path
        captured["symbols"] = request.url.params.get("symbols")
        captured["timeframe"] = request.url.params.get("timeframe")
        captured["start"] = request.url.params.get("start")
        captured["end"] = request.url.params.get("end")
        return httpx.Response(200, json=_BARS)

    series = await _provider(handler).price_series(["AAPL", "MSFT"], _START, _END)

    assert captured["key"] == "k"
    assert captured["secret"] == "s"
    assert captured["path"] == "/v2/stocks/bars"
    assert captured["symbols"] == "AAPL,MSFT"
    assert captured["timeframe"] == "1Day"
    assert captured["start"] == "2026-01-01T00:00:00Z"
    assert captured["end"] == "2026-01-03T00:00:00Z"

    # Bar close becomes the price point per timestamp.
    assert series.price_at("AAPL", datetime(2026, 1, 1, 12, tzinfo=UTC)) == Decimal("100.5")
    assert series.price_at("AAPL", datetime(2026, 1, 2, 12, tzinfo=UTC)) == Decimal("102.25")
    assert series.price_at("MSFT", datetime(2026, 1, 1, 12, tzinfo=UTC)) == Decimal("404")


async def test_single_symbol_parses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("symbols") == "AAPL"
        return httpx.Response(
            200,
            json={
                "bars": {"AAPL": [{"t": "2026-01-01T05:00:00Z", "c": 100.5}]},
                "next_page_token": None,
            },
        )

    series = await _provider(handler).price_series(["aapl"], _START, _END)  # normalizes case
    assert series.symbols() == ("AAPL",)
    assert series.price_at("AAPL", _END) == Decimal("100.5")


async def test_pagination_concatenates_pages() -> None:
    pages = [
        {
            "bars": {"AAPL": [{"t": "2026-01-01T05:00:00Z", "c": 100.5}]},
            "next_page_token": "tok-2",
        },
        {
            "bars": {"AAPL": [{"t": "2026-01-02T05:00:00Z", "c": 102.25}]},
            "next_page_token": None,
        },
    ]
    seen_tokens: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("page_token")
        seen_tokens.append(token)
        return httpx.Response(200, json=pages[len(seen_tokens) - 1])

    series = await _provider(handler).price_series(["AAPL"], _START, _END)

    assert seen_tokens == [None, "tok-2"]  # first page tokenless, second uses next_page_token
    assert series.price_at("AAPL", datetime(2026, 1, 1, 12, tzinfo=UTC)) == Decimal("100.5")
    assert series.price_at("AAPL", datetime(2026, 1, 2, 12, tzinfo=UTC)) == Decimal("102.25")


async def test_empty_symbols_makes_no_request() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be made for empty symbols")

    series = await _provider(handler).price_series([], _START, _END)
    assert series.symbols() == ()
    series = await _provider(handler).price_series(["  "], _START, _END)
    assert series.symbols() == ()


async def test_missing_symbol_simply_absent() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bars": {"AAPL": [{"t": "2026-01-01T05:00:00Z", "c": 100.5}]},
                "next_page_token": None,
            },
        )

    series = await _provider(handler).price_series(["AAPL", "ZZZZ"], _START, _END)
    assert series.symbols() == ("AAPL",)
    assert series.price_at("ZZZZ", _END) is None


async def test_empty_results() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    series = await _provider(handler).price_series(["AAPL"], _START, _END)
    assert series.symbols() == ()


async def test_auth_failure_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    with pytest.raises(AlpacaHistoricalDataError, match="auth failed"):
        await _provider(handler).price_series(["AAPL"], _START, _END)


async def test_server_error_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(AlpacaHistoricalDataError, match="failed"):
        await _provider(handler).price_series(["AAPL"], _START, _END)


async def test_naive_datetimes_are_serialized_as_utc() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["start"] = request.url.params.get("start")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    naive_start = datetime(2026, 1, 1)  # noqa: DTZ001 - deliberately naive input
    naive_end = datetime(2026, 1, 3)  # noqa: DTZ001 - deliberately naive input
    await _provider(handler).price_series(["AAPL"], naive_start, naive_end)
    assert captured["start"] == "2026-01-01T00:00:00Z"
