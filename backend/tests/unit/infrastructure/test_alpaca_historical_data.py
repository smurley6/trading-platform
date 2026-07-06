"""AlpacaHistoricalData against a mock HTTP transport — request shaping, daily-bar parsing
into a PriceSeries, pagination across next_page_token, error handling, the Task 09 end-clamp,
and the optional feed override. No network."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from tests.conftest import FixedClock
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


async def test_401_mentions_check_credentials() -> None:
    # 401 is a genuine credential problem -- "check credentials" is accurate and kept (Task 09).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(AlpacaHistoricalDataError, match="check credentials") as excinfo:
        await _provider(handler).price_series(["AAPL"], _START, _END)
    assert excinfo.value.status_code == 401
    assert excinfo.value.body == "unauthorized"


async def test_403_propagates_real_body_not_check_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Task 09: a 403 is NOT assumed to be a credentials problem (it may be a subscription/
    # entitlement error, e.g. Alpaca's "subscription does not permit querying recent SIP data").
    # The real body must propagate verbatim -- the old "check credentials" wording masked this.
    body = '{"message":"subscription does not permit querying recent SIP data"}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=body)

    with (
        caplog.at_level("ERROR"),
        pytest.raises(AlpacaHistoricalDataError) as excinfo,
    ):
        await _provider(handler).price_series(["AAPL"], _START, _END)

    assert "check credentials" not in str(excinfo.value)
    assert body in str(excinfo.value)
    assert excinfo.value.status_code == 403
    assert excinfo.value.body == body
    # Logged at ERROR before raising.
    assert any(body in record.message for record in caplog.records)


async def test_error_body_truncated_to_500_chars() -> None:
    huge_body = "x" * 2000

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=huge_body)

    with pytest.raises(AlpacaHistoricalDataError) as excinfo:
        await _provider(handler).price_series(["AAPL"], _START, _END)
    assert len(excinfo.value.body) == 500
    assert excinfo.value.body == huge_body[:500]


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


# --- Task 09: header-construction regression (spec item 4) -----------------------------------


async def test_header_construction_pins_exact_alpaca_auth_headers() -> None:
    """Pins the EXACT header names/values sent, from Settings-shaped key_id/secret_key -- the
    probe (task spec) already validated this behavior; this regression test pins it so a future
    refactor cannot silently rename/drop the auth headers."""
    captured_headers: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers["APCA-API-KEY-ID"] = request.headers.get("APCA-API-KEY-ID")
        captured_headers["APCA-API-SECRET-KEY"] = request.headers.get("APCA-API-SECRET-KEY")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="PKTESTKEYID123",
        secret_key="test-secret-value-456",
        base_url="https://data.alpaca.markets",
    )
    await provider.price_series(["AAPL"], _START, _END)

    assert captured_headers == {
        "APCA-API-KEY-ID": "PKTESTKEYID123",
        "APCA-API-SECRET-KEY": "test-secret-value-456",
    }


# --- Task 09: end clamping ---------------------------------------------------------------------


async def test_future_end_clamps_to_now_minus_16_minutes_under_fixed_clock() -> None:
    clock = FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC))
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["end"] = request.url.params.get("end")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
        clock=clock,
    )
    future_end = datetime(2026, 9, 26, 23, 5, 37, tzinfo=UTC)  # the app-shaped 403 case
    await provider.price_series(["AAPL"], _START, future_end)

    expected = clock.now() - timedelta(minutes=16)
    assert captured["end"] == expected.isoformat().replace("+00:00", "Z")


async def test_past_end_before_clamp_boundary_is_unchanged() -> None:
    clock = FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC))
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["end"] = request.url.params.get("end")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
        clock=clock,
    )
    # Well before now - 16min: the requested end wins the min(), unclamped.
    await provider.price_series(["AAPL"], _START, _END)
    assert captured["end"] == "2026-01-03T00:00:00Z"


async def test_end_exactly_at_clamp_boundary_is_unchanged() -> None:
    clock = FixedClock(datetime(2026, 7, 6, 12, 0, tzinfo=UTC))
    boundary = clock.now() - timedelta(minutes=16)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["end"] = request.url.params.get("end")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
        clock=clock,
    )
    await provider.price_series(["AAPL"], _START, boundary)
    assert captured["end"] == boundary.isoformat().replace("+00:00", "Z")


async def test_no_injected_clock_falls_back_to_wall_clock_and_still_clamps() -> None:
    # No clock injected: falls back to datetime.now(UTC) directly (documented decision). A
    # far-future requested end must still be clamped to (real) now - 16min, not sent verbatim.
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["end"] = request.url.params.get("end")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
    )
    far_future = datetime(2099, 1, 1, tzinfo=UTC)
    await provider.price_series(["AAPL"], _START, far_future)

    sent_end = datetime.fromisoformat(str(captured["end"]).replace("Z", "+00:00"))
    assert sent_end < far_future
    assert sent_end < datetime.now(UTC)


# --- Task 09: optional feed override -----------------------------------------------------------


async def test_no_feed_setting_sends_no_feed_param() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["has_feed"] = "feed" in request.url.params
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    await _provider(handler).price_series(["AAPL"], _START, _END)
    assert captured["has_feed"] is False


async def test_feed_setting_passed_through_as_query_param() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["feed"] = request.url.params.get("feed")
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    provider = AlpacaHistoricalData(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
        feed="iex",
    )
    await provider.price_series(["AAPL"], _START, _END)
    assert captured["feed"] == "iex"
