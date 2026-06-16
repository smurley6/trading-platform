"""Senate LDA lobbying adapter driven by a mock HTTP transport — real parsing of a committed
real-shaped LDA filings page, real pagination via the `next` URL, no network."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.lobbying import RELIABILITY_LOBBYING, LobbyingSource
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "lobbying"
# A cursor before every filing in the committed page (all dt_posted are Dec 2024).
_BEFORE_ALL = datetime(2024, 1, 1, tzinfo=UTC)


def _load(name: str) -> dict[str, Any]:
    decoded: object = json.loads((_FIXTURES / name).read_text())
    assert isinstance(decoded, dict)
    return decoded


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _source(client: httpx.AsyncClient, clock: FixedClock, **kwargs: Any) -> LobbyingSource:
    return LobbyingSource(
        client=client,
        entity_resolver=GazetteerEntityResolver(),
        clock=clock,
        **kwargs,
    )


async def test_emits_one_signal_per_resolvable_client(clock: FixedClock) -> None:
    """Page 1 has four filings; three clients resolve (LMT, MSFT, BA), one (a water authority)
    does not and is skipped. We do not follow `next` here because max_pages=1."""
    page1 = _load("filings_page1.json")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=page1)

    async with _client(handler) as client:
        signals = await _source(client, clock, max_pages=1).fetch_signals(_BEFORE_ALL)

    by_ticker = {signal.ticker: signal for signal in signals}
    assert set(by_ticker) == {"LMT", "MSFT", "BA"}
    lmt = by_ticker["LMT"]
    assert lmt.source_type is SignalSource.STATEMENT
    assert lmt.politician_id is None
    assert lmt.sentiment == 0.25  # mild positive: active engagement, not euphoric
    assert lmt.source_reliability == RELIABILITY_LOBBYING == 0.5
    # Real federal disclosure, no politician dependency -> REAL, so it can pass the guard.
    assert lmt.provenance is Provenance.REAL
    # raw_payload retains the whole filing (FR-ING-3).
    assert lmt.raw_payload["client"]["name"] == "LOCKHEED MARTIN CORPORATION"
    # dt_posted "2024-12-18T16:42:11.482915-05:00" normalized to UTC -> 21:42:11.
    assert lmt.created_at == datetime(2024, 12, 18, 21, 42, 11, 482915, tzinfo=UTC)


async def test_first_request_shapes_query_and_no_auth_header_when_anonymous(
    clock: FixedClock,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_load("filings_page1.json"))

    async with _client(handler) as client:
        await _source(client, clock, max_pages=1).fetch_signals(_BEFORE_ALL)

    assert captured["url"].startswith("https://lda.senate.gov/api/v1/filings/")
    assert captured["params"]["ordering"] == "-dt_posted"
    # filing_year spans the lookback window (FixedClock is mid-2026 -> includes 2026).
    assert "2026" in captured["params"]["filing_year"]
    # Anonymous access sends no Authorization header.
    assert captured["auth"] is None


async def test_api_key_sent_as_token_header(clock: FixedClock) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_load("filings_page1.json"))

    async with _client(handler) as client:
        await _source(client, clock, max_pages=1, api_key="secret-key").fetch_signals(_BEFORE_ALL)

    assert captured["auth"] == "Token secret-key"


async def test_larger_spend_yields_larger_weight(clock: FixedClock) -> None:
    """MSFT reports $5.84M expenses; LMT reports $320k income -> MSFT outweighs LMT."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load("filings_page1.json"))

    async with _client(handler) as client:
        signals = await _source(client, clock, max_pages=1).fetch_signals(_BEFORE_ALL)

    by_ticker = {signal.ticker: signal for signal in signals}
    assert by_ticker["MSFT"].politician_weight > by_ticker["LMT"].politician_weight
    assert all(signal.politician_weight >= 1.0 for signal in signals)


async def test_skips_filings_at_or_before_cursor(clock: FixedClock) -> None:
    """A cursor after every filing's dt_posted yields nothing fresh."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load("filings_page1.json"))

    async with _client(handler) as client:
        # All filings are Dec 2024; a 2025 cursor is after all of them.
        signals = await _source(client, clock, max_pages=1).fetch_signals(
            datetime(2025, 1, 1, tzinfo=UTC)
        )

    assert signals == []


async def test_paginates_via_next_url_until_null(clock: FixedClock) -> None:
    """Page 1 advertises a `next`; following it yields page 2 (RTX) whose `next` is null."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        page = "filings_page2.json" if "page=2" in str(request.url) else "filings_page1.json"
        return httpx.Response(200, json=_load(page))

    async with _client(handler) as client:
        signals = await _source(client, clock, max_pages=5).fetch_signals(_BEFORE_ALL)

    # Walked exactly two pages, then stopped on next=null (did not exhaust max_pages).
    assert len(requested) == 2
    assert "page=2" in requested[1]
    assert {signal.ticker for signal in signals} == {"LMT", "MSFT", "BA", "RTX"}


async def test_respects_max_pages_cap(clock: FixedClock) -> None:
    """Both pages advertise a `next` (we serve page1 every time); max_pages caps the walk."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        # Always return page1, which carries a non-null `next`, so only the cap stops us.
        return httpx.Response(200, json=_load("filings_page1.json"))

    async with _client(handler) as client:
        await _source(client, clock, max_pages=3).fetch_signals(_BEFORE_ALL)

    assert len(requested) == 3


async def test_raises_on_http_error(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "rate limited"})

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _source(client, clock, max_pages=1).fetch_signals(_BEFORE_ALL)
