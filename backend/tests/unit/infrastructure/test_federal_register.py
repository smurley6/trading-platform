"""Federal Register documents.json adapter driven by a mock HTTP transport — real parsing of a
committed REAL-shaped fixture, type/agency-conditioned sentiment, the ``since`` cursor, cursor
pagination, the empty/404 paths, and the default-OFF no-network guarantee. No network.

The committed fixture is a real-shaped ``documents.json`` page with five records: an FDA Rule
naming Pfizer and an EPA Rule naming Moderna (both resolve to tickers), a DoD Proposed Rule naming
Lockheed Martin, a Presidential Document, and an administrative rule that names no tradeable entity
(exercising the drop path). Ticker resolution uses the real ``GazetteerEntityResolver`` (as the FDA
test does).
"""

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
from trading.infrastructure.data_sources.federal_register import (
    RELIABILITY_FDA_REGULATORY,
    RELIABILITY_REGULATORY,
    FederalRegisterSource,
)
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURE = Path(__file__).parent / "fixtures" / "federal_register" / "documents_recent.json"

# A cursor far before any fixture publication date so every fresh document passes the window.
_OPEN_CURSOR = datetime(2026, 1, 1, tzinfo=UTC)


class _StubSentiment:
    """Deterministic stance toward any ticker (used only for the Presidential-Document path)."""

    def __init__(self, score: float = 0.4) -> None:
        self._score = score

    def score(self, text: str, ticker: str) -> float:
        return self._score


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _load_fixture() -> dict[str, Any]:
    decoded: object = json.loads(_FIXTURE.read_text())
    assert isinstance(decoded, dict)
    return decoded


def _fixture_handler(record: list[httpx.Request] | None = None) -> Handler:
    fixture = _load_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(200, json=fixture)

    return handler


def _source(
    client: httpx.AsyncClient,
    clock: FixedClock,
    **kwargs: Any,
) -> FederalRegisterSource:
    kwargs.setdefault("enabled", True)
    return FederalRegisterSource(
        client=client,
        entity_resolver=GazetteerEntityResolver(),
        clock=clock,
        **kwargs,
    )


async def test_name_property(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        assert _source(client, clock).name == "federal_register"


async def test_emits_real_regulatory_signals_with_correct_mapping(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    by_ticker = {signal.ticker: signal for signal in signals}
    # Pfizer (FDA Rule), Moderna (EPA Rule), Lockheed (DoD Proposed Rule) resolve; the immigration
    # rule names no tradeable entity -> dropped.
    assert set(by_ticker) == {"PFE", "MRNA", "LMT"}

    pfizer = by_ticker["PFE"]
    assert pfizer.source_type is SignalSource.STATEMENT
    assert pfizer.politician_id is None
    # An FDA Rule reads as an approval catalyst (+0.6) at the FDA regulatory reliability prior.
    assert pfizer.sentiment == pytest.approx(0.6)
    assert pfizer.source_reliability == RELIABILITY_FDA_REGULATORY
    assert pfizer.politician_weight >= 0
    # Real federal regulatory data with no politician dependency -> REAL, so it can drive a real
    # decision past the anti-placeholder guard.
    assert pfizer.provenance is Provenance.REAL
    # id preserves the document_number for traceability; raw payload is retained for audit.
    assert pfizer.id == "fr-document-2026-13300-PFE"
    assert pfizer.raw_payload["document_number"] == "2026-13300"
    assert pfizer.raw_payload["type"] == "Rule"
    assert pfizer.raw_payload["agencies"][0]["slug"] == "food-and-drug-administration"

    # A non-FDA Rule reads neutral-to-positive (+0.5) at the lower regulatory prior.
    moderna = by_ticker["MRNA"]
    assert moderna.sentiment == pytest.approx(0.5)
    assert moderna.source_reliability == RELIABILITY_REGULATORY

    # A Proposed Rule reads as uncertainty (+0.1).
    lockheed = by_ticker["LMT"]
    assert lockheed.sentiment == pytest.approx(0.1)
    assert lockheed.raw_payload["type"] == "Proposed Rule"


async def test_created_at_is_publication_date_at_utc_midnight(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    pfizer_at = {s.ticker: s for s in signals}["PFE"].created_at
    assert (pfizer_at.year, pfizer_at.month, pfizer_at.day) == (2026, 6, 29)
    assert (pfizer_at.hour, pfizer_at.minute) == (0, 0)
    assert pfizer_at.tzinfo is not None and pfizer_at.utcoffset() == UTC.utcoffset(None)


async def test_since_cursor_excludes_documents_at_or_before_it(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        # The newest fixture document is published 2026-06-29; a cursor at that instant excludes
        # every document (strict >), so nothing is emitted.
        signals = await _source(client, clock).fetch_signals(datetime(2026, 6, 29, tzinfo=UTC))

    assert signals == []


async def test_since_cursor_keeps_only_newer_documents(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        # Cursor between the EPA Moderna rule (2026-06-28) and the FDA Pfizer rule (2026-06-29):
        # only the FDA Pfizer document survives.
        signals = await _source(client, clock).fetch_signals(datetime(2026, 6, 28, tzinfo=UTC))

    assert {s.ticker for s in signals} == {"PFE"}


async def test_presidential_document_uses_injected_sentiment(clock: FixedClock) -> None:
    # Rewrite the Presidential Document so it names a resolvable entity, proving the analyzer path.
    fixture = _load_fixture()
    for record in fixture["results"]:
        if record["type"] == "Presidential Document":
            record["title"] = "Executive Order on Apple Inc Supply Chains"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture)

    async with _client(handler) as client:
        source = _source(client, clock, sentiment=_StubSentiment(0.42))
        signals = await source.fetch_signals(_OPEN_CURSOR)

    aapl = {s.ticker: s for s in signals}["AAPL"]
    # The Presidential-Document leg is scored by the injected analyzer, not a fixed prior.
    assert aapl.sentiment == pytest.approx(0.42)


async def test_empty_results_yield_no_signal(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"count": 0, "results": [], "next_page_url": None})

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    assert signals == []


async def test_not_found_is_treated_as_empty(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    assert signals == []


async def test_http_error_propagates(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _source(client, clock).fetch_signals(_OPEN_CURSOR)


async def test_request_shapes_window_order_and_optional_key(clock: FixedClock) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["path"] = request.url.path
        captured["gte"] = request.url.params.get("conditions[publication_date][gte]")
        captured["order"] = request.url.params.get("order")
        captured["per_page"] = request.url.params.get("per_page")
        captured["api_key"] = request.url.params.get("api_key")
        return httpx.Response(200, json={"results": [], "next_page_url": None})

    async with _client(handler) as client:
        await _source(client, clock, lookback_days=30, api_key="DEMO_KEY").fetch_signals(
            _OPEN_CURSOR
        )

    assert captured["host"] == "www.federalregister.gov"
    assert captured["path"] == "/api/v1/documents.json"
    # FixedClock default is 2026-06-12; a 30-day lookback starts 2026-05-13.
    assert captured["gte"] == "2026-05-13"
    assert captured["order"] == "newest"
    assert captured["per_page"] == "100"
    assert captured["api_key"] == "DEMO_KEY"


async def test_follows_cursor_pagination(clock: FixedClock) -> None:
    """Two pages linked by next_page_url are both walked and their signals combined."""
    page_two = _load_fixture()
    # Keep only the FDA Pfizer record on page one, pointing at page two; page two has the rest.
    page_one_results = [page_two["results"][0]]
    remaining = page_two["results"][1:]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "search_after_cursor" not in str(request.url):
            return httpx.Response(
                200,
                json={
                    "results": page_one_results,
                    "next_page_url": (
                        "https://www.federalregister.gov/api/v1/documents.json"
                        "?per_page=100&search_after_cursor=CURSORTOKEN"
                    ),
                },
            )
        return httpx.Response(200, json={"results": remaining, "next_page_url": None})

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    assert len(calls) == 2
    # Both pages contributed: Pfizer from page one, Moderna + Lockheed from page two.
    assert {s.ticker for s in signals} == {"PFE", "MRNA", "LMT"}


async def test_disabled_source_makes_no_network_call_and_yields_nothing(clock: FixedClock) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_load_fixture())

    async with _client(handler) as client:
        # Default OFF: constructed without enabled=True.
        source = FederalRegisterSource(
            client=client,
            entity_resolver=GazetteerEntityResolver(),
            clock=clock,
        )
        signals = await source.fetch_signals(_OPEN_CURSOR)

    assert signals == []
    # No socket touched on the default/disabled path.
    assert calls == []


async def test_max_pages_caps_the_walk(clock: FixedClock) -> None:
    """A page that always advertises a next cursor is walked at most max_pages times."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "results": [],
                "next_page_url": (
                    "https://www.federalregister.gov/api/v1/documents.json"
                    "?per_page=100&search_after_cursor=NEVERENDING"
                ),
            },
        )

    async with _client(handler) as client:
        await _source(client, clock, max_pages=3).fetch_signals(_OPEN_CURSOR)

    assert len(calls) == 3
