"""SEC EDGAR "latest filings" Form 4 Atom adapter, driven by a mock HTTP transport — no network.

The committed fixture is a real-shaped EDGAR ``getcurrent`` Atom feed with three Form 4 entries:
Apple and Microsoft (which resolve to tickers via the gazetteer) plus a foreign issuer that does
NOT resolve (exercising the drop path). A handler routes the single feed URL to the fixture, so
the whole flow (feed -> resolve -> Signal) runs offline. Ticker resolution uses the real
``GazetteerEntityResolver`` (as the FDA test does); sentiment is a tiny stub because a Form 4 feed
title carries no polarity words for the real lexicon to score.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.sec_edgar_atom import (
    RELIABILITY_INSIDER_TRADE,
    SecEdgarAtomSource,
)
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURE = Path(__file__).parent / "fixtures" / "edgar_form4_getcurrent.atom"
_USER_AGENT = "trading-platform research steven.murley@gmail.com"


class _StubSentiment:
    """Deterministic positive stance toward any ticker (a Form 4 filing is a bullish watch).

    A real Form 4 feed title (``4 - APPLE INC (...) (Issuer)``) has no polarity words, so the
    production lexicon would score 0.0 and skip everything; the adapter contract only needs a
    SentimentAnalyzerPort, so we inject a fixed-score stub here.
    """

    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    def score(self, text: str, ticker: str) -> float:
        return self._score


def _read() -> str:
    return _FIXTURE.read_text()


def _make_handler(seen_user_agents: list[str | None]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen_user_agents.append(request.headers.get("user-agent"))
        if request.url.path == "/cgi-bin/browse-edgar":
            return httpx.Response(200, text=_read())
        return httpx.Response(404, text="not found")

    return handler


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _source(
    client: httpx.AsyncClient,
    *,
    sentiment: _StubSentiment | None = None,
    max_filings: int = 100,
) -> SecEdgarAtomSource:
    return SecEdgarAtomSource(
        client=client,
        entity_resolver=GazetteerEntityResolver(),
        sentiment=sentiment or _StubSentiment(),
        user_agent=_USER_AGENT,
        max_filings=max_filings,
    )


async def test_emits_one_real_signal_per_resolved_filing(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client)
        # Cursor well before the 2026-06-12 filings -> everything qualifying is fresh.
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    by_ticker = {s.ticker: s for s in signals}
    # Apple + Microsoft resolve; the foreign pharma issuer has no ticker -> dropped.
    assert set(by_ticker) == {"AAPL", "MSFT"}

    aapl = by_ticker["AAPL"]
    assert aapl.source_type is SignalSource.DISCLOSED_TRADE
    assert aapl.politician_id is None
    assert aapl.sentiment == pytest.approx(0.5)
    assert aapl.politician_weight >= 0
    assert aapl.source_reliability == RELIABILITY_INSIDER_TRADE
    # Real SEC disclosure, no politician dependency -> REAL, so it can pass the anti-placeholder
    # guard and drive a real decision.
    assert aapl.provenance is Provenance.REAL
    assert aapl.raw_payload["issuer_cik"] == "0000320193"
    assert aapl.raw_payload["accession"] == "0000320193-26-000077"

    assert by_ticker["MSFT"].provenance is Provenance.REAL
    assert by_ticker["MSFT"].raw_payload["issuer_cik"] == "0000789019"


async def test_unresolved_issuer_is_dropped(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    # The Zhangzhou Pienzhuang filing resolves to no tracked ticker, so it yields no Signal.
    assert all("ZHANGZHOU" not in s.raw_payload["company"].upper() for s in signals)
    assert {s.ticker for s in signals} == {"AAPL", "MSFT"}


async def test_created_at_is_the_acceptance_timestamp_in_utc(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    aapl = next(s for s in signals if s.ticker == "AAPL")
    # Feed stamps 2026-06-12T16:30:11-04:00 -> 20:30:11 UTC.
    assert (aapl.created_at.year, aapl.created_at.month, aapl.created_at.day) == (2026, 6, 12)
    assert aapl.created_at.hour == 20
    assert aapl.created_at.tzinfo is not None and aapl.created_at.utcoffset() is not None


async def test_since_cursor_excludes_entries_at_or_before_it(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client)
        # Newest entry is 2026-06-12T16:30:11-04:00 == 20:30:11 UTC; a cursor at that instant
        # excludes every entry (strict >), so nothing is emitted.
        cursor = datetime(2026, 6, 12, 20, 30, 11, tzinfo=UTC)
        signals = await source.fetch_signals(cursor)

    assert signals == []


async def test_since_cursor_keeps_only_newer_entries(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client)
        # Cursor between Microsoft (20:12:44 UTC) and Apple (20:30:11 UTC): only Apple survives.
        cursor = datetime(2026, 6, 12, 20, 20, 0, tzinfo=UTC)
        signals = await source.fetch_signals(cursor)

    assert {s.ticker for s in signals} == {"AAPL"}


async def test_zero_sentiment_entry_is_skipped(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client, sentiment=_StubSentiment(0.0))
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    # No directional stance -> no Signal, even for resolvable issuers.
    assert signals == []


async def test_sets_user_agent_on_the_request(clock: FixedClock) -> None:
    seen: list[str | None] = []
    async with _client(_make_handler(seen)) as client:
        source = _source(client)
        await source.fetch_signals(clock.now().replace(month=1, day=1))

    # A single feed request, carrying the descriptive UA SEC requires.
    assert seen == [_USER_AGENT]


async def test_malformed_feed_yields_no_signal(clock: FixedClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<feed><broken>")

    async with _client(handler) as client:
        source = _source(client)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    assert signals == []
