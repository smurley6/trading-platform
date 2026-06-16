"""SEC EDGAR Form 4 insider-trade adapter, driven by a mock HTTP transport — no network.

Fixtures under ``fixtures/sec/`` are real-shaped EDGAR documents: the recent-Form-4 Atom feed,
each filing's directory ``index.json``, and the ownership XML (``ownershipDocument``) carrying
nonDerivativeTransaction rows. The handler routes each requested URL to the matching fixture, so
the whole flow (feed -> index.json -> ownership XML -> Signal) runs offline.
"""

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.sec_form4 import (
    RELIABILITY_INSIDER_TRADE,
    SecForm4Source,
    _value_weight,
)

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "sec"
_USER_AGENT = "trading-platform research steven.murley@gmail.com"

# Maps the path of each EDGAR URL the adapter requests to the fixture that answers it.
_FIXTURE_BY_PATH: dict[str, str] = {
    "/cgi-bin/browse-edgar": "recent_form4.atom",
    "/Archives/edgar/data/320193/000032019326000077/index.json": "apple_index.json",
    "/Archives/edgar/data/789019/000078901926000045/index.json": "microsoft_index.json",
    "/Archives/edgar/data/320193/000032019326000077/form4_apple_purchase.xml": (
        "form4_apple_purchase.xml"
    ),
    "/Archives/edgar/data/789019/000078901926000045/form4_microsoft_sale.xml": (
        "form4_microsoft_sale.xml"
    ),
}


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text()


def _make_handler(seen_user_agents: list[str | None]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen_user_agents.append(request.headers.get("user-agent"))
        fixture = _FIXTURE_BY_PATH.get(request.url.path)
        if fixture is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=_read(fixture))

    return handler


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _source(
    client: httpx.AsyncClient, clock: FixedClock, *, max_filings: int = 100
) -> SecForm4Source:
    return SecForm4Source(
        client=client,
        clock=clock,
        user_agent=_USER_AGENT,
        max_filings=max_filings,
    )


async def test_emits_purchase_and_sale_signals(clock: FixedClock) -> None:
    seen: list[str | None] = []
    async with _client(_make_handler(seen)) as client:
        source = _source(client, clock)
        # Cursor well before the 2026-06-11 transactions -> everything qualifying is fresh.
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    by_ticker = {s.ticker: s for s in signals}
    # Apple purchase (P) + Microsoft sale (S); the Microsoft AWARD (A) row is skipped.
    assert set(by_ticker) == {"AAPL", "MSFT"}

    aapl = by_ticker["AAPL"]
    assert aapl.source_type is SignalSource.DISCLOSED_TRADE
    assert aapl.politician_id is None
    assert aapl.sentiment == pytest.approx(0.5)  # open-market purchase -> bullish
    assert aapl.source_reliability == RELIABILITY_INSIDER_TRADE
    assert aapl.provenance is Provenance.REAL
    assert aapl.raw_payload["transaction_code"] == "P"

    msft = by_ticker["MSFT"]
    assert msft.sentiment == pytest.approx(-0.5)  # open-market sale -> bearish
    assert msft.provenance is Provenance.REAL
    assert msft.raw_payload["transaction_code"] == "S"


async def test_skips_award_grant_rows(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client, clock)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    # The Microsoft filing has an S row and an A (award) row; only S should survive.
    msft_codes = [s.raw_payload["transaction_code"] for s in signals if s.ticker == "MSFT"]
    assert msft_codes == ["S"]
    assert all(s.raw_payload["transaction_code"] != "A" for s in signals)


async def test_sets_user_agent_on_every_request(clock: FixedClock) -> None:
    seen: list[str | None] = []
    async with _client(_make_handler(seen)) as client:
        source = _source(client, clock)
        await source.fetch_signals(clock.now().replace(month=1, day=1))

    # Feed + 2 index.json + 2 ownership XML = 5 requests, all carrying the descriptive UA.
    assert len(seen) == 5
    assert all(ua == _USER_AGENT for ua in seen)


async def test_window_excludes_transactions_at_or_before_cursor(clock: FixedClock) -> None:
    async with _client(_make_handler([])) as client:
        source = _source(client, clock)
        # Transactions are dated 2026-06-11; a cursor on that day excludes them (strict >).
        signals = await source.fetch_signals(clock.now().replace(month=6, day=11))

    assert signals == []


async def test_max_filings_caps_the_walk(clock: FixedClock) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        fixture = _FIXTURE_BY_PATH.get(request.url.path)
        if fixture is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=_read(fixture))

    async with _client(handler) as client:
        source = _source(client, clock, max_filings=1)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    # Only the first feed entry (Apple) is followed; Microsoft's index/XML are never requested.
    assert {s.ticker for s in signals} == {"AAPL"}
    assert not any("789019" in path for path in requested_paths)


async def test_value_weight_scales_with_transaction_size() -> None:
    # Larger dollar value -> strictly larger weight, floored at 1.0 and capped at 2.0.
    assert _value_weight(0.0) == 1.0
    assert _value_weight(-5.0) == 1.0
    small = _value_weight(10_000.0)
    big = _value_weight(50_000_000.0)
    assert small < big
    assert 1.0 <= small <= big <= 2.0


async def test_malformed_ownership_xml_yields_no_signal(clock: FixedClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".xml") and "form4" in request.url.path:
            return httpx.Response(200, text="<ownershipDocument><broken>")
        fixture = _FIXTURE_BY_PATH.get(request.url.path)
        if fixture is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=_read(fixture))

    async with _client(handler) as client:
        source = _source(client, clock)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    assert signals == []
