"""SEC EDGAR Form 4 insider-trade adapter, driven by a mock HTTP transport — no network.

Fixtures under ``fixtures/sec/`` are real-shaped EDGAR documents: the recent-Form-4 Atom feed,
each filing's directory ``index.json``, and the ownership XML (``ownershipDocument``) carrying
nonDerivativeTransaction rows. The handler routes each requested URL to the matching fixture, so
the whole flow (feed -> index.json -> ownership XML -> Signal) runs offline.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.sec_form4 import (
    RELIABILITY_INSIDER_PURCHASE,
    RELIABILITY_INSIDER_SALE,
    SecForm4Source,
    _live_signal,
    _ParsedTransaction,
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
    # Side-aware reliability (Task 10, docs/RESEARCH-PLAN.md; Phase 6 label run
    # a38d3848-f1eb-4a55-8a6d-15e8f1464f7a, 2026-07-07): purchases keep a reduced-from-flat
    # reliability (was RELIABILITY_INSIDER_TRADE == 0.75 for both sides before Task 10).
    assert aapl.source_reliability == RELIABILITY_INSIDER_PURCHASE
    assert aapl.provenance is Provenance.REAL
    assert aapl.raw_payload["transaction_code"] == "P"

    msft = by_ticker["MSFT"]
    assert msft.sentiment == pytest.approx(-0.5)  # open-market sale -> bearish
    # Sales are anti-predictive (Phase 6) -> zero conviction contribution, though the signal is
    # still emitted (ingested/recorded/labeled) exactly as before.
    assert msft.source_reliability == RELIABILITY_INSIDER_SALE
    assert msft.provenance is Provenance.REAL
    assert msft.raw_payload["transaction_code"] == "S"


async def test_sale_signal_ingestion_unchanged_apart_from_reliability(clock: FixedClock) -> None:
    """Regression (Task 10 + task-19): a sale signal's ticker, sentiment, politician_weight,
    provenance, and raw_payload are unchanged by the reliability bench — only
    ``source_reliability`` differs (0.75 flat -> 0.0). The signal ``id`` DID change in task-19
    (review F25): it now follows the historical accession:index scheme
    (``sec-form4-{accession}:{transaction_index}``) instead of the day-granular
    issuer+date+code id, so a same-day multi-officer cluster no longer collapses in rm_signals."""
    async with _client(_make_handler([])) as client:
        source = _source(client, clock)
        signals = await source.fetch_signals(clock.now().replace(month=1, day=1))

    msft = next(s for s in signals if s.ticker == "MSFT")
    # accession 0000789019-26-000045 (from the archive URL), first (index 0) kept P/S row.
    assert msft.id == "sec-form4-0000789019-26-000045:0"
    assert msft.raw_payload["accession"] == "0000789019-26-000045"
    assert msft.raw_payload["source_uid"] == "0000789019-26-000045:0"
    assert msft.source_type is SignalSource.DISCLOSED_TRADE
    assert msft.politician_id is None
    assert msft.sentiment == pytest.approx(-0.5)
    assert msft.politician_weight >= 1.0
    assert msft.provenance is Provenance.REAL
    assert msft.raw_payload["transaction_code"] == "S"
    assert msft.raw_payload["ticker"] == "MSFT"
    # created_at is UNCHANGED by the id fix — still the transaction date on the live path.
    assert msft.created_at.date().isoformat() == "2026-06-11"
    # Only the reliability term changed; everything else about the emitted signal is unchanged.
    assert msft.source_reliability == 0.0


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


# --- review F25: live signal-id collisions in same-day insider clusters --------------------------


def _officer_txn(owner: str) -> _ParsedTransaction:
    """A same-issuer, same-day, same-code (P) transaction differing only by reporting owner —
    exactly the same-day multi-officer cluster the old day-granular live id collapsed."""
    return _ParsedTransaction(
        ticker="AAPL",
        issuer_cik="0000320193",
        owner=owner,
        is_officer=True,
        is_director=False,
        is_ten_percent_owner=False,
        officer_title="Officer",
        is_10b5_1=None,
        transaction_index=0,
        code="P",
        sentiment=0.5,
        shares=100.0,
        price=10.0,
        value=1_000.0,
        txn_date=datetime(2026, 6, 11, tzinfo=UTC),
    )


def test_same_day_three_officer_cluster_yields_three_distinct_ids() -> None:
    base = "https://www.sec.gov/Archives/edgar/data/320193/{acc}/form4.xml"
    urls = [
        base.format(acc=acc)
        for acc in ("000032019326000077", "000032019326000078", "000032019326000079")
    ]
    owners = ["COOK TIMOTHY D", "MAESTRI LUCA", "O'KELLY KATE"]
    signals = [
        _live_signal(_officer_txn(o), source_url=u) for o, u in zip(owners, urls, strict=True)
    ]

    ids = {s.id for s in signals}
    assert ids == {
        "sec-form4-0000320193-26-000077:0",
        "sec-form4-0000320193-26-000078:0",
        "sec-form4-0000320193-26-000079:0",
    }
    # All three share issuer + transaction date + code — the OLD id (issuer-date-code) collapsed
    # them into ONE rm_signals row; the per-filing accession now keeps all three.
    old_id_inputs = {
        (
            s.raw_payload["issuer_cik"],
            s.raw_payload["transaction_date"],
            s.raw_payload["transaction_code"],
        )
        for s in signals
    }
    assert old_id_inputs == {("0000320193", "2026-06-11", "P")}


def test_live_id_fallback_uses_owner_when_url_lacks_accession() -> None:
    """When the archive URL carries no accession directory, the fallback id still separates a
    same-day multi-officer cluster by owner (review F25)."""
    cook = _live_signal(_officer_txn("COOK TIMOTHY D"), source_url="http://example/x.xml")
    maestri = _live_signal(_officer_txn("MAESTRI LUCA"), source_url="http://example/x.xml")
    assert cook.id != maestri.id
    assert cook.raw_payload["accession"] is None
    assert cook.raw_payload["source_uid"] is None


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
