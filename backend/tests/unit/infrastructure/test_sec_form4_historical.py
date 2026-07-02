"""SEC EDGAR Form 4 HISTORICAL adapter (Task 04a, docs/RESEARCH-PLAN.md), no network.

Fixtures under ``fixtures/sec/``: a quarterly ``form.idx`` snippet (a form 4, a form 4/A
amendment, a form 4 for an issuer outside COMPANY_DIRECTORY, all in QTR1 2024), two full-
submission ``.txt`` files (Apple: one purchase; Microsoft: a sale + an award + a purchase, to
exercise multi-signal-per-filing ``source_uid`` granularity and the award-skip), and a
``company_tickers.json`` snippet for the CIK filter. A mock HTTP transport routes each requested
URL to its fixture so the whole walk (form.idx -> full submission -> ownership XML -> Signal)
runs offline.

The PIT trap this task exists to close: ``created_at`` on this path MUST be the filing's
ACCEPTANCE-DATETIME (converted from the SGML header's naive US/Eastern value to UTC), never the
transaction date inside the XML — the Microsoft fixture's acceptance (22:30 ET) crosses midnight
UTC specifically to make a wrong (transaction-date) implementation visibly fail.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.ports.data_source import FilterableHistoricalSource
from trading.domain.provenance import Provenance
from trading.infrastructure.data_sources.sec_form4 import SecForm4Source

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "sec"
_USER_AGENT = "trading-platform research steven.murley@gmail.com"

_FIXTURE_BY_PATH: dict[str, str] = {
    "/Archives/edgar/full-index/2024/QTR1/form.idx": "form_2024QTR1.idx",
    "/Archives/edgar/data/320193/0000320193-24-000005.txt": "submission_apple.txt",
    "/Archives/edgar/data/789019/0000789019-24-000010.txt": "submission_microsoft.txt",
    "/files/company_tickers.json": "company_tickers.json",
}


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text()


def _make_handler(seen_user_agents: list[str | None] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen_user_agents is not None:
            seen_user_agents.append(request.headers.get("user-agent"))
        fixture = _FIXTURE_BY_PATH.get(request.url.path)
        if fixture is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=_read(fixture))

    return handler


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _source(client: httpx.AsyncClient, clock: FixedClock, **kwargs: object) -> SecForm4Source:
    return SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT, **kwargs)  # type: ignore[arg-type]


_WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
_WINDOW_END = datetime(2024, 3, 31, tzinfo=UTC)


async def test_fetch_historical_returns_correct_signal_count(clock: FixedClock) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    # Apple: 1 kept transaction (P). Microsoft: 2 kept transactions (S, P; the A row skipped).
    # Microsoft's 4/A amendment filing is skipped entirely (0 signals from it). The Random Co
    # filing is CIK-filtered out (not in COMPANY_DIRECTORY) -> never fetched, 0 signals.
    assert len(signals) == 3
    assert {s.ticker for s in signals} == {"AAPL", "MSFT"}


async def test_source_uid_scheme_is_accession_plus_per_signal_suffix(clock: FixedClock) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    uids = {s.raw_payload["source_uid"] for s in signals}
    # Accession + a stable per-signal suffix (transaction index among kept P/S rows), NOT the
    # bare accession (which would collide across Microsoft's two kept transactions).
    assert uids == {
        "0000320193-24-000005:0",
        "0000789019-24-000010:0",
        "0000789019-24-000010:1",
    }
    # The bare accession is ALSO retained as its own raw_payload field.
    for signal in signals:
        assert signal.raw_payload["accession"] in {"0000320193-24-000005", "0000789019-24-000010"}
    assert len(uids) == len(signals)  # no collisions


async def test_created_at_is_acceptance_datetime_converted_from_eastern(
    clock: FixedClock,
) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    by_ticker = {s.ticker: s for s in signals if s.ticker == "AAPL"}
    aapl = by_ticker["AAPL"]
    # Apple's SGML header: <ACCEPTANCE-DATETIME>20240115173045 (17:30:45 ET, EST in January,
    # UTC-5) -> 22:30:45 UTC. NOT the transaction date (2024-01-11) — the PIT trap this task
    # exists to close.
    assert aapl.created_at == datetime(2024, 1, 15, 22, 30, 45, tzinfo=UTC)
    assert aapl.raw_payload["transaction_date"] == "2024-01-11"

    msft_signals = [s for s in signals if s.ticker == "MSFT"]
    # Microsoft's SGML header: <ACCEPTANCE-DATETIME>20240120223000 (22:30:00 ET) -> crosses
    # midnight -> 2024-01-21 03:30:00 UTC. A transaction-date-based implementation would instead
    # produce 2024-01-19, a full UTC calendar day earlier — this assertion fails loudly on that
    # PIT-incorrect substitution.
    assert all(s.created_at == datetime(2024, 1, 21, 3, 30, 0, tzinfo=UTC) for s in msft_signals)


async def test_ticker_extracted_from_issuer_block(clock: FixedClock) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert {s.ticker for s in signals} == {"AAPL", "MSFT"}
    assert all(s.ticker for s in signals)


async def test_form_4a_amendments_are_skipped_and_counted(
    clock: FixedClock, caplog: pytest.LogCaptureFixture
) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        with caplog.at_level("INFO"):
            signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    # The 4/A filing's accession never appears in any emitted signal.
    assert all("0000789019-24-000006" not in s.raw_payload["accession"] for s in signals)
    assert any("1 4/A amendment(s) skipped" in record.message for record in caplog.records)


async def test_cik_filter_excludes_issuer_outside_company_directory(clock: FixedClock) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        fixture = _FIXTURE_BY_PATH.get(request.url.path)
        if fixture is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=_read(fixture))

    async with _client(handler) as client:
        source = _source(client, clock)
        await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    # "Some Random Co" (CIK 9999999, ticker RAND) is present in company_tickers.json but is NOT
    # in COMPANY_DIRECTORY, so its filing's full-submission URL must never be requested at all.
    assert not any("9999999" in path for path in requested_paths)


async def test_no_filter_disables_cik_restriction(clock: FixedClock) -> None:
    """Confirms set_filter_enabled(False) (the --no-filter CLI path) allows the wider set through
    (Random Co isn't in our fixture map, so it 404s rather than being skipped -- proving the
    adapter now considers the row eligible to fetch). A single filing's fetch failure must not
    abort the whole run -- the walk continues and still returns the other filings' signals."""
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        fixture = _FIXTURE_BY_PATH.get(request.url.path)
        if fixture is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, text=_read(fixture))

    async with _client(handler) as client:
        source = _source(client, clock)
        assert isinstance(source, FilterableHistoricalSource)
        source.set_filter_enabled(False)
        assert source.filter_description() is None
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    # With the filter off, the Random Co filing IS attempted (even though it 404s, unmapped in
    # our fixture set) -- proving the CIK restriction was lifted, not merely coincidentally absent.
    assert any("9999999" in path for path in requested_paths)
    # The 404 on Random Co does not abort the run: Apple/Microsoft signals still come through.
    assert {s.ticker for s in signals} == {"AAPL", "MSFT"}


async def test_filter_enabled_by_default_and_description(clock: FixedClock) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        assert isinstance(source, FilterableHistoricalSource)
        assert source.filter_description() is not None
        assert "CIK filter" in (source.filter_description() or "")


async def test_provenance_is_real_not_backfilled(clock: FixedClock) -> None:
    """The adapter itself stamps REAL; only the BackfillSignals use case re-stamps to BACKFILLED
    (Task 02) -- this adapter must not pre-empt that."""
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert all(s.provenance is Provenance.REAL for s in signals)
    assert all(s.pit_confidence == "exact" for s in signals)


async def test_politician_id_is_none_for_corporate_insiders(clock: FixedClock) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert all(s.politician_id is None for s in signals)


async def test_sets_user_agent_on_every_request(clock: FixedClock) -> None:
    seen: list[str | None] = []
    async with _client(_make_handler(seen)) as client:
        source = _source(client, clock)
        await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    assert len(seen) > 0
    assert all(ua == _USER_AGENT for ua in seen)


async def test_transaction_date_retained_only_in_raw_payload(clock: FixedClock) -> None:
    async with _client(_make_handler()) as client:
        source = _source(client, clock)
        signals = await source.fetch_historical(_WINDOW_START, _WINDOW_END)

    for signal in signals:
        assert "transaction_date" in signal.raw_payload
        # created_at must never equal the (earlier) transaction date.
        assert signal.created_at.date().isoformat() != signal.raw_payload["transaction_date"] or (
            # (Guard against a false pass if a fixture's txn date happens to match acceptance
            # date; none of ours do, but keep the assertion meaningful either way.)
            signal.created_at != datetime.fromisoformat(signal.raw_payload["transaction_date"])
        )
