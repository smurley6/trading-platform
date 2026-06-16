"""openFDA Drugs@FDA approval adapter driven by a mock HTTP transport — real parsing of a
committed REAL-shaped fixture, request/auth shaping, supplement-date selection, the ``since``
cursor, pagination, and the openFDA empty-result 404. No network."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.conftest import FixedClock
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource
from trading.infrastructure.data_sources.fda_approvals import (
    RELIABILITY_REGULATORY,
    FdaApprovalsSource,
)
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURE = Path(__file__).parent / "fixtures" / "fda" / "drugsfda_recent_approvals.json"


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _load_fixture() -> dict[str, Any]:
    decoded: object = json.loads(_FIXTURE.read_text())
    assert isinstance(decoded, dict)
    return decoded


def _source(client: httpx.AsyncClient, clock: FixedClock, **kwargs: Any) -> FdaApprovalsSource:
    return FdaApprovalsSource(
        client=client, entity_resolver=GazetteerEntityResolver(), clock=clock, **kwargs
    )


async def test_emits_real_bullish_approval_signal(clock: FixedClock) -> None:
    fixture = _load_fixture()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture)

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(clock.now().replace(month=1, day=1))

    by_ticker = {signal.ticker: signal for signal in signals}
    # Pfizer, Eli Lilly, AbbVie resolve; "Aurobindo Pharma" has no ticker -> skipped.
    assert set(by_ticker) == {"PFE", "LLY", "ABBV"}
    pfizer = by_ticker["PFE"]
    assert pfizer.source_type is SignalSource.STATEMENT
    assert pfizer.politician_id is None
    assert pfizer.sentiment == pytest.approx(0.6)  # bullish catalyst
    assert pfizer.politician_weight >= 0
    assert pfizer.source_reliability == RELIABILITY_REGULATORY
    # Real federal regulatory data with no politician dependency -> REAL provenance, so it can
    # drive a real decision past the anti-placeholder guard.
    assert pfizer.provenance is Provenance.REAL
    assert pfizer.raw_payload["sponsor_name"] == "PFIZER INC"


async def test_created_at_is_the_approval_date_utc(clock: FixedClock) -> None:
    fixture = _load_fixture()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture)

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(clock.now().replace(month=1, day=1))

    by_ticker = {signal.ticker: signal for signal in signals}
    pfizer_at = by_ticker["PFE"].created_at
    assert (pfizer_at.year, pfizer_at.month, pfizer_at.day) == (2026, 6, 3)
    assert pfizer_at.tzinfo is not None and pfizer_at.utcoffset() is not None


async def test_uses_latest_approved_supplement_date(clock: FixedClock) -> None:
    """AbbVie's record has an approved supplement (20260515) plus the original 2018 approval;
    the signal is dated to the recent supplement (the fresh catalyst), not the original."""
    fixture = _load_fixture()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture)

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(clock.now().replace(month=1, day=1))

    abbvie = {s.ticker: s for s in signals}["ABBV"].created_at
    assert (abbvie.year, abbvie.month, abbvie.day) == (2026, 5, 15)


async def test_request_shapes_search_window_and_limit(clock: FixedClock) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["host"] = request.url.host
        captured["search"] = request.url.params.get("search")
        captured["limit"] = request.url.params.get("limit")
        captured["skip"] = request.url.params.get("skip")
        captured["api_key"] = request.url.params.get("api_key")
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        await _source(client, clock, lookback_days=30).fetch_signals(
            clock.now().replace(month=1, day=1)
        )

    assert captured["host"] == "api.fda.gov"
    assert captured["path"] == "/drug/drugsfda.json"
    search = captured["search"] or ""
    assert "submissions.submission_status:AP" in search
    # FixedClock default is 2026-06-12; a 30-day lookback ends today and starts 2026-05-13.
    assert "submissions.submission_status_date:[20260513 TO 20260612]" in search
    assert captured["limit"] == "100"
    assert captured["skip"] == "0"
    assert captured["api_key"] is None  # omitted when no key is configured


async def test_api_key_is_sent_when_configured(clock: FixedClock) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["api_key"] = request.url.params.get("api_key")
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        await _source(client, clock, api_key="secret-key").fetch_signals(
            clock.now().replace(month=1, day=1)
        )

    assert captured["api_key"] == "secret-key"


async def test_skips_approvals_at_or_before_cursor(clock: FixedClock) -> None:
    fixture = _load_fixture()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture)

    async with _client(handler) as client:
        # Cursor after every approval date in the fixture (latest is 2026-06-03) -> nothing fresh.
        signals = await _source(client, clock).fetch_signals(clock.now())

    assert signals == []


async def test_empty_404_is_treated_as_no_results(clock: FixedClock) -> None:
    """openFDA returns 404 when a search matches zero records; that is a normal empty result."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "No matches"}})

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(clock.now().replace(month=1, day=1))

    assert signals == []


async def test_server_error_raises(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _source(client, clock).fetch_signals(clock.now().replace(month=1, day=1))


async def test_paginates_with_skip_until_short_page(clock: FixedClock) -> None:
    fixture = _load_fixture()
    pfizer_record = fixture["results"][0]
    lilly_record = fixture["results"][1]
    skips_requested: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        skips_requested.append(request.url.params.get("skip"))
        # page_limit=1 forces paging: a full page (1 row) advertises more; the 2nd full page
        # returns a different approval; the 3rd is empty (short) and ends the walk.
        skip = request.url.params.get("skip")
        if skip == "0":
            return httpx.Response(200, json={"results": [pfizer_record]})
        if skip == "1":
            return httpx.Response(200, json={"results": [lilly_record]})
        return httpx.Response(200, json={"results": []})

    async with _client(handler) as client:
        signals = await _source(client, clock, page_limit=1, max_pages=5).fetch_signals(
            clock.now().replace(month=1, day=1)
        )

    assert skips_requested == ["0", "1", "2"]
    assert {signal.ticker for signal in signals} == {"PFE", "LLY"}


async def test_respects_max_pages_cap(clock: FixedClock) -> None:
    fixture = _load_fixture()
    pfizer_record = fixture["results"][0]
    skips_requested: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        skips_requested.append(request.url.params.get("skip"))
        # Every page is full (1 row at page_limit=1) so only the max_pages cap stops the walk.
        return httpx.Response(200, json={"results": [pfizer_record]})

    async with _client(handler) as client:
        await _source(client, clock, page_limit=1, max_pages=3).fetch_signals(
            clock.now().replace(month=1, day=1)
        )

    assert skips_requested == ["0", "1", "2"]
