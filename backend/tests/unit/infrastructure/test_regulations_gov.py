"""Regulations.gov v4 documents adapter driven by a mock HTTP transport — real parsing of a
committed REAL-shaped fixture, request/auth shaping, the ``since`` cursor, pagination, the
default-OFF (no key -> no network) path, and HTTP-error handling. No network.

The committed fixture mirrors a real ``/v4/documents`` page: a Boeing airworthiness *Proposed
Rule* open for comment and a Chevron refinery *Proposed Rule* (both resolve to tickers and read
bearish), a Tesla *withdrawn* rule (resolves, reads bullish), and two dockets whose titles resolve
to no tracked issuer (a knotweed biological-control notice, an FMCSA information-collection
notice) which are dropped. Ticker resolution uses the real ``GazetteerEntityResolver`` (as the FDA
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
from trading.infrastructure.data_sources.regulations_gov import (
    RELIABILITY_REGULATORY,
    RegulationsGovSource,
)
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURE = Path(__file__).parent / "fixtures" / "regulations_gov" / "documents_recent.json"
_DEMO_KEY = "DEMO_KEY"
# A cursor far before any fixture posted date so every document passes the window.
_OPEN_CURSOR = datetime(2026, 1, 1, tzinfo=UTC)


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
    *,
    api_key: str | None = _DEMO_KEY,
    **kwargs: Any,
) -> RegulationsGovSource:
    return RegulationsGovSource(
        client=client,
        entity_resolver=GazetteerEntityResolver(),
        clock=clock,
        api_key=api_key,
        **kwargs,
    )


async def test_emits_real_regulatory_signals_with_correct_mapping(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    by_ticker = {signal.ticker: signal for signal in signals}
    # Boeing + Chevron proposed rules resolve (bearish); Tesla withdrawal resolves (bullish); the
    # knotweed notice and FMCSA information-collection notice resolve to no ticker -> dropped.
    assert set(by_ticker) == {"BA", "CVX", "TSLA"}
    assert all(signal.source_type is SignalSource.STATEMENT for signal in signals)

    boeing = by_ticker["BA"]
    assert boeing.politician_id is None
    assert boeing.sentiment < 0  # a proposed rule open for comment is regulatory risk
    assert boeing.politician_weight >= 1.0
    assert boeing.source_reliability == RELIABILITY_REGULATORY
    # Real federal regulatory data with no politician dependency -> REAL provenance, so it can
    # drive a real decision past the anti-placeholder guard.
    assert boeing.provenance is Provenance.REAL
    assert boeing.id == "regulations-FAA-2026-7195-0001-BA"
    # The full attributes dict is retained on the Signal for the audit log.
    assert boeing.raw_payload["agencyId"] == "FAA"
    assert boeing.raw_payload["docketId"] == "FAA-2026-7195"

    # A withdrawal / deregulation reads bullish.
    assert by_ticker["TSLA"].sentiment > 0
    assert by_ticker["TSLA"].raw_payload["withdrawn"] is True


async def test_created_at_is_the_posted_date_utc(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    boeing_at = {s.ticker: s for s in signals}["BA"].created_at
    # Fixture stamps postedDate 2026-06-22T04:00:00Z.
    assert (boeing_at.year, boeing_at.month, boeing_at.day) == (2026, 6, 22)
    assert boeing_at.tzinfo is not None and boeing_at.utcoffset() is not None


async def test_short_comment_deadline_weighs_more(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    by_ticker = {signal.ticker: signal for signal in signals}
    # Boeing: ~10-day comment window (2026-06-22 -> 2026-07-02) is imminent; Chevron: ~90-day
    # window (2026-06-21 -> 2026-09-22). The nearer deadline earns the larger urgency weight.
    assert by_ticker["BA"].politician_weight > by_ticker["CVX"].politician_weight


async def test_request_shapes_window_key_and_pagination(clock: FixedClock) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["path"] = request.url.path
        captured["ge"] = request.url.params.get("filter[postedDate][ge]")
        captured["le"] = request.url.params.get("filter[postedDate][le]")
        captured["page_size"] = request.url.params.get("page[size]")
        captured["page_number"] = request.url.params.get("page[number]")
        captured["x_api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"data": [], "meta": {}})

    async with _client(handler) as client:
        await _source(client, clock, lookback_days=7).fetch_signals(_OPEN_CURSOR)

    assert captured["host"] == "api.regulations.gov"
    assert captured["path"] == "/v4/documents"
    # FixedClock default is 2026-06-12; a 7-day lookback ends today and starts 2026-06-05.
    assert captured["ge"] == "2026-06-05"
    assert captured["le"] == "2026-06-12"
    assert captured["page_size"] == "250"
    assert captured["page_number"] == "1"
    # The key is sent as the X-Api-Key header (never hardcoded in the adapter).
    assert captured["x_api_key"] == _DEMO_KEY


async def test_skips_documents_at_or_before_cursor(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        # Cursor at the latest posted date in the fixture (2026-06-22T04:00:00Z) -> strict >
        # excludes every document, so nothing is emitted.
        cursor = datetime(2026, 6, 22, 4, 0, 0, tzinfo=UTC)
        signals = await _source(client, clock).fetch_signals(cursor)

    assert signals == []


async def test_since_cursor_keeps_only_newer_documents(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        # Cursor between the Chevron (2026-06-21) and Boeing (2026-06-22) proposed rules: only the
        # Boeing rule survives (Tesla is 2026-06-20, before the cursor).
        cursor = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
        signals = await _source(client, clock).fetch_signals(cursor)

    assert {s.ticker for s in signals} == {"BA"}


async def test_empty_page_yields_no_signals(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "meta": {"totalElements": 0}})

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    assert signals == []


async def test_server_error_raises(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _source(client, clock).fetch_signals(_OPEN_CURSOR)


async def test_disabled_without_api_key_makes_no_network_call(clock: FixedClock) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_load_fixture())

    async with _client(handler) as client:
        # Default OFF: no key configured -> yields no signals AND makes no network call.
        source = _source(client, clock, api_key=None)
        signals = await source.fetch_signals(_OPEN_CURSOR)

    assert signals == []
    assert calls == []


async def test_paginates_until_short_page(clock: FixedClock) -> None:
    fixture = _load_fixture()
    boeing_doc = fixture["data"][0]
    chevron_doc = fixture["data"][1]
    pages_requested: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page[number]")
        pages_requested.append(page)
        # page_size=1 forces paging: a full page (1 row) advertises more; the 2nd full page returns
        # a different document; the 3rd is empty (short) and ends the walk.
        if page == "1":
            return httpx.Response(200, json={"data": [boeing_doc], "meta": {}})
        if page == "2":
            return httpx.Response(200, json={"data": [chevron_doc], "meta": {}})
        return httpx.Response(200, json={"data": [], "meta": {}})

    async with _client(handler) as client:
        signals = await _source(client, clock, page_size=1, max_pages=5).fetch_signals(_OPEN_CURSOR)

    assert pages_requested == ["1", "2", "3"]
    assert {signal.ticker for signal in signals} == {"BA", "CVX"}


async def test_name_property(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        assert _source(client, clock).name == "regulations_gov"
