"""Congress.gov Bill Actions adapter driven by a mock HTTP transport — real parsing of committed
REAL-shaped fixtures, the two-hop (list -> per-bill actions) flow, ticker resolution from bill
titles, the ``since`` cursor, request/auth shaping, the empty-window 404, HTTP errors, and the
default-OFF no-network guarantee. No network."""

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
from trading.infrastructure.data_sources.congress_bills import (
    RELIABILITY_BILL_ACTIONS,
    SENTIMENT_PASSAGE,
    SENTIMENT_REFERRAL,
    CongressBillsSource,
)
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURES = Path(__file__).parent / "fixtures" / "congress_bills"
_BILL_LIST = json.loads((_FIXTURES / "bill_list.json").read_text())
_ACTIONS_HR1421 = json.loads((_FIXTURES / "actions_hr1421.json").read_text())
_ACTIONS_S88 = json.loads((_FIXTURES / "actions_s88.json").read_text())

# A cursor far before any fixture action date so every fresh action passes the window.
_OPEN_CURSOR = datetime(2026, 1, 1, tzinfo=UTC)


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _congress_handler(record: list[httpx.Request] | None = None) -> Handler:
    """Stand-in for api.congress.gov: routes the bill-list and per-bill action sub-resources."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        path = request.url.path
        if path == "/v3/bill":
            return httpx.Response(200, json=_BILL_LIST)
        if path == "/v3/bill/118/hr/1421/actions":
            return httpx.Response(200, json=_ACTIONS_HR1421)
        if path == "/v3/bill/118/s/88/actions":
            return httpx.Response(200, json=_ACTIONS_S88)
        # HR2200 (Rural Broadband) resolves to no ticker, so its actions are never fetched.
        return httpx.Response(404, json={"error": "not found"})

    return handler


def _source(
    client: httpx.AsyncClient, clock: FixedClock, *, enabled: bool = True, **kwargs: Any
) -> CongressBillsSource:
    return CongressBillsSource(
        client=client,
        entity_resolver=GazetteerEntityResolver(),
        clock=clock,
        enabled=enabled,
        **kwargs,
    )


async def test_emits_real_statement_signals_from_bill_actions(clock: FixedClock) -> None:
    async with _client(_congress_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    # HR1421 -> Boeing (BA): a passage action + a referral action. S88 -> Lockheed (LMT): one
    # referral. HR2200 -> Rural Broadband: no resolvable ticker, dropped entirely.
    by_ticker: dict[str, list[Any]] = {}
    for signal in signals:
        by_ticker.setdefault(signal.ticker, []).append(signal)
    assert set(by_ticker) == {"BA", "LMT"}
    assert all(s.source_type is SignalSource.STATEMENT for s in signals)
    assert all(s.provenance is Provenance.REAL for s in signals)
    assert all(s.source_reliability == RELIABILITY_BILL_ACTIONS for s in signals)
    assert all(s.politician_weight >= 0 for s in signals)

    passage = next(
        s for s in by_ticker["BA"] if "passed" in s.raw_payload["action"]["text"].lower()
    )
    assert passage.sentiment == pytest.approx(SENTIMENT_PASSAGE)  # passage reads bullish
    assert passage.politician_id == "S001176"  # primary sponsor BioGuide ID

    lmt = by_ticker["LMT"][0]
    assert lmt.sentiment == pytest.approx(SENTIMENT_REFERRAL)  # referral reads mildly positive
    assert lmt.politician_id == "W000817"


async def test_raw_payload_retains_bill_and_action(clock: FixedClock) -> None:
    async with _client(_congress_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    passage = next(
        s
        for s in signals
        if s.ticker == "BA" and s.raw_payload["action"].get("actionCode") == "H37100"
    )
    payload = passage.raw_payload
    assert payload["title"] == "Boeing Aircraft Procurement Modernization Act"
    assert payload["congress"] == 118
    assert payload["type"] == "HR"
    assert payload["number"] == "1421"
    # Full action record (including recorded votes) is retained for the audit log.
    assert payload["action"]["recordedVotes"][0]["rollNumber"] == 210
    assert payload["sponsors"][0]["bioguideId"] == "S001176"


async def test_created_at_combines_action_date_and_time_utc(clock: FixedClock) -> None:
    async with _client(_congress_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    passage = next(
        s
        for s in signals
        if s.ticker == "BA" and s.raw_payload["action"].get("actionCode") == "H37100"
    )
    # actionDate 2026-06-05 + actionTime 14:20:11 -> 2026-06-05T14:20:11Z.
    at = passage.created_at
    assert (at.year, at.month, at.day, at.hour, at.minute, at.second) == (2026, 6, 5, 14, 20, 11)
    assert at.tzinfo is not None and at.utcoffset() is not None


async def test_since_cursor_excludes_actions_at_or_before_it(clock: FixedClock) -> None:
    async with _client(_congress_handler()) as client:
        # Cursor after every fixture action (latest is 2026-06-05T14:20:11Z) -> nothing fresh.
        cursor = datetime(2026, 6, 6, tzinfo=UTC)
        signals = await _source(client, clock).fetch_signals(cursor)

    assert signals == []


async def test_since_cursor_keeps_only_newer_actions(clock: FixedClock) -> None:
    async with _client(_congress_handler()) as client:
        # Cursor between the HR1421 referral (2026-05-20) and its passage (2026-06-05): only the
        # passage-and-later actions survive. S88's referral is 2026-06-04 -> also survives.
        cursor = datetime(2026, 5, 25, tzinfo=UTC)
        signals = await _source(client, clock).fetch_signals(cursor)

    dates = {s.created_at.date().isoformat() for s in signals}
    assert "2026-05-20" not in dates
    assert {"2026-06-05", "2026-06-04"} <= dates


async def test_request_shapes_window_paging_and_key(clock: FixedClock) -> None:
    record: list[httpx.Request] = []
    async with _client(_congress_handler(record)) as client:
        await _source(client, clock, api_key="DEMO_KEY", lookback_days=30).fetch_signals(
            _OPEN_CURSOR
        )

    bill_req = next(r for r in record if r.url.path == "/v3/bill")
    assert bill_req.url.host == "api.congress.gov"
    params = bill_req.url.params
    # FixedClock default is 2026-06-12; a 30-day lookback starts 2026-05-13.
    assert params.get("fromDateTime") == "2026-05-13T14:30:00Z"
    assert params.get("toDateTime") == "2026-06-12T14:30:00Z"
    assert params.get("limit") == "100"
    assert params.get("offset") == "0"
    assert params.get("api_key") == "DEMO_KEY"
    assert params.get("format") == "json"

    # The key is also forwarded on the per-bill actions sub-resource request.
    actions_req = next(r for r in record if r.url.path.endswith("/actions"))
    assert actions_req.url.params.get("api_key") == "DEMO_KEY"


async def test_api_key_omitted_when_not_configured(clock: FixedClock) -> None:
    record: list[httpx.Request] = []
    async with _client(_congress_handler(record)) as client:
        await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    bill_req = next(r for r in record if r.url.path == "/v3/bill")
    assert bill_req.url.params.get("api_key") is None


async def test_unresolvable_bill_actions_never_fetched(clock: FixedClock) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return _congress_handler()(request)

    async with _client(handler) as client:
        await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    # HR2200 has no resolvable ticker, so its /actions sub-resource is never requested.
    assert not any("2200" in p for p in requested_paths)
    assert any("1421/actions" in p for p in requested_paths)
    assert any("88/actions" in p for p in requested_paths)


async def test_empty_window_404_is_treated_as_no_results(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no bills in window"})

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    assert signals == []


async def test_server_error_raises(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _source(client, clock).fetch_signals(_OPEN_CURSOR)


async def test_disabled_source_makes_no_network_call(clock: FixedClock) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_BILL_LIST)

    async with _client(handler) as client:
        # Default OFF: enabled defaults to False, so no request is ever issued.
        source = CongressBillsSource(
            client=client, entity_resolver=GazetteerEntityResolver(), clock=clock
        )
        signals = await source.fetch_signals(_OPEN_CURSOR)

    assert signals == []
    assert calls == []


async def test_name_property(clock: FixedClock) -> None:
    async with _client(_congress_handler()) as client:
        assert _source(client, clock).name == "congress_bills"
