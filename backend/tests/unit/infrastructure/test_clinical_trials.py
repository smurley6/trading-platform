"""ClinicalTrials.gov v2 /studies adapter driven by a mock HTTP transport — real parsing of a
committed REAL-shaped fixture, status/phase-conditioned sentiment, the ``since`` cursor, page-token
pagination, the empty/404 paths, and the default-OFF no-network guarantee. No network.

The committed fixture is a real-shaped ``/studies`` page with four studies: a Phase 3 COMPLETED
trial whose lead sponsor is a university but whose *collaborator* is Pfizer (bullish, exercises the
collaborator-resolution path), a Moderna-sponsored TERMINATED trial (bearish), a Merck-sponsored
RECRUITING trial (neutral), and a university-sponsored study naming no tradeable entity (the drop
path). Ticker resolution uses the real ``GazetteerEntityResolver`` (as the FDA / Federal Register
tests do).
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
from trading.infrastructure.data_sources.clinical_trials import (
    RELIABILITY_CLINICAL,
    ClinicalTrialsSource,
)
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver

Handler = Callable[[httpx.Request], httpx.Response]

_FIXTURE = Path(__file__).parent / "fixtures" / "clinical_trials" / "studies_recent.json"

# A cursor far before any fixture last-update date so every fresh study passes the window.
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
    **kwargs: Any,
) -> ClinicalTrialsSource:
    kwargs.setdefault("enabled", True)
    return ClinicalTrialsSource(
        client=client,
        entity_resolver=GazetteerEntityResolver(),
        clock=clock,
        **kwargs,
    )


async def test_name_property(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        assert _source(client, clock).name == "clinical_trials"


async def test_emits_real_clinical_signals_with_correct_mapping(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    by_ticker = {signal.ticker: signal for signal in signals}
    # Pfizer (collaborator on the completed trial), Moderna (terminated trial), and Merck
    # (recruiting trial) resolve; the university-sponsored study names no tradeable entity, dropped.
    assert set(by_ticker) == {"PFE", "MRNA", "MRK"}

    pfizer = by_ticker["PFE"]
    assert pfizer.source_type is SignalSource.STATEMENT
    assert pfizer.politician_id is None
    assert pfizer.source_reliability == RELIABILITY_CLINICAL
    assert pfizer.politician_weight >= 0
    # Real federal clinical data with no politician dependency -> REAL, so it can drive a real
    # decision past the anti-placeholder guard.
    assert pfizer.provenance is Provenance.REAL
    # A COMPLETED Phase 3 trial reads bullish (+0.6), resolved via the *collaborator* (Pfizer),
    # since the lead sponsor is an academic center that resolves to no ticker.
    assert pfizer.sentiment == pytest.approx(0.6)
    # id preserves the NCT id for traceability; the full study record is retained for audit.
    assert pfizer.id == "clinical-trial-NCT05326464-PFE"
    assert pfizer.raw_payload["protocolSection"]["identificationModule"]["nctId"] == "NCT05326464"
    assert (
        pfizer.raw_payload["protocolSection"]["sponsorCollaboratorsModule"]["collaborators"][0][
            "name"
        ]
        == "Pfizer"
    )

    # A TERMINATED trial reads bearish (-0.6).
    moderna = by_ticker["MRNA"]
    assert moderna.sentiment == pytest.approx(-0.6)
    assert moderna.raw_payload["protocolSection"]["statusModule"]["overallStatus"] == "TERMINATED"

    # A RECRUITING (in-progress) trial is not yet a directional catalyst -> neutral (0.0).
    merck = by_ticker["MRK"]
    assert merck.sentiment == pytest.approx(0.0)


async def test_created_at_is_last_update_post_date_at_utc_midnight(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    pfizer_at = {s.ticker: s for s in signals}["PFE"].created_at
    assert (pfizer_at.year, pfizer_at.month, pfizer_at.day) == (2026, 6, 29)
    assert (pfizer_at.hour, pfizer_at.minute) == (0, 0)
    assert pfizer_at.tzinfo is not None and pfizer_at.utcoffset() == UTC.utcoffset(None)


async def test_since_cursor_excludes_studies_at_or_before_it(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        # The newest fixture study is posted 2026-06-29; a cursor at that instant excludes every
        # study (strict >), so nothing is emitted.
        signals = await _source(client, clock).fetch_signals(datetime(2026, 6, 29, tzinfo=UTC))

    assert signals == []


async def test_since_cursor_keeps_only_newer_studies(clock: FixedClock) -> None:
    async with _client(_fixture_handler()) as client:
        # Cursor between the Merck recruiting trial (2026-06-25) and the Moderna terminated trial
        # (2026-06-26): the completed Pfizer (06-29) and terminated Moderna (06-26) studies survive;
        # Merck (06-25) is excluded (strict >).
        signals = await _source(client, clock).fetch_signals(datetime(2026, 6, 25, tzinfo=UTC))

    assert {s.ticker for s in signals} == {"PFE", "MRNA"}


async def test_empty_results_yield_no_signal(clock: FixedClock) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"studies": [], "nextPageToken": None})

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


async def test_request_shapes_window_sort_and_fields(clock: FixedClock) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["path"] = request.url.path
        captured["advanced"] = request.url.params.get("filter.advanced")
        captured["sort"] = request.url.params.get("sort")
        captured["page_size"] = request.url.params.get("pageSize")
        captured["fields"] = request.url.params.get("fields")
        return httpx.Response(200, json={"studies": [], "nextPageToken": None})

    async with _client(handler) as client:
        await _source(client, clock, lookback_days=14).fetch_signals(_OPEN_CURSOR)

    assert captured["host"] == "clinicaltrials.gov"
    assert captured["path"] == "/api/v2/studies"
    # FixedClock default is 2026-06-12; a 14-day lookback starts 2026-05-29.
    assert captured["advanced"] == "AREA[LastUpdatePostDate]RANGE[2026-05-29,MAX]"
    assert captured["sort"] == "LastUpdatePostDate:desc"
    assert captured["page_size"] == "100"
    assert captured["fields"] is not None and "LastUpdatePostDateStruct" in captured["fields"]


async def test_follows_page_token_pagination(clock: FixedClock) -> None:
    """Two pages linked by nextPageToken are both walked and their signals combined."""
    fixture = _load_fixture()
    # Keep only the COMPLETED Pfizer study on page one, pointing at page two; page two has the rest.
    page_one_studies = [fixture["studies"][0]]
    remaining = fixture["studies"][1:]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.params.get("pageToken") is None:
            return httpx.Response(
                200, json={"studies": page_one_studies, "nextPageToken": "CURSORTOKEN"}
            )
        return httpx.Response(200, json={"studies": remaining, "nextPageToken": None})

    async with _client(handler) as client:
        signals = await _source(client, clock).fetch_signals(_OPEN_CURSOR)

    assert len(calls) == 2
    # Both pages contributed: Pfizer from page one, Moderna + Merck from page two.
    assert {s.ticker for s in signals} == {"PFE", "MRNA", "MRK"}


async def test_disabled_source_makes_no_network_call_and_yields_nothing(clock: FixedClock) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_load_fixture())

    async with _client(handler) as client:
        # Default OFF: constructed without enabled=True.
        source = ClinicalTrialsSource(
            client=client,
            entity_resolver=GazetteerEntityResolver(),
            clock=clock,
        )
        signals = await source.fetch_signals(_OPEN_CURSOR)

    assert signals == []
    # No socket touched on the default/disabled path.
    assert calls == []


async def test_max_pages_caps_the_walk(clock: FixedClock) -> None:
    """A page that always advertises a next token is walked at most max_pages times."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"studies": [], "nextPageToken": "NEVERENDING"})

    async with _client(handler) as client:
        await _source(client, clock, max_pages=3).fetch_signals(_OPEN_CURSOR)

    assert len(calls) == 3
