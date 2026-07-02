"""Regression: a None-ticker signal (Task 04b, unresolved-entity backfill) must never crash
`GET /conviction` or `GET /proposals` — both endpoints group in-window signals by ticker
(`sorted({s.ticker for s in window})`), which raises `TypeError` comparing `None` to `str` the
moment a mixed-type set needs sorting. This exercises the REAL HTTP routes end-to-end, not just
the use cases directly, since the crash risk is in `interface/app.py`'s route body itself.
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.proposals import ScanProposals
from trading.application.risk import KillSwitch
from trading.application.signal_recording import record_signals
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionParams, ConvictionThresholds
from trading.domain.provenance import Provenance
from trading.domain.sleeves import Sleeve
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app
from trading.strategies.political_signal.strategy import PoliticalSignalStrategy


@pytest.fixture
async def client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    scan_proposals = ScanProposals(
        strategy=PoliticalSignalStrategy(
            thresholds=ConvictionThresholds(), window_days=21, conviction_params=ConvictionParams()
        ),
        read_model=read_model,
        clock=clock,
        thresholds=ConvictionThresholds(),
        conviction_params=ConvictionParams(),
        window_days=21,
    )
    app = create_app(
        AppDeps(
            event_store=event_store,
            read_model=read_model,
            kill_switch=kill_switch,
            sleeve_registry=sleeve_registry,
            evaluators={},
            default_sleeve_id=sleeve.id,
            reconcilers=[],
            order_syncs=[],
            clock=clock,
            conviction_thresholds=ConvictionThresholds(),
            conviction_window_days=21,
            scan_proposals=scan_proposals,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed(
    event_store: ProjectingEventStore, clock: FixedClock, make_signal: SignalFactory
) -> None:
    # A None-ticker (unresolved-entity) signal alongside a normal, high-conviction one, both in
    # the corroboration window — the exact shape a Task 04b house-disclosures backfill produces.
    unresolved = make_signal(
        ticker=None, sentiment=0.9, politician_weight=3.0, provenance=Provenance.BACKFILLED
    )
    resolved = make_signal(
        ticker="LMT", sentiment=1.0, politician_weight=3.0, provenance=Provenance.REAL
    )
    await record_signals(
        event_store=event_store,
        clock=clock,
        signals=[unresolved, resolved],
        correlation_id="seed-1",
    )


async def test_conviction_excludes_none_ticker_signal(
    client: httpx.AsyncClient,
    event_store: ProjectingEventStore,
    clock: FixedClock,
    make_signal: SignalFactory,
) -> None:
    await _seed(event_store, clock, make_signal)

    response = await client.get("/conviction")

    assert response.status_code == 200
    items = response.json()
    tickers = {item["ticker"] for item in items}
    assert None not in tickers
    assert "LMT" in tickers


async def test_proposals_excludes_none_ticker_signal(
    client: httpx.AsyncClient,
    event_store: ProjectingEventStore,
    clock: FixedClock,
    make_signal: SignalFactory,
) -> None:
    await _seed(event_store, clock, make_signal)

    response = await client.get("/proposals")

    assert response.status_code == 200
    body = response.json()
    symbols = {p["symbol"] for p in body["proposals"]} | {p["symbol"] for p in body["predicted"]}
    assert None not in symbols
    assert "LMT" in symbols


async def test_conviction_handles_all_signals_unresolved(
    client: httpx.AsyncClient,
    event_store: ProjectingEventStore,
    clock: FixedClock,
    make_signal: SignalFactory,
) -> None:
    """Even the degenerate all-None-ticker case (the set comprehension would otherwise be a
    single-element {None} set, which sorts fine alone but must still resolve to zero items)."""
    only_unresolved = make_signal(ticker=None, provenance=Provenance.BACKFILLED)
    await record_signals(
        event_store=event_store, clock=clock, signals=[only_unresolved], correlation_id="seed-2"
    )

    response = await client.get("/conviction")

    assert response.status_code == 200
    assert response.json() == []
