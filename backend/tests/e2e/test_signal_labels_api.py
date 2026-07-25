"""Research surface: GET /signals/labels (UI Phase D §5.5).

Drives the endpoint through create_app + httpx AsyncClient (no network). Asserts the join of the
recent-signals window against the outcome_labels projection, the graceful disabled shape before any
labeler run, and that the read appends NO events. Mirrors test_risk_precheck_api.py.
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.ports.outcome_labels import OutcomeLabel
from trading.application.risk import KillSwitch
from trading.application.signal_recording import record_signals
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.infrastructure.persistence.memory_outcome_labels import InMemoryOutcomeLabels
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app

_RUN_ID = "run-phase-d"


def _label(signal_id: str, *, horizon: int, benchmark: str, ret: float | None) -> OutcomeLabel:
    return OutcomeLabel(
        signal_id=signal_id,
        ticker="AAPL",
        horizon_days=horizon,
        abnormal_return=ret,
        benchmark=benchmark,
        status="ok" if ret is not None else "unresolvable",
        reason=None if ret is not None else "insufficient_forward_bars",
        label_run_id=_RUN_ID,
    )


def _make_client(
    *,
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
    outcome_labels: InMemoryOutcomeLabels,
) -> httpx.AsyncClient:
    app = create_app(
        AppDeps(
            event_store=event_store,
            read_model=read_model,
            kill_switch=kill_switch,
            sleeve_registry=sleeve_registry,
            evaluators={},
            default_sleeve_id="sleeve-test",
            reconcilers=[],
            order_syncs=[],
            clock=clock,
            conviction_thresholds=ConvictionThresholds(),
            conviction_window_days=21,
            outcome_labels=outcome_labels,
        )
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
    outcome_labels: InMemoryOutcomeLabels,
) -> AsyncIterator[httpx.AsyncClient]:
    c = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve_registry=sleeve_registry,
        clock=clock,
        outcome_labels=outcome_labels,
    )
    async with c:
        yield c


@pytest.fixture
def outcome_labels() -> InMemoryOutcomeLabels:
    return InMemoryOutcomeLabels()


async def _seed_signals(
    event_store: ProjectingEventStore, clock: FixedClock, make_signal: SignalFactory
) -> list[str]:
    ids = []
    for i in range(2):
        signal = make_signal(ticker="AAPL")
        await record_signals(
            event_store=event_store, clock=clock, signals=[signal], correlation_id=f"corr-{i}"
        )
        ids.append(signal.id)
    return ids


async def test_labels_joined_to_recent_signals(
    client: httpx.AsyncClient,
    event_store: ProjectingEventStore,
    clock: FixedClock,
    make_signal: SignalFactory,
    outcome_labels: InMemoryOutcomeLabels,
) -> None:
    ids = await _seed_signals(event_store, clock, make_signal)
    await outcome_labels.replace_run(
        _RUN_ID,
        [
            _label(ids[0], horizon=1, benchmark="sector:XLK", ret=0.01),
            _label(ids[0], horizon=5, benchmark="sector:XLK", ret=0.02),
            _label(ids[0], horizon=20, benchmark="sector:XLK", ret=None),
            _label(ids[1], horizon=1, benchmark="market", ret=-0.03),
        ],
    )

    before = len(await event_store.read_all())
    response = await client.get("/signals/labels", params={"limit": "100"})
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["label_run_id"] == _RUN_ID
    assert len(body["labels"]) == 4
    by_signal = {label["signal_id"] for label in body["labels"]}
    assert by_signal == {ids[0], ids[1]}
    # Benchmark prefixes preserved for the sector-vs-market split; unresolvable => null return.
    unresolvable = next(
        label
        for label in body["labels"]
        if label["signal_id"] == ids[0] and label["horizon_days"] == 20
    )
    assert unresolvable["abnormal_return"] is None
    assert unresolvable["status"] == "unresolvable"
    market = next(label for label in body["labels"] if label["signal_id"] == ids[1])
    assert market["benchmark"] == "market"

    # Read-only: no events appended by the GET.
    assert len(await event_store.read_all()) == before


async def test_disabled_shape_before_any_labeler_run(client: httpx.AsyncClient) -> None:
    response = await client.get("/signals/labels")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["label_run_id"] is None
    assert body["labels"] == []
