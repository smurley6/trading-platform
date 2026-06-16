"""API-key auth on mutating endpoints: open when unset, enforced when set."""

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.sleeves import Sleeve
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app

_SIGNAL = {"source_type": "statement", "ticker": "AAPL", "sentiment": 0.9, "raw_payload": {}}


def _deps(
    *,
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    clock: FixedClock,
    api_key: str | None,
) -> AppDeps:
    return AppDeps(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve_registry=sleeve_registry,
        evaluators={sleeve.id: evaluate_signals},
        default_sleeve_id=sleeve.id,
        reconcilers=[],
        order_syncs=[],
        clock=clock,
        conviction_thresholds=ConvictionThresholds(),
        conviction_window_days=21,
        api_key=api_key,
    )


async def _client(deps: AppDeps) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(deps))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def open_client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    deps = _deps(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        evaluate_signals=evaluate_signals,
        clock=clock,
        api_key=None,
    )
    async with await _client(deps) as c:
        yield c


@pytest.fixture
async def secured_client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    deps = _deps(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        evaluate_signals=evaluate_signals,
        clock=clock,
        api_key="s3cret",
    )
    async with await _client(deps) as c:
        yield c


# --- open mode (api_key unset) ---------------------------------------------------------------


async def test_open_mode_allows_mutations_without_key(open_client: httpx.AsyncClient) -> None:
    assert (await open_client.post("/signals", json=[_SIGNAL])).status_code == 200
    assert (await open_client.post("/kill-switch", json={"engaged": True})).status_code == 200


# --- secured mode (api_key set) --------------------------------------------------------------


async def test_secured_mode_rejects_missing_key(secured_client: httpx.AsyncClient) -> None:
    resp = await secured_client.post("/signals", json=[_SIGNAL])
    assert resp.status_code == 401


async def test_secured_mode_rejects_wrong_key(secured_client: httpx.AsyncClient) -> None:
    resp = await secured_client.post(
        "/kill-switch", json={"engaged": True}, headers={"X-API-Key": "nope"}
    )
    assert resp.status_code == 401


async def test_secured_mode_accepts_bearer(secured_client: httpx.AsyncClient) -> None:
    resp = await secured_client.post(
        "/signals", json=[_SIGNAL], headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code == 200


async def test_secured_mode_accepts_x_api_key(secured_client: httpx.AsyncClient) -> None:
    resp = await secured_client.post(
        "/kill-switch", json={"engaged": False}, headers={"X-API-Key": "s3cret"}
    )
    assert resp.status_code == 200


async def test_get_reads_stay_open_when_secured(secured_client: httpx.AsyncClient) -> None:
    # GET read endpoints are never guarded, even with auth enabled.
    assert (await secured_client.get("/events")).status_code == 200
    assert (await secured_client.get("/sleeves")).status_code == 200
    assert (await secured_client.get("/kill-switch")).status_code == 200


async def test_health_endpoints_stay_open_when_secured(secured_client: httpx.AsyncClient) -> None:
    assert (await secured_client.get("/healthz")).status_code == 200
    assert (await secured_client.get("/readyz")).status_code == 200
    assert (await secured_client.get("/metrics")).status_code == 200
