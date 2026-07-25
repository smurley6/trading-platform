"""System screen backing read: GET /system/status (UI Phase D §5.5).

Drives the endpoint through create_app + httpx AsyncClient (no network). Asserts the four-pane
backing data: risk caps configured-vs-ceiling + kill-switch state, the scheduler loops (reported
even when disabled, so intervals show), and the read-only data-source config mirror. Scheduler
last/next run stay null here because the test app's intervals are 0 (no scheduler starts).
"""

from collections.abc import AsyncIterator
from decimal import Decimal

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.risk import KillSwitch, RiskLimits
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app
from trading.interface.system_status import DataSourceStatus

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("2"),
    max_position_concentration_pct=Decimal("10"),
    max_drawdown_pct=Decimal("8"),
)
_CEILINGS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("5"),
    max_position_concentration_pct=Decimal("15"),
    max_drawdown_pct=Decimal("10"),
)
_DATA_SOURCES = [
    DataSourceStatus(source_id="news", label="News", enabled=False, reliability_prior=0.6),
    DataSourceStatus(
        source_id="sec_form4", label="SEC Form 4", enabled=True, reliability_prior=0.75
    ),
]


def _make_client(
    *,
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
    risk_wired: bool,
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
            risk_limits=_LIMITS if risk_wired else None,
            live_ceilings=_CEILINGS if risk_wired else None,
            data_sources=_DATA_SOURCES,
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
) -> AsyncIterator[httpx.AsyncClient]:
    c = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve_registry=sleeve_registry,
        clock=clock,
        risk_wired=True,
    )
    async with c:
        yield c


async def test_status_reports_risk_caps_jobs_and_sources(client: httpx.AsyncClient) -> None:
    response = await client.get("/system/status")
    assert response.status_code == 200
    body = response.json()

    # Risk pane: three configured-vs-ceiling caps + kill-switch state.
    assert body["risk"]["enabled"] is True
    assert body["risk"]["kill_switch_engaged"] is False
    caps = {c["name"]: c for c in body["risk"]["caps"]}
    assert set(caps) == {"trade_size", "concentration", "drawdown"}
    assert caps["trade_size"]["configured"] == "2"
    assert caps["trade_size"]["ceiling"] == "5"
    assert all(c["unit"] == "%" for c in caps.values())

    # Jobs pane: all three loops present (disabled here, intervals 0, no run recorded yet).
    jobs = {j["name"]: j for j in body["jobs"]}
    assert set(jobs) == {"reconcile", "ingest", "order_sync"}
    assert all(j["enabled"] is False for j in jobs.values())
    assert all(j["last_run"] is None and j["next_run"] is None for j in jobs.values())

    # Data sources pane: the read-only config mirror.
    sources = {s["source_id"]: s for s in body["data_sources"]}
    assert sources["news"]["enabled"] is False
    assert sources["sec_form4"]["enabled"] is True
    assert sources["sec_form4"]["reliability_prior"] == 0.75


async def test_status_reflects_engaged_kill_switch(
    client: httpx.AsyncClient, kill_switch: KillSwitch
) -> None:
    kill_switch.engage()
    body = (await client.get("/system/status")).json()
    assert body["risk"]["kill_switch_engaged"] is True


async def test_status_risk_disabled_shape_when_limits_unwired(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
) -> None:
    c = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve_registry=sleeve_registry,
        clock=clock,
        risk_wired=False,
    )
    async with c:
        body = (await c.get("/system/status")).json()
    assert body["risk"]["enabled"] is False
    assert body["risk"]["caps"] == []
    # Jobs + data sources are still reported.
    assert {j["name"] for j in body["jobs"]} == {"reconcile", "ingest", "order_sync"}
    assert len(body["data_sources"]) == 2
