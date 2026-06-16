"""Observability surface: /healthz, /readyz, /metrics, request-id propagation."""

from collections.abc import AsyncIterator

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.infrastructure.observability.metrics import MetricsRegistry
from trading.infrastructure.persistence.health import DatabaseReadiness
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app


def _deps(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    clock: FixedClock,
    *,
    metrics: MetricsRegistry,
    enable_metrics: bool = True,
) -> AppDeps:
    return AppDeps(
        event_store=event_store,
        read_model=read_model,
        kill_switch=KillSwitch(),
        sleeve_registry=SleeveRegistry([]),
        evaluators={},
        default_sleeve_id="sleeve-political",
        reconcilers=[],
        order_syncs=[],
        clock=clock,
        conviction_thresholds=ConvictionThresholds(),
        conviction_window_days=21,
        metrics=metrics,
        enable_metrics=enable_metrics,
        db_readiness=DatabaseReadiness(None),
    )


@pytest.fixture
async def client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    clock: FixedClock,
) -> AsyncIterator[tuple[httpx.AsyncClient, MetricsRegistry]]:
    metrics = MetricsRegistry()
    deps = _deps(event_store, read_model, clock, metrics=metrics)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, metrics


async def test_healthz_liveness(client: tuple[httpx.AsyncClient, MetricsRegistry]) -> None:
    c, _ = client
    resp = await c.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_readyz_ready_with_in_memory(
    client: tuple[httpx.AsyncClient, MetricsRegistry],
) -> None:
    c, _ = client
    resp = await c.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


async def test_existing_health_still_works(
    client: tuple[httpx.AsyncClient, MetricsRegistry],
) -> None:
    c, _ = client
    assert (await c.get("/health")).json() == {"status": "ok"}


async def test_metrics_reflects_traffic(
    client: tuple[httpx.AsyncClient, MetricsRegistry],
) -> None:
    c, _ = client
    await c.get("/health")
    resp = await c.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds_bucket" in body
    assert 'path="/health"' in body


async def test_request_id_header_is_echoed(
    client: tuple[httpx.AsyncClient, MetricsRegistry],
) -> None:
    c, _ = client
    resp = await c.get("/health", headers={"X-Request-ID": "trace-123"})
    assert resp.headers["X-Request-ID"] == "trace-123"


async def test_request_id_generated_when_absent(
    client: tuple[httpx.AsyncClient, MetricsRegistry],
) -> None:
    c, _ = client
    resp = await c.get("/health")
    assert resp.headers.get("X-Request-ID")


async def test_metrics_can_be_disabled(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    clock: FixedClock,
) -> None:
    deps = _deps(event_store, read_model, clock, metrics=MetricsRegistry(), enable_metrics=False)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        assert (await c.get("/metrics")).status_code == 404


class _FailingReadiness(DatabaseReadiness):
    """Pretends a database is configured but unreachable."""

    def __init__(self) -> None:
        super().__init__(None)

    @property
    def configured(self) -> bool:
        return True

    async def check(self) -> bool:
        raise RuntimeError("db down")


async def test_readyz_503_when_database_unreachable(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    clock: FixedClock,
) -> None:
    deps = _deps(event_store, read_model, clock, metrics=MetricsRegistry())
    deps.db_readiness = _FailingReadiness()
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/readyz")
        assert resp.status_code == 503
        assert resp.json()["detail"]["checks"]["database"] == "error"
