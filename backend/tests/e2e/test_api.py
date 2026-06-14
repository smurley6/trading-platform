"""HTTP surface: signals in, events out, kill switch as a recorded user action."""

from collections.abc import AsyncIterator
from decimal import Decimal

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.backtest import Backtest, BacktestRequest, BacktestResult
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.live_trading import AssessLiveReadiness, RequestLiveTrading
from trading.application.projections import signals_from_events
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch, RiskLimits
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.events import EventType
from trading.domain.sleeves import Sleeve
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory
from trading.infrastructure.market_data.historical import synthetic_price_series
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.interface.app import AppDeps, create_app
from trading.strategies.trivial import TrivialStrategy


@pytest.fixture
async def client(
    event_store: InMemoryEventStore,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    backtest = Backtest(env_factory=InMemoryBacktestEnvFactory())

    async def run_backtest() -> BacktestResult:
        signals = signals_from_events(await event_store.read_all())
        times = sorted({s.created_at for s in signals})
        prices = synthetic_price_series({"AAPL": Decimal("100"), "MSFT": Decimal("200")}, times)
        return await backtest.run(
            BacktestRequest(
                strategy=TrivialStrategy(position_size=Decimal("5")),
                signals=signals,
                risk_limits=RiskLimits(
                    max_trade_pct_of_sleeve=Decimal("50"),
                    max_position_concentration_pct=Decimal("90"),
                    max_drawdown_pct=Decimal("90"),
                ),
                prices=prices,
                starting_cash=Decimal("10000"),
                window_days=21,
            )
        )

    tight = RiskLimits(
        max_trade_pct_of_sleeve=Decimal("5"),
        max_position_concentration_pct=Decimal("15"),
        max_drawdown_pct=Decimal("10"),
    )
    assess = AssessLiveReadiness(
        allow_live=True,
        live_broker_configured=True,
        configured_limits=tight,
        live_ceilings=tight,
        kill_switch=kill_switch,
        reconcilers={sleeve.id: reconcile},
        event_store=event_store,
        clock=clock,
    )
    request_live = RequestLiveTrading(
        registry=sleeve_registry, readiness=assess, event_store=event_store, clock=clock
    )

    app = create_app(
        AppDeps(
            event_store=event_store,
            kill_switch=kill_switch,
            sleeve_registry=sleeve_registry,
            evaluators={sleeve.id: evaluate_signals},
            default_sleeve_id=sleeve.id,
            reconcilers=[reconcile],
            clock=clock,
            conviction_thresholds=ConvictionThresholds(),
            conviction_window_days=21,
            backtest_runner=run_backtest,
            assess_readiness=assess,
            request_live=request_live,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


_SIGNAL = {
    "source_type": "statement",
    "ticker": "AAPL",
    "sentiment": 0.9,
    "raw_payload": {"headline": "Senator praises Apple"},
}


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_signal_to_order_via_api(client: httpx.AsyncClient) -> None:
    response = await client.post("/signals", json=[_SIGNAL])
    assert response.status_code == 200
    body = response.json()
    assert len(body["orders"]) == 1
    assert body["orders"][0]["status"] == "filled"

    events = await client.get("/events", params={"correlation_id": body["correlation_id"]})
    types = {e["event_type"] for e in events.json()}
    assert EventType.ORDER_FILLED in types


async def test_kill_switch_is_recorded_and_enforced(client: httpx.AsyncClient) -> None:
    response = await client.post("/kill-switch", json={"engaged": True})
    assert response.json() == {"engaged": True}

    blocked = await client.post("/signals", json=[_SIGNAL])
    assert blocked.json()["orders"] == []

    events = await client.get("/events", params={"stream_id": "user-actions"})
    assert [e["event_type"] for e in events.json()] == [EventType.USER_ACTION]


async def test_reconcile_endpoint(client: httpx.AsyncClient) -> None:
    await client.post("/signals", json=[_SIGNAL])
    response = await client.post("/reconcile")
    assert response.status_code == 200
    reports = response.json()
    assert len(reports) == 1
    assert reports[0]["clean"] is True


async def test_backtest_endpoint(client: httpx.AsyncClient) -> None:
    await client.post("/signals", json=[_SIGNAL])
    response = await client.post("/backtest")
    assert response.status_code == 200
    body = response.json()
    assert body["starting_equity"] == 10000.0
    assert len(body["trades"]) == 1
    assert body["trades"][0]["symbol"] == "AAPL"
    assert len(body["equity_curve"]) >= 1


async def test_live_readiness_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.post("/sleeves/sleeve-test/readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["sleeve_id"] == "sleeve-test"
    assert body["ready"] is True
    assert {c["name"] for c in body["checks"]} >= {"master_switch", "kill_switch_operational"}


async def test_go_live_requires_confirmation(client: httpx.AsyncClient) -> None:
    blocked = await client.post(
        "/sleeves/sleeve-test/mode", json={"mode": "live", "confirm": False}
    )
    assert blocked.status_code == 409
    sleeves = await client.get("/sleeves")
    assert sleeves.json()[0]["mode"] == "paper"


async def test_go_live_with_confirmation_flips_mode(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/sleeves/sleeve-test/mode", json={"mode": "live", "confirm": True}
    )
    assert response.status_code == 200
    assert response.json()["sleeve"]["mode"] == "live"
    assert response.json()["readiness"]["ready"] is True

    sleeves = await client.get("/sleeves")
    assert sleeves.json()[0]["mode"] == "live"

    back = await client.post("/sleeves/sleeve-test/mode", json={"mode": "paper"})
    assert back.json()["sleeve"]["mode"] == "paper"


async def test_unknown_sleeve_mode_404(client: httpx.AsyncClient) -> None:
    response = await client.post("/sleeves/nope/mode", json={"mode": "paper"})
    assert response.status_code == 404
