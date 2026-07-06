"""Read-only risk pre-check surface: GET /risk/precheck (UI Phase C §5.2).

Drives PrecheckRisk through create_app + httpx AsyncClient (no network), asserting the DTO
shape, unknown-sleeve 404, invalid side 422, the disabled shape when unconfigured, and — the
load-bearing contract — that the read-only GET appends NO events. Mirrors test_capital_api.py.
"""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.ports.market_data import Quote
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.application.risk_precheck import PrecheckRisk
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.orders import OrderSide, OrderType
from trading.domain.sleeves import Sleeve
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app

_AS_OF = datetime(2026, 6, 13, 20, 0, tzinfo=UTC)

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("10"),
    max_position_concentration_pct=Decimal("25"),
    max_drawdown_pct=Decimal("20"),
)


class _StubBroker:
    def __init__(self, positions: Sequence[BrokerPosition] = ()) -> None:
        self._account = BrokerAccount(
            cash=Decimal("10000"), equity=Decimal("10000"), buying_power=Decimal("0")
        )
        self._positions = list(positions)

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return self._positions

    async def get_account(self) -> BrokerAccount:
        return self._account


class _StubMarketData:
    async def get_quote(self, symbol: str) -> Quote:
        last = Decimal("100")
        return Quote(symbol=symbol, bid=last, ask=last, last=last, as_of=_AS_OF)


def _make_client(
    *,
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
    configured: bool,
) -> httpx.AsyncClient:
    precheckers: dict[str, PrecheckRisk] = {}
    if configured:
        precheckers[sleeve.id] = PrecheckRisk(
            risk_gate=RiskGate(_LIMITS, kill_switch),
            risk_limits=_LIMITS,
            broker=_StubBroker(),
            market_data=_StubMarketData(),
            kill_switch=kill_switch,
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
            precheckers=precheckers,
        )
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def enabled_client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    client = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        clock=clock,
        configured=True,
    )
    async with client:
        yield client


@pytest.fixture
async def disabled_client(
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    client = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        clock=clock,
        configured=False,
    )
    async with client:
        yield client


async def test_precheck_returns_dto_with_checks(
    enabled_client: httpx.AsyncClient, event_store: ProjectingEventStore
) -> None:
    before = len(await event_store.read_all())
    response = await enabled_client.get(
        "/risk/precheck", params={"symbol": "LMT", "side": "buy", "quantity": "5"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["symbol"] == "LMT"
    assert body["side"] == "buy"
    assert body["requested_quantity"] == "5"
    assert body["verdict"] == "approved"
    names = {c["name"] for c in body["checks"]}
    assert names == {
        "kill_switch",
        "price",
        "sleeve_equity",
        "drawdown",
        "trade_size",
        "concentration",
        "buying_power",
    }
    assert all(c["passed"] for c in body["checks"])

    # Read-only: no events appended by the GET.
    after = len(await event_store.read_all())
    assert after == before


async def test_precheck_sell_omits_buy_only_checks(enabled_client: httpx.AsyncClient) -> None:
    response = await enabled_client.get(
        "/risk/precheck", params={"symbol": "LMT", "side": "sell", "quantity": "5"}
    )
    assert response.status_code == 200
    names = {c["name"] for c in response.json()["checks"]}
    assert "concentration" not in names
    assert "buying_power" not in names


async def test_precheck_unknown_sleeve_is_404(enabled_client: httpx.AsyncClient) -> None:
    response = await enabled_client.get(
        "/risk/precheck",
        params={"symbol": "LMT", "side": "buy", "quantity": "5", "sleeve_id": "nope"},
    )
    assert response.status_code == 404


async def test_precheck_invalid_side_is_422(enabled_client: httpx.AsyncClient) -> None:
    response = await enabled_client.get(
        "/risk/precheck", params={"symbol": "LMT", "side": "hold", "quantity": "5"}
    )
    assert response.status_code == 422


async def test_precheck_disabled_shape_when_unconfigured(
    disabled_client: httpx.AsyncClient,
) -> None:
    response = await disabled_client.get(
        "/risk/precheck", params={"symbol": "LMT", "side": "buy", "quantity": "5"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["checks"] == []
