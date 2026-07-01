"""Shared capital pool surface: GET /capital.

The one general capital location both algos draw from = the live account (its buying power /
cash / equity) plus each algo's slice of the single book. Driven through a stub pool broker +
stub quote feed injected via AppDeps — no network. Mirrors tests/e2e/test_quotes_api.py.
"""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.ports.market_data import Quote
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.orders import OrderSide, OrderType
from trading.domain.sleeves import Sleeve
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app

_AS_OF = datetime(2026, 6, 13, 20, 0, tzinfo=UTC)


class StubQuoteFeed:
    def __init__(self, *, enabled: bool, prices: dict[str, Decimal]) -> None:
        self._enabled = enabled
        self._prices = prices

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        return [
            Quote(
                symbol=s,
                bid=self._prices[s],
                ask=self._prices[s],
                last=self._prices[s],
                as_of=_AS_OF,
            )
            for s in symbols
            if s in self._prices
        ]


class StubPoolBroker:
    """The one shared account. Only get_account is read by /capital; the rest satisfy BrokerPort."""

    def __init__(self, *, cash: Decimal, equity: Decimal, buying_power: Decimal) -> None:
        self._account = BrokerAccount(cash=cash, equity=equity, buying_power=buying_power)

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
        return []

    async def get_account(self) -> BrokerAccount:
        return self._account


def _make_client(
    *,
    event_store: ProjectingEventStore,
    read_model: InMemoryReadModel,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
    feed: StubQuoteFeed | None,
    pool_broker: StubPoolBroker | None,
) -> httpx.AsyncClient:
    app = create_app(
        AppDeps(
            event_store=event_store,
            read_model=read_model,
            kill_switch=kill_switch,
            sleeve_registry=sleeve_registry,
            evaluators={sleeve.id: evaluate_signals},
            default_sleeve_id=sleeve.id,
            reconcilers=[reconcile],
            order_syncs=[],
            clock=clock,
            conviction_thresholds=ConvictionThresholds(),
            conviction_window_days=21,
            pool_broker=pool_broker,
            quote_feed=feed,
            default_watchlist=["AAPL"],
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
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    client = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        evaluate_signals=evaluate_signals,
        reconcile=reconcile,
        clock=clock,
        feed=StubQuoteFeed(enabled=True, prices={"AAPL": Decimal("232.12")}),
        pool_broker=StubPoolBroker(
            cash=Decimal("55000.00"), equity=Decimal("100000.00"), buying_power=Decimal("180000.00")
        ),
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
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    client = _make_client(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        evaluate_signals=evaluate_signals,
        reconcile=reconcile,
        clock=clock,
        feed=None,
        pool_broker=None,  # None => pool disabled
    )
    async with client:
        yield client


_SIGNAL = {
    "source_type": "statement",
    "ticker": "AAPL",
    "sentiment": 0.9,
    "raw_payload": {"headline": "Senator praises Apple"},
}


async def test_capital_reports_live_pool_and_per_algo_slice(
    enabled_client: httpx.AsyncClient,
) -> None:
    # One algo takes a position in the shared book (5 AAPL @ 100 via the trivial strategy).
    await enabled_client.post("/signals", json=[_SIGNAL])

    response = await enabled_client.get("/capital")
    assert response.status_code == 200
    body = response.json()

    # The single shared pool = the live account (buying power is the headline).
    assert body["enabled"] is True
    assert body["cash"] == "55000.00"
    assert body["equity"] == "100000.00"
    assert body["buying_power"] == "180000.00"

    # Attribution: this algo's slice, marked to the live feed (5 * 232.12 = 1160.60).
    allocations = {a["sleeve_id"]: a for a in body["allocations"]}
    slice_ = allocations["sleeve-test"]
    assert slice_["strategy_id"] == "trivial"
    assert slice_["positions"] == 1
    assert slice_["market_value"] == "1160.60"


async def test_capital_disabled_when_no_account_wired(disabled_client: httpx.AsyncClient) -> None:
    response = await disabled_client.get("/capital")
    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "cash": "0",
        "equity": "0",
        "buying_power": "0",
        "allocations": [],
    }
