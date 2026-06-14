"""Live market-data surface: /quotes, /quotes/tracked, /positions/live.

Drives the app through a stub QuoteFeedPort injected via AppDeps — no network — covering
both enabled and disabled feed states. Mirrors tests/e2e/test_api.py.
"""

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from tests.conftest import FixedClock
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.ports.market_data import Quote
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.sleeves import Sleeve
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.interface.app import AppDeps, create_app

_AS_OF = datetime(2026, 6, 13, 20, 0, tzinfo=UTC)


class StubQuoteFeed:
    def __init__(self, *, enabled: bool, prices: dict[str, Decimal]) -> None:
        self._enabled = enabled
        self._prices = prices
        self.calls: list[list[str]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        self.calls.append(list(symbols))
        out: list[Quote] = []
        for symbol in symbols:
            price = self._prices.get(symbol)
            if price is None:
                continue
            out.append(Quote(symbol=symbol, bid=price, ask=price, last=price, as_of=_AS_OF))
        return out


def _make_client(
    *,
    event_store: InMemoryEventStore,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
    feed: StubQuoteFeed | None,
    watchlist: list[str],
) -> httpx.AsyncClient:
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
            quote_feed=feed,
            default_watchlist=watchlist,
        )
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def enabled_client(
    event_store: InMemoryEventStore,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
) -> AsyncIterator[tuple[httpx.AsyncClient, StubQuoteFeed]]:
    feed = StubQuoteFeed(
        enabled=True,
        prices={"AAPL": Decimal("232.12"), "MSFT": Decimal("410.20")},
    )
    client = _make_client(
        event_store=event_store,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        evaluate_signals=evaluate_signals,
        reconcile=reconcile,
        clock=clock,
        feed=feed,
        watchlist=["AAPL", "MSFT"],
    )
    async with client:
        yield client, feed


@pytest.fixture
async def disabled_client(
    event_store: InMemoryEventStore,
    kill_switch: KillSwitch,
    sleeve: Sleeve,
    sleeve_registry: SleeveRegistry,
    evaluate_signals: EvaluateSignals,
    reconcile: Reconcile,
    clock: FixedClock,
) -> AsyncIterator[httpx.AsyncClient]:
    client = _make_client(
        event_store=event_store,
        kill_switch=kill_switch,
        sleeve=sleeve,
        sleeve_registry=sleeve_registry,
        evaluate_signals=evaluate_signals,
        reconcile=reconcile,
        clock=clock,
        feed=None,  # None => disabled
        watchlist=["NVDA", "TSLA"],
    )
    async with client:
        yield client


_SIGNAL = {
    "source_type": "statement",
    "ticker": "AAPL",
    "sentiment": 0.9,
    "raw_payload": {"headline": "Senator praises Apple"},
}


async def test_quotes_enabled_returns_string_decimals(
    enabled_client: tuple[httpx.AsyncClient, StubQuoteFeed],
) -> None:
    client, _ = enabled_client
    response = await client.get("/quotes", params={"symbols": "AAPL,MSFT"})
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["as_of"] is not None
    quotes = {q["symbol"]: q for q in body["quotes"]}
    assert quotes["AAPL"]["last"] == "232.12"  # serialized as string
    assert isinstance(quotes["AAPL"]["bid"], str)
    assert quotes["MSFT"]["ask"] == "410.20"


async def test_quotes_falls_back_to_watchlist_when_no_symbols(
    enabled_client: tuple[httpx.AsyncClient, StubQuoteFeed],
) -> None:
    client, feed = enabled_client
    response = await client.get("/quotes")
    assert response.status_code == 200
    assert {q["symbol"] for q in response.json()["quotes"]} == {"AAPL", "MSFT"}
    assert feed.calls[-1] == ["AAPL", "MSFT"]  # used the default watchlist


async def test_quotes_disabled_returns_empty_200(disabled_client: httpx.AsyncClient) -> None:
    response = await disabled_client.get("/quotes", params={"symbols": "AAPL"})
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "as_of": None, "quotes": []}


async def test_tracked_falls_back_to_watchlist_when_no_positions(
    disabled_client: httpx.AsyncClient,
) -> None:
    response = await disabled_client.get("/quotes/tracked")
    assert response.status_code == 200
    assert response.json() == {"symbols": ["NVDA", "TSLA"]}


async def test_tracked_reflects_held_positions(
    enabled_client: tuple[httpx.AsyncClient, StubQuoteFeed],
) -> None:
    client, _ = enabled_client
    await client.post("/signals", json=[_SIGNAL])  # buys AAPL via the trivial strategy
    response = await client.get("/quotes/tracked")
    assert response.json() == {"symbols": ["AAPL"]}


async def test_positions_live_enabled_marks_with_feed(
    enabled_client: tuple[httpx.AsyncClient, StubQuoteFeed],
) -> None:
    client, _ = enabled_client
    await client.post("/signals", json=[_SIGNAL])
    response = await client.get("/positions/live")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    [position] = body["positions"]
    assert position["symbol"] == "AAPL"
    assert position["sleeve_id"] == "sleeve-test"
    assert position["last"] == "232.12"
    # quantity 5 @ 100 (StaticMarketData), marked at 232.12
    assert position["quantity"] == "5"
    assert position["avg_price"] == "100"
    assert position["market_value"] == "1160.60"
    assert position["unrealized_pnl"] == "660.60"


async def test_positions_live_disabled_marks_at_avg(
    disabled_client: httpx.AsyncClient,
) -> None:
    await disabled_client.post("/signals", json=[_SIGNAL])
    response = await disabled_client.get("/positions/live")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    [position] = body["positions"]
    assert position["last"] == "100"  # falls back to avg cost
    assert position["unrealized_pnl"] == "0"


async def test_positions_live_empty_when_no_positions(
    disabled_client: httpx.AsyncClient,
) -> None:
    response = await disabled_client.get("/positions/live")
    assert response.json() == {"enabled": False, "positions": []}
