"""Composition root: the only place where concrete adapters are wired to ports."""

from collections.abc import Awaitable, Callable
from datetime import timedelta
from decimal import Decimal
from functools import partial

import httpx
from fastapi import FastAPI

from trading.application.backtest import Backtest, BacktestRequest, BacktestResult
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.ingest_signals import IngestSignals
from trading.application.live_trading import AssessLiveReadiness, RequestLiveTrading
from trading.application.place_order import PlaceOrder
from trading.application.political_cycle import PoliticalCycle
from trading.application.ports.broker import BrokerPort
from trading.application.ports.data_source import DataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.market_data import QuoteFeedPort
from trading.application.projections import signals_from_events
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.application.sleeve_registry import SleeveRegistry, SleeveState
from trading.config import Settings
from trading.domain.conviction import ConvictionThresholds
from trading.domain.sectors import Sector
from trading.domain.sleeves import Sleeve, SleeveStatus
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory
from trading.infrastructure.brokers.alpaca import AlpacaBroker
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.brokers.routing import ModeRoutingBroker
from trading.infrastructure.clock import SystemClock
from trading.infrastructure.data_sources.congress_trades import CongressTradesSource
from trading.infrastructure.data_sources.news import NewsSource
from trading.infrastructure.data_sources.usaspending import UsaSpendingSource
from trading.infrastructure.market_data.alpaca import AlpacaMarketData
from trading.infrastructure.market_data.disabled import DisabledQuoteFeed
from trading.infrastructure.market_data.historical import synthetic_price_series
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver
from trading.infrastructure.nlp.sentiment import LexiconSentiment
from trading.infrastructure.persistence.broadcasting import BroadcastingEventStore
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.infrastructure.reference import COMPANY_DIRECTORY
from trading.infrastructure.reference.sectors import StaticSectorMap
from trading.interface.app import AppDeps, create_app
from trading.observability import configure_logging
from trading.strategies.base import Strategy
from trading.strategies.political_signal import PoliticalSignalStrategy
from trading.strategies.trivial import TrivialStrategy

# Development quotes; a real MarketDataPort adapter replaces this in a later milestone.
# Every ticker the entity resolver can produce needs a quote so the paper broker can fill.
_SECTOR_BASE_PRICE = {
    Sector.TECHNOLOGY: Decimal("200"),
    Sector.DEFENSE: Decimal("450"),
    Sector.HEALTHCARE: Decimal("120"),
    Sector.ENERGY: Decimal("110"),
    Sector.FINANCIALS: Decimal("180"),
    Sector.INDUSTRIALS: Decimal("190"),
    Sector.CONSUMER: Decimal("90"),
    Sector.COMMUNICATIONS: Decimal("150"),
    Sector.MATERIALS: Decimal("80"),
    Sector.UTILITIES: Decimal("70"),
    Sector.REAL_ESTATE: Decimal("100"),
}
_DEV_PRICES = {
    ticker: _SECTOR_BASE_PRICE[sector] for ticker, (sector, _aliases) in COMPANY_DIRECTORY.items()
}


def build_dependencies(settings: Settings) -> AppDeps:
    clock = SystemClock()

    inner_store: EventStorePort
    if settings.database_url:
        from sqlalchemy.ext.asyncio import create_async_engine

        from trading.infrastructure.persistence.postgres import PostgresEventStore

        inner_store = PostgresEventStore(create_async_engine(settings.database_url))
    else:
        inner_store = InMemoryEventStore(clock)
    # Wrap the store so appended events fan out to live WebSocket subscribers (FR-API-1).
    event_bus = BroadcastingEventStore(inner_store)
    event_store: EventStorePort = event_bus

    market_data = StaticMarketData(_DEV_PRICES, clock)
    kill_switch = KillSwitch()  # global: shared across all sleeves (FR-RISK-3)
    risk_limits = RiskLimits(
        max_trade_pct_of_sleeve=settings.max_trade_pct_of_sleeve,
        max_position_concentration_pct=settings.max_position_concentration_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
    )
    risk_gate = RiskGate(risk_limits, kill_switch)
    thresholds = ConvictionThresholds(
        trade=settings.trade_threshold, watch=settings.watch_threshold
    )

    political_strategy = PoliticalSignalStrategy(
        thresholds=thresholds, window_days=settings.corroboration_window_days
    )

    # A single live venue (one Alpaca account) is wired only when live is allowed and
    # credentials are present; otherwise it stays None and no sleeve can route to it.
    live_aclosers: list[Callable[[], Awaitable[None]]] = []
    live_broker: BrokerPort | None = None
    if settings.allow_live and settings.alpaca_live_key_id and settings.alpaca_live_secret_key:
        live_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        live_broker = AlpacaBroker(
            client=live_client,
            key_id=settings.alpaca_live_key_id,
            secret_key=settings.alpaca_live_secret_key,
            base_url=settings.alpaca_live_base_url,
        )
        live_aclosers.append(live_client.aclose)

    # Live quote feed (live stock tracking + live P&L). OFF by construction: built only when
    # enabled AND data credentials resolve (Alpaca data accepts paper or live keys); else the
    # disabled feed keeps `python -m trading.main` network-silent. Reused for /quotes,
    # /quotes/tracked and /positions/live.
    quote_feed: QuoteFeedPort = DisabledQuoteFeed()
    data_key_id = settings.data_key_id()
    data_secret_key = settings.data_secret_key()
    if settings.enable_live_quotes and data_key_id and data_secret_key:
        data_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        quote_feed = AlpacaMarketData(
            client=data_client,
            key_id=data_key_id,
            secret_key=data_secret_key,
            base_url=settings.alpaca_data_base_url,
        )
        live_aclosers.append(data_client.aclose)

    # Each sleeve is an isolated capital pool: its own paper broker, evaluator, and reconciler.
    # Adding a sleeve = one more spec here.
    sleeve_specs: list[tuple[str, str, Strategy]] = [
        ("sleeve-political", "political_signal", political_strategy),
        ("sleeve-trivial", "trivial", TrivialStrategy()),
    ]
    registry = SleeveRegistry(
        [
            SleeveState(
                sleeve=Sleeve(
                    id=sleeve_id,
                    strategy_id=strategy_id,
                    allocated_capital=settings.starting_cash,
                    mode=settings.mode_default,
                    status=SleeveStatus.RUNNING,
                ),
                mode=settings.mode_default,
                status=SleeveStatus.RUNNING,
            )
            for sleeve_id, strategy_id, _ in sleeve_specs
        ]
    )

    evaluators: dict[str, EvaluateSignals] = {}
    reconcilers_by_id: dict[str, Reconcile] = {}
    political_evaluator: EvaluateSignals | None = None
    for sleeve_id, _strategy_id, strategy in sleeve_specs:
        sleeve = registry.get(sleeve_id).sleeve
        paper_broker = PaperBroker(
            market_data=market_data, clock=clock, starting_cash=settings.starting_cash
        )
        # One broker handle per sleeve that routes to paper or live by current mode, so the
        # paper->live switch never touches strategy, risk, or execution code (spec §11).
        router = ModeRoutingBroker(
            paper=paper_broker,
            live=live_broker,
            mode=partial(registry.mode, sleeve_id),
        )
        evaluator = EvaluateSignals(
            strategy=strategy,
            sleeve=sleeve,
            risk_gate=risk_gate,
            place_order=PlaceOrder(broker=router, event_store=event_store, clock=clock),
            broker=router,
            market_data=market_data,
            event_store=event_store,
            clock=clock,
        )
        evaluators[sleeve_id] = evaluator
        reconcilers_by_id[sleeve_id] = Reconcile(
            broker=router, event_store=event_store, clock=clock, sleeve_id=sleeve_id
        )
        if sleeve_id == "sleeve-political":
            political_evaluator = evaluator
    reconcilers = list(reconcilers_by_id.values())

    live_ceilings = RiskLimits(
        max_trade_pct_of_sleeve=settings.live_max_trade_pct_of_sleeve,
        max_position_concentration_pct=settings.live_max_position_concentration_pct,
        max_drawdown_pct=settings.live_max_drawdown_pct,
    )
    assess_readiness = AssessLiveReadiness(
        allow_live=settings.allow_live,
        live_broker_configured=live_broker is not None,
        configured_limits=risk_limits,
        live_ceilings=live_ceilings,
        kill_switch=kill_switch,
        reconcilers=reconcilers_by_id,
        event_store=event_store,
        clock=clock,
    )
    request_live = RequestLiveTrading(
        registry=registry,
        readiness=assess_readiness,
        event_store=event_store,
        clock=clock,
    )

    sources, aclosers = _build_sources(settings, clock)
    political_cycle: PoliticalCycle | None = None
    if sources and political_evaluator is not None:
        lookback_start = clock.now() - timedelta(days=settings.corroboration_window_days)
        political_cycle = PoliticalCycle(
            ingest=IngestSignals(
                sources=sources,
                event_store=event_store,
                clock=clock,
                lookback_start=lookback_start,
            ),
            evaluate=political_evaluator,
            event_store=event_store,
            clock=clock,
            window_days=settings.corroboration_window_days,
        )

    # Backtesting (FR-BT): replay the recorded signals through the political strategy and the
    # same risk/execution path, in an isolated synthetic environment.
    backtest = Backtest(env_factory=InMemoryBacktestEnvFactory())

    async def run_backtest() -> BacktestResult:
        recorded = signals_from_events(await event_store.read_all())
        times = sorted({s.created_at for s in recorded})
        prices = synthetic_price_series(_DEV_PRICES, times)
        return await backtest.run(
            BacktestRequest(
                strategy=political_strategy,
                signals=recorded,
                risk_limits=risk_limits,
                prices=prices,
                starting_cash=settings.starting_cash,
                window_days=settings.corroboration_window_days,
            )
        )

    deps = AppDeps(
        event_store=event_store,
        kill_switch=kill_switch,
        sleeve_registry=registry,
        evaluators=evaluators,
        default_sleeve_id="sleeve-political",
        reconcilers=reconcilers,
        clock=clock,
        conviction_thresholds=thresholds,
        conviction_window_days=settings.corroboration_window_days,
        quote_feed=quote_feed,
        default_watchlist=settings.default_watchlist,
        political_cycle=political_cycle,
        backtest_runner=run_backtest,
        assess_readiness=assess_readiness,
        request_live=request_live,
        event_bus=event_bus,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
        ingest_interval_seconds=settings.ingest_interval_seconds,
    )
    deps.aclosers.extend(aclosers)
    deps.aclosers.extend(live_aclosers)
    return deps


def _build_sources(
    settings: Settings, clock: SystemClock
) -> tuple[list[DataSourcePort], list[Callable[[], Awaitable[None]]]]:
    """Construct only the enabled data sources, sharing one HTTP client (FR-ING-2/4)."""
    news_enabled = bool(settings.news_feed_urls)
    trades_enabled = bool(
        settings.disclosed_trade_leg_enabled
        and settings.congress_trades_endpoint
        and settings.fmp_api_key
    )
    if not (news_enabled or settings.usaspending_enabled or trades_enabled):
        return [], []

    client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
    entity_resolver = GazetteerEntityResolver()
    sentiment = LexiconSentiment()
    politicians = StaticPoliticianRepository()
    sector_map = StaticSectorMap()

    sources: list[DataSourcePort] = []
    if news_enabled:
        sources.append(
            NewsSource(
                client=client,
                feed_urls=settings.news_feed_urls,
                entity_resolver=entity_resolver,
                sentiment=sentiment,
                politicians=politicians,
                sector_map=sector_map,
                clock=clock,
            )
        )
    if settings.usaspending_enabled:
        sources.append(
            UsaSpendingSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                lookback_days=settings.usaspending_lookback_days,
            )
        )
    if trades_enabled:
        assert settings.congress_trades_endpoint is not None
        assert settings.fmp_api_key is not None
        sources.append(
            CongressTradesSource(
                client=client,
                endpoint=settings.congress_trades_endpoint,
                api_key=settings.fmp_api_key,
                politicians=politicians,
                sector_map=sector_map,
                clock=clock,
            )
        )
    return sources, [client.aclose]


def create_application() -> FastAPI:
    settings = Settings()
    configure_logging(settings.log_level)
    return create_app(build_dependencies(settings))


def main() -> None:
    import uvicorn

    uvicorn.run(create_application(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
