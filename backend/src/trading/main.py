"""Composition root: the only place where concrete adapters are wired to ports."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from decimal import Decimal
from functools import partial

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from trading.application.apply_trade_update import ApplyTradeUpdate
from trading.application.backtest import Backtest, BacktestRequest, BacktestResult
from trading.application.cancel_order import CancelOrder
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.ingest_signals import IngestSignals
from trading.application.live_trading import AssessLiveReadiness, RequestLiveTrading
from trading.application.place_order import PlaceOrder
from trading.application.political_cycle import PoliticalCycle
from trading.application.ports.broker import BrokerPort
from trading.application.ports.data_source import DataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.historical import HistoricalPricesPort
from trading.application.ports.market_data import MarketDataPort, QuoteFeedPort, QuoteStreamPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.read_model import ReadModelPort
from trading.application.ports.trade_updates import TradeUpdatesStreamPort
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.application.sleeve_registry import SleeveRegistry, SleeveState
from trading.application.sync_orders import SyncOrders
from trading.config import Settings
from trading.domain.conviction import ConvictionThresholds
from trading.domain.sectors import Sector
from trading.domain.sleeves import Sleeve, SleeveStatus
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory
from trading.infrastructure.brokers.alpaca import AlpacaBroker
from trading.infrastructure.brokers.disabled_trade_updates import DisabledTradeUpdatesStream
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.brokers.routing import ModeRoutingBroker
from trading.infrastructure.brokers.trade_updates_stream import AlpacaTradeUpdatesStream
from trading.infrastructure.clock import SystemClock
from trading.infrastructure.data_sources.congress_trades import CongressTradesSource
from trading.infrastructure.data_sources.news import NewsSource
from trading.infrastructure.data_sources.usaspending import UsaSpendingSource
from trading.infrastructure.market_data.alpaca import AlpacaMarketData
from trading.infrastructure.market_data.alpaca_history import AlpacaHistoricalData
from trading.infrastructure.market_data.alpaca_stream import AlpacaQuoteStream
from trading.infrastructure.market_data.disabled import DisabledQuoteFeed, DisabledQuoteStream
from trading.infrastructure.market_data.feed_backed import FeedBackedMarketData
from trading.infrastructure.market_data.historical import SyntheticHistoricalPrices
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver
from trading.infrastructure.nlp.sentiment import LexiconSentiment
from trading.infrastructure.observability.counting_event_store import CountingEventStore
from trading.infrastructure.observability.metrics import MetricsRegistry
from trading.infrastructure.persistence.broadcasting import BroadcastingEventStore
from trading.infrastructure.persistence.health import DatabaseReadiness
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.infrastructure.politicians.congress_legislators import (
    CongressLegislatorsRepository,
    fetch_and_parse_roster,
)
from trading.infrastructure.politicians.repository import StaticPoliticianRepository
from trading.infrastructure.reference import COMPANY_DIRECTORY
from trading.infrastructure.reference.sectors import StaticSectorMap
from trading.interface.app import AppDeps, create_app
from trading.observability import configure_logging
from trading.strategies import registry as strategy_registry
from trading.strategies.base import Strategy, StrategyConfig

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

    # Observability registry: backs /metrics and is bumped on each append (events/orders).
    metrics = MetricsRegistry()

    # Event store (system of record) + read model (CQRS read side). The read model is kept
    # current on every append so hot reads query maintained projections, not a full-log replay.
    inner_store: EventStorePort
    read_model: ReadModelPort
    db_engine: AsyncEngine | None = None
    if settings.database_url:
        from sqlalchemy.ext.asyncio import create_async_engine

        from trading.infrastructure.persistence.postgres import PostgresEventStore
        from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

        db_engine = create_async_engine(settings.database_url)
        pg_read_model = PostgresReadModel(db_engine)
        # Read-model rows are written in the append transaction, atomically with the event.
        inner_store = PostgresEventStore(db_engine, on_append=pg_read_model.apply)
        read_model = pg_read_model
    else:
        # Default path stays fully in-memory: a projecting wrapper folds each appended event
        # into the in-memory read model synchronously (read-after-write consistency).
        memory_read_model = InMemoryReadModel()
        inner_store = ProjectingEventStore(InMemoryEventStore(clock), memory_read_model)
        read_model = memory_read_model
    # /readyz probes DB connectivity when a database is configured; trivially ready otherwise.
    db_readiness = DatabaseReadiness(db_engine)
    # Wrap the store so appended events fan out to live WebSocket subscribers (FR-API-1)...
    event_bus = BroadcastingEventStore(inner_store)
    # ...then count appends for observability (events_appended_total / orders_placed_total).
    event_store: EventStorePort = CountingEventStore(event_bus, metrics)

    # Decision/fill price seam: static dev prices by default, swapped to the live feed below
    # when it is enabled so the strategy decides and the paper broker fills at real prices.
    market_data: MarketDataPort = StaticMarketData(_DEV_PRICES, clock)
    kill_switch = KillSwitch()  # global: shared across all sleeves (FR-RISK-3)
    risk_limits = RiskLimits(
        max_trade_pct_of_sleeve=settings.max_trade_pct_of_sleeve,
        max_position_concentration_pct=settings.max_position_concentration_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
    )
    risk_gate = RiskGate(risk_limits, kill_switch)
    # Translate global Settings into the strategies layer's own typed config so every strategy
    # is built through the registry (spec section 18: add a strategy = one module + register).
    # `thresholds` is also handed to the API layer (deps below) for the /conviction endpoint.
    thresholds = ConvictionThresholds(
        trade=settings.trade_threshold, watch=settings.watch_threshold
    )
    strategy_config = StrategyConfig(
        thresholds=thresholds,
        window_days=settings.corroboration_window_days,
        decay_half_life_days=settings.conviction_decay_half_life_days,
        corroboration_bonus_per_source=settings.conviction_corroboration_bonus_per_source,
        min_sources_for_full_corroboration=settings.conviction_min_sources_for_full_corroboration,
        max_corroboration_factor=settings.conviction_max_corroboration_factor,
        conviction_scale=settings.conviction_scale,
    )

    political_strategy = strategy_registry.create("political_signal", strategy_config)

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

    # Optional Alpaca PAPER venue (real paper account, no real money). One Alpaca account is one
    # book, so it backs exactly one sleeve (`alpaca_paper_sleeve_id`); that sleeve's execution
    # and reconciliation then run against the authoritative external broker (FR-REC). Built only
    # when enabled AND paper/data credentials resolve; otherwise the in-process simulator is used
    # and no network calls are made. Constructing AlpacaBroker dials nothing.
    alpaca_paper_broker: BrokerPort | None = None
    paper_key_id = settings.paper_key_id()
    paper_secret_key = settings.paper_secret_key()
    if settings.enable_alpaca_paper and paper_key_id and paper_secret_key:
        paper_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        alpaca_paper_broker = AlpacaBroker(
            client=paper_client,
            key_id=paper_key_id,
            secret_key=paper_secret_key,
            base_url=settings.alpaca_paper_base_url,
        )
        live_aclosers.append(paper_client.aclose)

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
        # Route the decision/fill price seam through the live feed (real prices), keeping the
        # static dev prices as the fallback for symbols the feed cannot price.
        market_data = FeedBackedMarketData(quote_feed, market_data)

    # Real-time quote stream behind /ws/quotes. Same switch as the REST feed: built only when
    # enabled AND data creds resolve; otherwise the disabled stream opens no socket. Seeded with
    # the default watchlist so it streams something out of the box; clients extend the upstream
    # subscription via {"action":"subscribe",...} and each connection also adds tracked symbols.
    # Constructing AlpacaQuoteStream opens NOTHING — its run() task is started on app startup.
    quote_stream: QuoteStreamPort = DisabledQuoteStream()
    background_tasks: list[Callable[[], Awaitable[None]]] = []
    if settings.enable_live_quotes and data_key_id and data_secret_key:
        alpaca_stream = AlpacaQuoteStream(
            key_id=data_key_id,
            secret_key=data_secret_key,
            symbols=settings.default_watchlist,
            url=settings.alpaca_data_stream_url,
        )
        quote_stream = alpaca_stream
        background_tasks.append(alpaca_stream.run)

    # Historical price source for backtests. Default OFF -> synthetic (offline, no network);
    # real Alpaca daily bars are wired only when enabled AND the same data credentials resolve.
    historical_prices: HistoricalPricesPort = SyntheticHistoricalPrices(_DEV_PRICES)
    if settings.enable_historical_data and data_key_id and data_secret_key:
        history_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        historical_prices = AlpacaHistoricalData(
            client=history_client,
            key_id=data_key_id,
            secret_key=data_secret_key,
            base_url=settings.alpaca_data_base_url,
        )
        live_aclosers.append(history_client.aclose)

    # Each sleeve is an isolated capital pool: its own paper broker, evaluator, and reconciler.
    # Adding a sleeve = one more spec here.
    sleeve_specs: list[tuple[str, str, Strategy]] = [
        ("sleeve-political", "political_signal", political_strategy),
        ("sleeve-trivial", "trivial", strategy_registry.create("trivial", strategy_config)),
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

    # --- Anti-placeholder data-integrity gate (no fake/placeholder data in real trades) -------
    # Engages automatically whenever a REAL Alpaca account is wired (paper or live), or on an
    # explicit REQUIRE_REAL_DATA opt-in. When on: synthetic signals are dropped at ingestion,
    # a real-broker evaluation aborts on any non-REAL signal, and — here — startup is refused if a
    # real broker would trade/mark against synthetic prices. Default off keeps dev/tests unchanged.
    require_real_data = (
        settings.require_real_data or settings.enable_alpaca_paper or settings.allow_live
    )
    real_market_data = bool(
        settings.enable_live_quotes and settings.data_key_id() and settings.data_secret_key()
    )
    if require_real_data and not real_market_data:
        raise RuntimeError(
            "Data integrity: a real broker is wired (or REQUIRE_REAL_DATA is set) but live market "
            "data is unavailable (need ENABLE_LIVE_QUOTES=true with resolvable Alpaca data "
            "credentials). A real account must never trade or mark against synthetic prices — "
            "enable the live feed, or disable the real broker / REQUIRE_REAL_DATA."
        )

    evaluators: dict[str, EvaluateSignals] = {}
    reconcilers_by_id: dict[str, Reconcile] = {}
    order_syncs_by_id: dict[str, SyncOrders] = {}  # per-sleeve: ingest async broker fills
    brokers_by_id: dict[str, BrokerPort] = {}  # per-sleeve routing broker, for manual cancel
    political_evaluator: EvaluateSignals | None = None
    for sleeve_id, _strategy_id, strategy in sleeve_specs:
        sleeve = registry.get(sleeve_id).sleeve
        # The designated sleeve executes paper orders on the real Alpaca paper account when
        # configured; every other sleeve uses the in-process simulator (isolated capital).
        paper_venue: BrokerPort
        if alpaca_paper_broker is not None and sleeve_id == settings.alpaca_paper_sleeve_id:
            paper_venue = alpaca_paper_broker
        else:
            paper_venue = PaperBroker(
                market_data=market_data, clock=clock, starting_cash=settings.starting_cash
            )
        # One broker handle per sleeve that routes to paper or live by current mode, so the
        # paper->live switch never touches strategy, risk, or execution code (spec §11).
        router = ModeRoutingBroker(
            paper=paper_venue,
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
            require_real_data=require_real_data,
        )
        evaluators[sleeve_id] = evaluator
        brokers_by_id[sleeve_id] = router
        reconcilers_by_id[sleeve_id] = Reconcile(
            broker=router, event_store=event_store, clock=clock, sleeve_id=sleeve_id
        )
        order_syncs_by_id[sleeve_id] = SyncOrders(
            broker=router, event_store=event_store, clock=clock, sleeve_id=sleeve_id
        )
        if sleeve_id == "sleeve-political":
            political_evaluator = evaluator
    reconcilers = list(reconcilers_by_id.values())
    order_syncs = list(order_syncs_by_id.values())

    # Real-time trade-updates stream (FR-REC/FR-EXE): the push complement to the 30s polling
    # order-sync. OFF by construction — built only when the stream is enabled AND the Alpaca
    # PAPER venue is actually wired for its sleeve AND paper creds resolve; otherwise the
    # disabled stream opens no socket. Constructing AlpacaTradeUpdatesStream dials NOTHING; its
    # run() task is started on app startup. It records the SAME ORDER_FILLED/STATUS_CHANGED
    # events the poller emits, idempotently, for the paper sleeve.
    trade_updates_stream: TradeUpdatesStreamPort = DisabledTradeUpdatesStream()
    if (
        settings.enable_trade_updates_stream
        and alpaca_paper_broker is not None
        and paper_key_id
        and paper_secret_key
    ):
        paper_sleeve_id = settings.alpaca_paper_sleeve_id
        apply_update = ApplyTradeUpdate(
            event_store=event_store, clock=clock, sleeve_id=paper_sleeve_id
        )
        updates_stream = AlpacaTradeUpdatesStream(
            key_id=paper_key_id,
            secret_key=paper_secret_key,
            url=settings.trade_updates_stream_url(),
            sleeve_id=paper_sleeve_id,
        )
        trade_updates_stream = updates_stream

        async def _run_trade_updates() -> None:
            await updates_stream.run(apply_update)

        background_tasks.append(_run_trade_updates)

    cancel_order = CancelOrder(
        brokers=brokers_by_id,
        event_store=event_store,
        clock=clock,
        default_sleeve_id="sleeve-political",
    )

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
        window_days=settings.corroboration_window_days,
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
                require_real_data=require_real_data,
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
        recorded = list(await read_model.all_signals())
        symbols = sorted({s.ticker for s in recorded})
        times = sorted({s.created_at for s in recorded})
        # Date range: bound by the recorded signal timestamps, widened to at least
        # backtest_lookback_days so real bars span enough history to mark every signal.
        end = times[-1] if times else clock.now()
        lookback = timedelta(days=settings.backtest_lookback_days)
        start = min(times[0], end - lookback) if times else end - lookback
        prices = await historical_prices.price_series(symbols, start, end)
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
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve_registry=registry,
        evaluators=evaluators,
        default_sleeve_id="sleeve-political",
        reconcilers=reconcilers,
        order_syncs=order_syncs,
        clock=clock,
        conviction_thresholds=thresholds,
        conviction_window_days=settings.corroboration_window_days,
        quote_feed=quote_feed,
        quote_stream=quote_stream,
        trade_updates_stream=trade_updates_stream,
        background_tasks=background_tasks,
        default_watchlist=settings.default_watchlist,
        political_cycle=political_cycle,
        backtest_runner=run_backtest,
        assess_readiness=assess_readiness,
        request_live=request_live,
        cancel_order=cancel_order,
        event_bus=event_bus,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
        ingest_interval_seconds=settings.ingest_interval_seconds,
        order_sync_interval_seconds=settings.order_sync_interval_seconds,
        metrics=metrics,
        enable_metrics=settings.enable_metrics,
        db_readiness=db_readiness,
        api_key=settings.api_key,
    )
    deps.aclosers.extend(aclosers)
    deps.aclosers.extend(live_aclosers)
    # Release the Postgres connection pool on shutdown (FastAPI lifespan) and on a one-shot
    # job teardown (trading.jobs.cycle). `AsyncEngine.dispose` is awaitable, matching the
    # aclosers signature, so both teardown paths close the pool with no special-casing.
    if db_engine is not None:
        deps.aclosers.append(db_engine.dispose)
    return deps


async def _load_congress_roster(settings: Settings) -> CongressLegislatorsRepository:
    """Fetch + parse the REAL congressional roster once at startup (FR-POL-3, anti-placeholder).

    Uses a short-lived client so the fetch is self-contained. The parser stamps every member
    Provenance.REAL, so the news/disclosed-trade signals they back can clear the data-integrity
    guard. Only invoked when ``enable_congress_roster`` is set and a politician-using source is on.
    """
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        return CongressLegislatorsRepository(await fetch_and_parse_roster(client))


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
    politicians: PoliticianRepositoryPort
    if settings.enable_congress_roster and (news_enabled or trades_enabled):
        # Opt-in REAL roster (fetched once at startup; sync composition -> asyncio.run). With it,
        # news/disclosed-trade signals attach to REAL members and can pass the guard; without it,
        # the synthetic seed roster yields SYNTHETIC, guard-blocked signals.
        politicians = asyncio.run(_load_congress_roster(settings))
    else:
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
