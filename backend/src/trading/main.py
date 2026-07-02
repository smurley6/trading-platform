"""Composition root: the only place where concrete adapters are wired to ports."""

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
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
from trading.application.outcome_labels import LabelOutcomes
from trading.application.place_order import PlaceOrder
from trading.application.political_cycle import PoliticalCycle
from trading.application.ports.backfill_seen import BackfillSeenPort
from trading.application.ports.broker import BrokerPort
from trading.application.ports.data_source import DataSourcePort, HistoricalDataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.historical import HistoricalPricesPort
from trading.application.ports.market_data import MarketDataPort, QuoteFeedPort, QuoteStreamPort
from trading.application.ports.outcome_labels import OutcomeLabelsPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.read_model import ReadModelPort
from trading.application.ports.trade_updates import TradeUpdate, TradeUpdatesStreamPort
from trading.application.proposals import ExecuteProposals, ScanProposals
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
from trading.infrastructure.data_sources.clinical_trials import ClinicalTrialsSource
from trading.infrastructure.data_sources.congress_bills import CongressBillsSource
from trading.infrastructure.data_sources.congress_trades import CongressTradesSource
from trading.infrastructure.data_sources.fda_approvals import FdaApprovalsSource
from trading.infrastructure.data_sources.federal_register import FederalRegisterSource
from trading.infrastructure.data_sources.house_disclosures import HouseDisclosuresSource
from trading.infrastructure.data_sources.lobbying import LobbyingSource
from trading.infrastructure.data_sources.news import NewsSource
from trading.infrastructure.data_sources.regulations_gov import RegulationsGovSource
from trading.infrastructure.data_sources.sec_edgar_atom import SecEdgarAtomSource
from trading.infrastructure.data_sources.sec_form4 import SecForm4Source
from trading.infrastructure.data_sources.senate_efd import SenateEfdSource
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
from trading.infrastructure.persistence.memory_backfill_seen import InMemoryBackfillSeen
from trading.infrastructure.persistence.memory_outcome_labels import InMemoryOutcomeLabels
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
    backfill_seen: BackfillSeenPort
    outcome_labels: OutcomeLabelsPort
    db_engine: AsyncEngine | None = None
    if settings.database_url:
        from sqlalchemy.ext.asyncio import create_async_engine

        from trading.infrastructure.persistence.postgres import PostgresEventStore
        from trading.infrastructure.persistence.postgres_backfill_seen import PostgresBackfillSeen
        from trading.infrastructure.persistence.postgres_outcome_labels import (
            PostgresOutcomeLabels,
        )
        from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

        # pool_pre_ping issues a lightweight liveness check (SELECT 1) before handing out a
        # pooled connection, transparently discarding ones the server has since dropped. Neon
        # scales compute to zero when idle, which silently kills pooled connections; without
        # pre-ping the next read on a long-running server hits a stale socket and 500s with
        # ConnectionResetError. The short-lived cron job rarely sees this, but the always-on
        # API does, so pre-ping is the correct fix for the served app (cost: one cheap round-trip
        # per checkout after an idle gap).
        db_engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        pg_read_model = PostgresReadModel(db_engine)
        # Read-model rows are written in the append transaction, atomically with the event.
        inner_store = PostgresEventStore(db_engine, on_append=pg_read_model.apply)
        read_model = pg_read_model
        # Backfill dedupe lives in its own table (not the read model); Postgres-backed here.
        backfill_seen = PostgresBackfillSeen(db_engine)
        # Outcome labels: a rebuildable RESEARCH projection (Task 03), written by the labeler run,
        # NOT folded from events on append — its own table, Postgres-backed here.
        outcome_labels = PostgresOutcomeLabels(db_engine)
    else:
        # Default path stays fully in-memory: a projecting wrapper folds each appended event
        # into the in-memory read model synchronously (read-after-write consistency).
        memory_read_model = InMemoryReadModel()
        inner_store = ProjectingEventStore(InMemoryEventStore(clock), memory_read_model)
        read_model = memory_read_model
        backfill_seen = InMemoryBackfillSeen()
        outcome_labels = InMemoryOutcomeLabels()
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
    # book = the ONE shared capital pool; when wired it backs EVERY sleeve (see the shared_paper
    # routing below), and pool-level reconciliation runs against this authoritative external broker
    # (FR-REC). Built only when enabled AND paper/data credentials resolve; otherwise the in-process
    # simulator is used and no network calls are made. Constructing AlpacaBroker dials nothing.
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

    # Historical price source for backtests AND the outcome labeler. Default OFF -> synthetic
    # (offline, no network); real Alpaca daily bars are wired only when enabled AND the same data
    # credentials resolve. `real_historical_prices` tells the labeler job whether the REAL adapter
    # is wired (its --confirm guard refuses to store labels computed from the synthetic provider).
    historical_prices: HistoricalPricesPort = SyntheticHistoricalPrices(_DEV_PRICES)
    real_historical_prices = False
    if settings.enable_historical_data and data_key_id and data_secret_key:
        history_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        historical_prices = AlpacaHistoricalData(
            client=history_client,
            key_id=data_key_id,
            secret_key=data_secret_key,
            base_url=settings.alpaca_data_base_url,
        )
        live_aclosers.append(history_client.aclose)
        real_historical_prices = True

    # Sleeves are algos that share ONE capital pool (the single Alpaca book), not isolated pools:
    # each has its own strategy + evaluator, but they trade the same shared broker and every
    # order/fill is attributed by the sleeve_id stamped on its events. `sleeve-testing` is an inert
    # drop-in slot for a second algorithm (see strategies/testing.py). Adding an algo = one spec.
    sleeve_specs: list[tuple[str, str, Strategy]] = [
        ("sleeve-political", "political_signal", political_strategy),
        ("sleeve-testing", "testing", strategy_registry.create("testing", strategy_config)),
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

    # ONE shared execution book = the single capital pool. When the real Alpaca paper account is
    # wired it backs EVERY sleeve, so both algos trade one book and are attributed by the sleeve_id
    # already stamped on every order/fill event. Without it, one in-process simulator stands in
    # (still one shared book) so dev/offline runs unchanged. This is the shared-pool switch.
    shared_paper: BrokerPort = (
        alpaca_paper_broker
        if alpaca_paper_broker is not None
        else PaperBroker(market_data=market_data, clock=clock, starting_cash=settings.starting_cash)
    )

    evaluators: dict[str, EvaluateSignals] = {}
    order_syncs_by_id: dict[str, SyncOrders] = {}  # per-sleeve: ingest async broker fills
    brokers_by_id: dict[str, BrokerPort] = {}  # per-sleeve routing broker, for manual cancel
    political_evaluator: EvaluateSignals | None = None
    for sleeve_id, _strategy_id, strategy in sleeve_specs:
        sleeve = registry.get(sleeve_id).sleeve
        # One broker handle per sleeve that routes to the SHARED paper venue (or live) by the
        # sleeve's current mode, so the paper->live switch never touches strategy/risk/execution
        # code (spec §11). All sleeves' handles wrap the same underlying book.
        router = ModeRoutingBroker(
            paper=shared_paper,
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
        # Order-sync stays per-sleeve: each SyncOrders acts ONLY on client_order_ids in its own
        # log, so polling the shared book never cross-imports another algo's fills (FR-REC).
        order_syncs_by_id[sleeve_id] = SyncOrders(
            broker=router, event_store=event_store, clock=clock, sleeve_id=sleeve_id
        )
        if sleeve_id == "sleeve-political":
            political_evaluator = evaluator

    # Reconciliation is POOL-LEVEL. The shared book is one account, so the local projection is
    # summed across ALL sleeves (sleeve_id=None) and diffed against that one book — a per-sleeve
    # diff would flag every OTHER algo's holdings as spurious drift. One reconciler serves the
    # /reconcile endpoint + scheduler, and is keyed to every sleeve so the per-sleeve live-readiness
    # preflight reuses the same authoritative account check.
    pool_reconcile = Reconcile(
        broker=shared_paper, event_store=event_store, clock=clock, sleeve_id=None
    )
    reconcilers_by_id: dict[str, Reconcile] = {
        sleeve_id: pool_reconcile for sleeve_id, _s, _st in sleeve_specs
    }
    reconcilers = [pool_reconcile]
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
        # Shared book => fan each update to a per-sleeve applier. Each self-scopes to the orders in
        # its own log and tags the fill with its own sleeve_id, so exactly the algo that placed an
        # order records its fill — correct the instant a real second algo is dropped in.
        appliers = [
            ApplyTradeUpdate(event_store=event_store, clock=clock, sleeve_id=sleeve_id)
            for sleeve_id, _s, _st in sleeve_specs
        ]

        async def _apply_to_all(update: TradeUpdate, *, sleeve_id: str | None = None) -> None:
            for applier in appliers:
                await applier.execute(update)

        updates_stream = AlpacaTradeUpdatesStream(
            key_id=paper_key_id,
            secret_key=paper_secret_key,
            url=settings.trade_updates_stream_url(),
            sleeve_id=sleeve_specs[0][0],  # nominal; each applier self-scopes by client_order_id
        )
        trade_updates_stream = updates_stream

        async def _run_trade_updates() -> None:
            await updates_stream.run(_apply_to_all)

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

    # Backfill (Task 02+04a): the registry of HISTORICAL sources (those implementing
    # fetch_historical). Task 04a registers sec_form4 here; house_disclosures (04b) is not yet
    # wired. The backfill job fails cleanly on an unknown/unsupported --source, listing whatever
    # is registered. Keyed by source name.
    historical_sources, historical_aclosers = _build_historical_sources(settings, clock)
    aclosers = [*aclosers, *historical_aclosers]

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
            trade_mode=settings.cycle_trade_mode,
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

    # Proposals (FR-EXE): a read-only conviction scan (TRADE -> proposals, WATCH -> predicted)
    # and a manual execute path that reuses the real evaluate -> risk -> place pipeline.
    scan_proposals = ScanProposals(
        strategy=political_strategy,
        read_model=read_model,
        clock=clock,
        thresholds=thresholds,
        conviction_params=strategy_config.conviction_params(),
        window_days=settings.corroboration_window_days,
        top_n=settings.proposal_top_n,
    )
    execute_proposals = ExecuteProposals(
        evaluators=evaluators,
        default_sleeve_id="sleeve-political",
        read_model=read_model,
        clock=clock,
        window_days=settings.corroboration_window_days,
    )

    # Outcome labeler (Task 03): computes forward event-study abnormal returns for ticker-resolved
    # REAL+BACKFILLED signals and writes them to the rebuildable outcome_labels projection (never
    # the event log). Consumes the SAME historical-prices port the backtest uses (daily bars) and
    # the static sector map for the sector-ETF benchmark. Driven only by `python -m
    # trading.jobs.label_outcomes` (no HTTP route); the job's --confirm guard uses
    # `real_historical_prices` to refuse storing labels computed from the synthetic price provider.
    label_outcomes = LabelOutcomes(
        read_model=read_model,
        prices=historical_prices,
        sector_map=StaticSectorMap(),
        labels_store=outcome_labels,
        event_store=event_store,
        clock=clock,
    )

    deps = AppDeps(
        event_store=event_store,
        read_model=read_model,
        kill_switch=kill_switch,
        sleeve_registry=registry,
        # The shared capital pool reads live buying power/equity from the one Alpaca book; None
        # when no real account is wired (the /capital endpoint then reports disabled).
        pool_broker=alpaca_paper_broker,
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
        cycle_trade_mode=settings.cycle_trade_mode,
        cors_allow_origins=[
            o.strip() for o in (settings.cors_allow_origins or "").split(",") if o.strip()
        ],
        scan_proposals=scan_proposals,
        execute_proposals=execute_proposals,
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
        backfill_seen=backfill_seen,
        historical_sources=historical_sources,
        outcome_labels=outcome_labels,
        label_outcomes=label_outcomes,
        real_historical_prices=real_historical_prices,
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


def _load_roster_blocking(settings: Settings) -> CongressLegislatorsRepository:
    """Drive the async roster fetch from the synchronous composition root, in any context.

    ``build_dependencies`` is synchronous but may run either with no event loop (the FastAPI
    entrypoint / tests) or from *inside* one — the cron job builds deps within
    ``asyncio.run(run_cycle())``. ``asyncio.run`` cannot nest in a running loop, so when one is
    already running we fetch on a dedicated worker thread with its own loop. The fetch is
    self-contained (its own short-lived client) and returns plain in-memory data, so no async
    object crosses the thread/loop boundary; blocking on the result here is fine because the
    composition root hasn't scheduled any other loop work yet.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_load_congress_roster(settings))
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_load_congress_roster(settings))).result()


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
    house_enabled = settings.house_disclosures_enabled
    senate_enabled = settings.senate_efd_enabled
    sec_enabled = bool(settings.sec_form4_enabled and settings.sec_form4_user_agent)
    lobbying_enabled = settings.lobbying_enabled
    fda_enabled = settings.fda_approvals_enabled
    clinical_trials_enabled = settings.clinical_trials_enabled
    federal_register_enabled = settings.federal_register_enabled
    congress_bills_enabled = bool(
        settings.congress_bills_enabled and settings.congress_bills_api_key
    )
    regulations_gov_enabled = bool(
        settings.regulations_gov_enabled and settings.regulations_gov_api_key
    )
    sec_edgar_atom_enabled = bool(settings.sec_edgar_atom_enabled and settings.sec_form4_user_agent)
    # House/Senate PTRs attach to the roster, so they count as politician-using sources too.
    politician_source_enabled = news_enabled or trades_enabled or house_enabled or senate_enabled
    if not (
        news_enabled
        or settings.usaspending_enabled
        or trades_enabled
        or house_enabled
        or senate_enabled
        or sec_enabled
        or lobbying_enabled
        or fda_enabled
        or clinical_trials_enabled
        or federal_register_enabled
        or congress_bills_enabled
        or regulations_gov_enabled
        or sec_edgar_atom_enabled
    ):
        return [], []

    client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
    entity_resolver = GazetteerEntityResolver()
    sentiment = LexiconSentiment()
    politicians: PoliticianRepositoryPort
    if settings.enable_congress_roster and politician_source_enabled:
        # Opt-in REAL roster (fetched once at startup; sync composition -> asyncio.run). With it,
        # news/disclosed-trade signals attach to REAL members and can pass the guard; without it,
        # the synthetic seed roster yields SYNTHETIC, guard-blocked signals.
        politicians = _load_roster_blocking(settings)
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
    if house_enabled:
        sources.append(
            HouseDisclosuresSource(
                client=client,
                politicians=politicians,
                sector_map=sector_map,
                clock=clock,
                lookback_days=settings.house_disclosures_lookback_days,
                max_filings=settings.house_disclosures_max_filings,
            )
        )
    if senate_enabled:
        sources.append(
            SenateEfdSource(
                client=client,
                politicians=politicians,
                sector_map=sector_map,
                clock=clock,
                lookback_days=settings.senate_efd_lookback_days,
            )
        )
    if sec_enabled:
        assert settings.sec_form4_user_agent is not None
        sources.append(
            SecForm4Source(
                client=client,
                clock=clock,
                user_agent=settings.sec_form4_user_agent,
                lookback_days=settings.sec_form4_lookback_days,
            )
        )
    if lobbying_enabled:
        sources.append(
            LobbyingSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                lookback_days=settings.lobbying_lookback_days,
                api_key=settings.lda_api_key,
            )
        )
    if fda_enabled:
        sources.append(
            FdaApprovalsSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                lookback_days=settings.fda_lookback_days,
                api_key=settings.openfda_api_key,
            )
        )
    if clinical_trials_enabled:
        sources.append(
            ClinicalTrialsSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                enabled=True,
                lookback_days=settings.clinical_trials_lookback_days,
            )
        )
    if federal_register_enabled:
        sources.append(
            FederalRegisterSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                sentiment=sentiment,
                enabled=True,
                lookback_days=settings.federal_register_lookback_days,
                api_key=settings.federal_register_api_key,
            )
        )
    if congress_bills_enabled:
        sources.append(
            CongressBillsSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                api_key=settings.congress_bills_api_key,
                enabled=True,
                lookback_days=settings.congress_bills_lookback_days,
            )
        )
    if regulations_gov_enabled:
        sources.append(
            RegulationsGovSource(
                client=client,
                entity_resolver=entity_resolver,
                clock=clock,
                api_key=settings.regulations_gov_api_key,
                lookback_days=settings.regulations_gov_lookback_days,
            )
        )
    if sec_edgar_atom_enabled:
        assert settings.sec_form4_user_agent is not None
        sources.append(
            SecEdgarAtomSource(
                client=client,
                entity_resolver=entity_resolver,
                sentiment=sentiment,
                user_agent=settings.sec_form4_user_agent,
                max_filings=settings.sec_edgar_atom_max_filings,
            )
        )
    return sources, [client.aclose]


def _build_historical_sources(
    settings: Settings, clock: SystemClock
) -> tuple[dict[str, HistoricalDataSourcePort], list[Callable[[], Awaitable[None]]]]:
    """Construct the registry of backfill-capable (historical) sources, keyed by name.

    Registered independently of the LIVE `*_enabled` flags — backfill is a research op, gated
    only by the source having what it needs to make requests at all (e.g. sec_form4 needs a
    declared User-Agent per SEC fair-access policy, same field the live adapter uses, but not
    `sec_form4_enabled`). house_disclosures (Task 04b) is not yet wired here.

    Returns its own aclosers (e.g. the shared httpx client) rather than reusing `_build_sources`'
    client, since a historical-only run (no live source enabled) would otherwise get no client.
    """
    aclosers: list[Callable[[], Awaitable[None]]] = []
    sources: dict[str, HistoricalDataSourcePort] = {}

    if settings.sec_form4_user_agent:
        client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        aclosers.append(client.aclose)
        form4 = SecForm4Source(
            client=client,
            clock=clock,
            user_agent=settings.sec_form4_user_agent,
            lookback_days=settings.sec_form4_lookback_days,
        )
        sources[form4.name] = form4

    return sources, aclosers


def create_application() -> FastAPI:
    settings = Settings()
    configure_logging(settings.log_level)
    return create_app(build_dependencies(settings))


def main() -> None:
    import uvicorn

    # loop="asyncio" (NOT uvloop): asyncpg's TLS handshake to Neon fails under uvloop with
    # "ConnectionResetError: [Errno 54]" inside uvloop's start_tls, while the stdlib asyncio
    # loop connects cleanly (the scheduled cron job, which runs under asyncio.run, talks to
    # Neon without issue). The marginal uvloop throughput win is irrelevant for this low-rate,
    # I/O-bound service, so we prefer the loop that actually connects.
    uvicorn.run(create_application(), host="127.0.0.1", port=8000, loop="asyncio")


if __name__ == "__main__":
    main()
