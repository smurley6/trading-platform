"""FastAPI application. Receives fully wired dependencies from the composition root."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from trading.application.backtest import BacktestResult
from trading.application.cancel_order import CancelFailedError, CancelOrder
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.live_trading import (
    AssessLiveReadiness,
    LiveTradingError,
    RequestLiveTrading,
)
from trading.application.political_cycle import PoliticalCycle
from trading.application.ports.backfill_seen import BackfillSeenPort
from trading.application.ports.broker import BrokerPort, OrderNotFoundError
from trading.application.ports.clock import ClockPort
from trading.application.ports.data_source import HistoricalDataSourcePort
from trading.application.ports.event_bus import EventBroadcasterPort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.market_data import Quote, QuoteFeedPort, QuoteStreamPort
from trading.application.ports.read_model import ReadModelPort
from trading.application.ports.trade_updates import TradeUpdatesStreamPort
from trading.application.proposals import ExecuteProposals, ScanProposals
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch
from trading.application.signal_recording import record_signals
from trading.application.sleeve_registry import SleeveRegistry, UnknownSleeveError
from trading.application.sync_orders import SyncOrders
from trading.domain.conviction import ConvictionThresholds, compute_conviction
from trading.domain.events import DomainEvent, EventType
from trading.domain.positions import MarkedPosition, mark_positions
from trading.domain.sleeves import TradingMode
from trading.infrastructure.observability.metrics import MetricsRegistry
from trading.infrastructure.persistence.health import DatabaseReadiness
from trading.interface.auth import ApiKeyAuth
from trading.interface.middleware.observability import ObservabilityMiddleware
from trading.interface.schemas import (
    BacktestOut,
    CancelOut,
    CapitalAllocationOut,
    CapitalOut,
    ConvictionItem,
    CycleModeIn,
    CycleModeOut,
    EvaluateOut,
    EventOut,
    ExecuteProposalsIn,
    ExecuteProposalsOut,
    IngestOut,
    KillSwitchIn,
    KillSwitchOut,
    LivePositionOut,
    LivePositionsOut,
    LiveReadinessOut,
    OrderOut,
    OrderSyncOut,
    PredictedItem,
    ProposalItem,
    ProposalsOut,
    QuoteOut,
    QuotesOut,
    ReconcileOut,
    SignalFeedItem,
    SignalIn,
    SleeveModeIn,
    SleeveModeOut,
    SleeveOut,
    TrackedSymbolsOut,
)

_USER_ACTION_STREAM = "user-actions"

logger = logging.getLogger(__name__)


@dataclass
class AppDeps:
    event_store: EventStorePort
    read_model: ReadModelPort  # CQRS read side: current-state projections (no per-request replay)
    kill_switch: KillSwitch
    sleeve_registry: SleeveRegistry
    evaluators: dict[str, EvaluateSignals]  # keyed by sleeve_id
    default_sleeve_id: str  # target for manual signal pushes
    reconcilers: list[Reconcile]  # pool-level: reconciles the whole shared book (FR-REC)
    order_syncs: list[SyncOrders]  # one per sleeve: ingest async broker fills (FR-REC)
    clock: ClockPort
    conviction_thresholds: ConvictionThresholds
    conviction_window_days: int
    # The single shared capital pool's broker (the one Alpaca book both algos draw from). Backs
    # /capital's live buying power/equity. None when no real account is wired => /capital disabled.
    pool_broker: BrokerPort | None = None
    quote_feed: QuoteFeedPort | None = None  # live tracking/marks; None => disabled
    quote_stream: QuoteStreamPort | None = None  # /ws/quotes real-time push; None => disabled
    # Real-time broker trade-updates stream (records fills/status as events); disabled => no
    # socket. Run as a background task by the composition root, not driven by any HTTP route.
    trade_updates_stream: TradeUpdatesStreamPort | None = None
    default_watchlist: list[str] = field(default_factory=list)
    political_cycle: PoliticalCycle | None = None
    # Trade mode + proposals (FR-EXE): 'auto' auto-places via the cron; 'manual' surfaces
    # actionable trades as proposals for human approval. Scan is read-only; execute places.
    cycle_trade_mode: str = "auto"
    scan_proposals: ScanProposals | None = None
    execute_proposals: ExecuteProposals | None = None
    backtest_runner: Callable[[], Awaitable[BacktestResult]] | None = None
    assess_readiness: AssessLiveReadiness | None = None
    request_live: RequestLiveTrading | None = None
    cancel_order: CancelOrder | None = None  # manual order cancellation (FR-EXE); None => off
    event_bus: EventBroadcasterPort | None = None  # live WebSocket fan-out (FR-API-1)
    reconcile_interval_seconds: int = 0  # 0 disables the scheduler (tests)
    ingest_interval_seconds: int = 0
    order_sync_interval_seconds: int = 0  # 0 disables the scheduled order-sync job
    # Observability (spec §15). Always present; the registry backs /metrics and the request
    # middleware. `enable_metrics` toggles the /metrics route's exposure only.
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    enable_metrics: bool = True
    # Readiness: probes DB connectivity when DATABASE_URL is configured (else trivially ready).
    db_readiness: DatabaseReadiness | None = None
    # Auth: API-key guard for mutating/sensitive endpoints. None => open (dev default).
    api_key: str | None = None
    # CORS: allowed browser origins when the SPA is served from a different origin than this API
    # (e.g. a static host). Empty => no CORS headers (same-origin only).
    cors_allow_origins: list[str] = field(default_factory=list)
    # Long-running background coroutines started on startup and cancelled on shutdown (e.g. the
    # Alpaca quote-stream consumer and the trade-updates stream). Each is a factory so nothing
    # runs until the app starts — when empty (the default) startup launches no task, no socket.
    background_tasks: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    # Resources to release on shutdown (e.g. the shared HTTP client for data sources).
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    # Backfill (Task 02): the dedupe store and the registry of historical sources keyed by name.
    # Read by the `python -m trading.jobs.backfill` job (not by any HTTP route). The registry is
    # EMPTY until the historical fetchers (Tasks 04a/04b) register their adapters here.
    backfill_seen: BackfillSeenPort | None = None
    historical_sources: dict[str, HistoricalDataSourcePort] = field(default_factory=dict)


def create_app(deps: AppDeps) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        scheduler = None
        jobs = _scheduled_jobs(deps)
        if jobs:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            scheduler = AsyncIOScheduler()
            for func, seconds in jobs:
                scheduler.add_job(func, "interval", seconds=seconds)
            scheduler.start()
        # Start any long-running background consumers (e.g. the live quote stream). Empty by
        # default, so a bare run launches nothing and opens no socket.
        tasks = [asyncio.ensure_future(factory()) for factory in deps.background_tasks]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                # Await cancellation; swallow CancelledError and any teardown error.
                await asyncio.gather(*tasks, return_exceptions=True)
            if scheduler is not None:
                scheduler.shutdown(wait=False)
            for aclose in deps.aclosers:
                await aclose()

    app = FastAPI(title="Trading Platform", version="0.2.0", lifespan=lifespan)
    # Cross-origin access for a browser SPA served from a DIFFERENT origin than this API (e.g. a
    # static host). Empty => no CORS headers (same-origin only), preserving the default.
    if deps.cors_allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=deps.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # Structured JSON access logging + request/latency metrics + a propagated request id.
    app.add_middleware(ObservabilityMiddleware, metrics=deps.metrics)

    # API-key guard for mutating/sensitive endpoints. Unset => open (dev default; auth logs a
    # single startup warning). Set => protected routes require the key. GET reads stay open.
    auth = ApiKeyAuth(deps.api_key)
    protected = [Depends(auth)]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    async def liveness() -> dict[str, str]:
        # Liveness: the process is up and serving. No dependency checks (those are /readyz).
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness() -> dict[str, object]:
        # Readiness: safe to receive traffic. Probes the database when one is configured; the
        # in-memory default has no external dependency and is trivially ready.
        checks: dict[str, str] = {}
        ready = True
        if deps.db_readiness is not None and deps.db_readiness.configured:
            try:
                await deps.db_readiness.check()
                checks["database"] = "ok"
            except Exception:
                # Surface any DB failure as not-ready (503), not a 500.
                logger.warning("readiness: database check failed", exc_info=True)
                checks["database"] = "error"
                ready = False
        if not ready:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
        return {"status": "ready", "checks": checks}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        if not deps.enable_metrics:
            raise HTTPException(status_code=404, detail="metrics are disabled")
        # Prometheus text-exposition format (hand-rolled; no client-library dependency).
        return PlainTextResponse(
            deps.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/events", response_model=list[EventOut])
    async def list_events(
        stream_id: Annotated[str | None, Query()] = None,
        correlation_id: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> list[EventOut]:
        if stream_id is not None:
            stored = await deps.event_store.read_stream(stream_id)
        elif correlation_id is not None:
            stored = await deps.event_store.read_by_correlation(correlation_id)
        else:
            stored = await deps.event_store.read_all(limit=limit)
        return [EventOut.from_stored(s) for s in stored[-limit:]]

    @app.post("/signals", response_model=EvaluateOut, dependencies=protected)
    async def submit_signals(
        signals: list[SignalIn],
        sleeve_id: Annotated[str | None, Query()] = None,
    ) -> EvaluateOut:
        target_sleeve = sleeve_id or deps.default_sleeve_id
        evaluator = deps.evaluators.get(target_sleeve)
        if evaluator is None:
            raise HTTPException(status_code=404, detail=f"unknown sleeve '{target_sleeve}'")
        correlation_id = str(uuid4())
        domain_signals = [s.to_domain(deps.clock.now()) for s in signals]
        # Record the signals, then evaluate under the same correlation id (spec section 9).
        await record_signals(
            event_store=deps.event_store,
            clock=deps.clock,
            signals=domain_signals,
            correlation_id=correlation_id,
        )
        result = await evaluator.execute(domain_signals, correlation_id=correlation_id)
        return EvaluateOut(
            correlation_id=result.correlation_id,
            orders=[OrderOut.from_broker(o) for o in result.orders],
        )

    @app.get("/signals/feed", response_model=list[SignalFeedItem])
    async def signal_feed(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> list[SignalFeedItem]:
        records = await deps.read_model.recent_signals(limit=limit)
        return [SignalFeedItem.from_record(r) for r in records]

    @app.get("/conviction", response_model=list[ConvictionItem])
    async def conviction() -> list[ConvictionItem]:
        window = await deps.read_model.signals_in_window(
            now=deps.clock.now(), window_days=deps.conviction_window_days
        )
        items: list[ConvictionItem] = []
        for ticker in sorted({s.ticker for s in window}):
            result = compute_conviction(
                ticker,
                window,
                window_days=deps.conviction_window_days,
                thresholds=deps.conviction_thresholds,
            )
            items.append(
                ConvictionItem(
                    ticker=ticker,
                    score=result.score,
                    action=result.action.value,
                    contributing_signal_ids=list(result.contributing_signal_ids),
                )
            )
        return items

    @app.get("/proposals", response_model=ProposalsOut)
    async def proposals() -> ProposalsOut:
        # Read-only conviction scan: TRADE-tier -> actionable proposals (manual-approve / buy),
        # WATCH-tier -> predicted/incoming ("about to buy"). Empty until real data is collected.
        if deps.scan_proposals is None:
            return ProposalsOut(mode=deps.cycle_trade_mode, proposals=[], predicted=[])
        scan = await deps.scan_proposals.execute()
        return ProposalsOut(
            mode=deps.cycle_trade_mode,
            proposals=[ProposalItem.from_candidate(c) for c in scan.proposals],
            predicted=[PredictedItem.from_candidate(c) for c in scan.predicted],
        )

    @app.post("/proposals/execute", response_model=ExecuteProposalsOut, dependencies=protected)
    async def execute_proposals(body: ExecuteProposalsIn) -> ExecuteProposalsOut:
        # Manual buy: individual (symbols=[...]) or blanket (symbols omitted -> all proposals).
        if deps.execute_proposals is None:
            raise HTTPException(status_code=409, detail="trading is not configured")
        result = await deps.execute_proposals.execute(symbols=body.symbols)
        return ExecuteProposalsOut(
            correlation_id=result.correlation_id,
            orders=[OrderOut.from_broker(o) for o in result.orders],
        )

    @app.get("/quotes", response_model=QuotesOut)
    async def quotes(symbols: Annotated[str | None, Query()] = None) -> QuotesOut:
        feed = deps.quote_feed
        if feed is None or not feed.enabled:
            return QuotesOut(enabled=False, as_of=None, quotes=[])
        wanted = _parse_symbols(symbols) or deps.default_watchlist
        fetched = await feed.get_quotes(wanted)
        as_of = max((q.as_of for q in fetched), default=deps.clock.now())
        return QuotesOut(
            enabled=True,
            as_of=as_of,
            quotes=[QuoteOut.from_quote(q) for q in fetched],
        )

    @app.get("/quotes/tracked", response_model=TrackedSymbolsOut)
    async def tracked_symbols() -> TrackedSymbolsOut:
        held = await deps.read_model.held_positions()
        symbols = sorted({p.symbol for p in held})
        return TrackedSymbolsOut(symbols=symbols or list(deps.default_watchlist))

    @app.get("/positions/live", response_model=LivePositionsOut)
    async def live_positions() -> LivePositionsOut:
        held = await deps.read_model.held_positions()
        feed = deps.quote_feed
        enabled = feed is not None and feed.enabled
        marks: dict[str, Decimal] = {}
        if enabled and held:
            assert feed is not None
            symbols = sorted({p.symbol for p in held})
            marks = {q.symbol: q.last for q in await feed.get_quotes(symbols)}
        marked = mark_positions(held, marks)
        return LivePositionsOut(
            enabled=enabled,
            positions=[LivePositionOut.from_marked(m) for m in marked],
        )

    @app.get("/capital", response_model=CapitalOut)
    async def capital() -> CapitalOut:
        # The single shared capital pool both algos draw from = the live Alpaca account. Reports
        # buying power / cash / equity for that one book, plus each algo's slice of it (market value
        # of what that sleeve holds, marked to the live feed when available). Disabled shape when no
        # real account is wired, so the SPA renders a graceful empty state like the other feeds.
        broker = deps.pool_broker
        if broker is None:
            return CapitalOut(
                enabled=False,
                cash=Decimal("0"),
                equity=Decimal("0"),
                buying_power=Decimal("0"),
                allocations=[],
            )
        account = await broker.get_account()
        held = await deps.read_model.held_positions()
        feed = deps.quote_feed
        marks: dict[str, Decimal] = {}
        if feed is not None and feed.enabled and held:
            symbols = sorted({p.symbol for p in held})
            marks = {q.symbol: q.last for q in await feed.get_quotes(symbols)}
        by_sleeve: dict[str, list[MarkedPosition]] = {}
        for marked in mark_positions(held, marks):
            by_sleeve.setdefault(marked.sleeve_id, []).append(marked)
        allocations = [
            CapitalAllocationOut(
                sleeve_id=state.sleeve.id,
                strategy_id=state.sleeve.strategy_id,
                market_value=sum(
                    (m.market_value for m in by_sleeve.get(state.sleeve.id, [])), Decimal("0")
                ),
                unrealized_pnl=sum(
                    (m.unrealized_pnl for m in by_sleeve.get(state.sleeve.id, [])), Decimal("0")
                ),
                positions=len(by_sleeve.get(state.sleeve.id, [])),
            )
            for state in deps.sleeve_registry.all()
        ]
        return CapitalOut(
            enabled=True,
            cash=account.cash,
            equity=account.equity,
            buying_power=account.buying_power,
            allocations=allocations,
        )

    @app.post("/ingest", response_model=IngestOut, dependencies=protected)
    async def ingest() -> IngestOut:
        if deps.political_cycle is None:
            raise HTTPException(
                status_code=409, detail="no data sources configured; ingestion is disabled"
            )
        cycle = await deps.political_cycle.execute()
        return IngestOut(
            ingestion_correlation_id=cycle.ingestion.correlation_id,
            signals_ingested=len(cycle.ingestion.signals),
            failed_sources=list(cycle.ingestion.failed_sources),
            evaluation_correlation_id=cycle.evaluation.correlation_id,
            orders=[OrderOut.from_broker(o) for o in cycle.evaluation.orders],
        )

    @app.post("/backtest", response_model=BacktestOut, dependencies=protected)
    async def run_backtest() -> BacktestOut:
        if deps.backtest_runner is None:
            raise HTTPException(status_code=409, detail="backtesting is not configured")
        return BacktestOut.from_result(await deps.backtest_runner())

    @app.get("/sleeves", response_model=list[SleeveOut])
    async def list_sleeves() -> list[SleeveOut]:
        return [SleeveOut.from_state(s) for s in deps.sleeve_registry.all()]

    @app.post(
        "/sleeves/{sleeve_id}/readiness",
        response_model=LiveReadinessOut,
        dependencies=protected,
    )
    async def assess_readiness(sleeve_id: str) -> LiveReadinessOut:
        if deps.assess_readiness is None:
            raise HTTPException(status_code=409, detail="live trading is not configured")
        _require_sleeve(deps, sleeve_id)
        report = await deps.assess_readiness.assess(sleeve_id)
        return LiveReadinessOut.from_report(report)

    @app.post("/sleeves/{sleeve_id}/mode", response_model=SleeveModeOut, dependencies=protected)
    async def set_sleeve_mode(sleeve_id: str, body: SleeveModeIn) -> SleeveModeOut:
        if deps.request_live is None:
            raise HTTPException(status_code=409, detail="live trading is not configured")
        _require_sleeve(deps, sleeve_id)
        readiness: LiveReadinessOut | None = None
        try:
            if body.mode is TradingMode.LIVE:
                report = await deps.request_live.enable(sleeve_id, confirm=body.confirm)
                readiness = LiveReadinessOut.from_report(report)
            else:
                await deps.request_live.disable(sleeve_id)
        except LiveTradingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        state = deps.sleeve_registry.get(sleeve_id)
        return SleeveModeOut(sleeve=SleeveOut.from_state(state), readiness=readiness)

    @app.get("/cycle/mode", response_model=CycleModeOut)
    async def get_cycle_mode() -> CycleModeOut:
        return CycleModeOut(mode=deps.cycle_trade_mode)

    @app.post("/cycle/mode", response_model=CycleModeOut, dependencies=protected)
    async def set_cycle_mode(body: CycleModeIn) -> CycleModeOut:
        # Live arm/disarm of autonomous trading for the scheduled political cycle. Updates the
        # reporting copy AND the running PoliticalCycle so the next scheduled tick honours it
        # with no redeploy. NOT persisted: reverts to the CYCLE_TRADE_MODE env default on
        # restart, so a restart never silently resumes 'auto'.
        deps.cycle_trade_mode = body.mode
        if deps.political_cycle is not None:
            deps.political_cycle.trade_mode = body.mode
        return CycleModeOut(mode=body.mode)

    @app.get("/kill-switch", response_model=KillSwitchOut)
    async def get_kill_switch() -> KillSwitchOut:
        return KillSwitchOut(engaged=deps.kill_switch.engaged)

    @app.post("/kill-switch", response_model=KillSwitchOut, dependencies=protected)
    async def set_kill_switch(body: KillSwitchIn) -> KillSwitchOut:
        correlation_id = str(uuid4())
        if body.engaged:
            deps.kill_switch.engage()
        else:
            deps.kill_switch.disengage()
        # Every user interaction is an event (spec section 9).
        await _append(
            deps,
            _USER_ACTION_STREAM,
            EventType.USER_ACTION,
            {"action": "kill_switch", "engaged": body.engaged},
            correlation_id,
        )
        await _append(
            deps,
            "risk",
            EventType.KILL_SWITCH_CHANGED,
            {"engaged": body.engaged},
            correlation_id,
        )
        return KillSwitchOut(engaged=deps.kill_switch.engaged)

    @app.post("/reconcile", response_model=list[ReconcileOut], dependencies=protected)
    async def run_reconcile() -> list[ReconcileOut]:
        return [ReconcileOut.from_report(await r.execute()) for r in deps.reconcilers]

    @app.post("/orders/sync", response_model=list[OrderSyncOut], dependencies=protected)
    async def sync_orders() -> list[OrderSyncOut]:
        return [OrderSyncOut.from_report(await s.execute()) for s in deps.order_syncs]

    @app.post("/orders/{client_order_id}/cancel", response_model=CancelOut, dependencies=protected)
    async def cancel_order(
        client_order_id: str,
        sleeve_id: Annotated[str | None, Query()] = None,
    ) -> CancelOut:
        if deps.cancel_order is None:
            raise HTTPException(status_code=409, detail="order cancellation is not configured")
        try:
            result = await deps.cancel_order.execute(
                client_order_id=client_order_id, sleeve_id=sleeve_id
            )
        except OrderNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CancelFailedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CancelOut.from_result(result)

    @app.websocket("/ws/events")
    async def stream_events(websocket: WebSocket) -> None:
        if deps.event_bus is None:
            await websocket.close(code=1011)
            return
        await websocket.accept()
        queue = deps.event_bus.subscribe()
        try:
            while True:
                stored = await queue.get()
                await websocket.send_text(EventOut.from_stored(stored).model_dump_json())
        except WebSocketDisconnect:
            pass
        finally:
            deps.event_bus.unsubscribe(queue)

    @app.websocket("/ws/quotes")
    async def stream_quotes(websocket: WebSocket) -> None:
        await websocket.accept()
        stream = deps.quote_stream
        enabled = stream is not None and stream.enabled
        # Contract: the first frame is always the status, so the client knows whether to
        # expect live quotes or fall back to charts-only.
        await websocket.send_json({"type": "status", "enabled": enabled})
        if not enabled or stream is None:
            # Disabled: hold the socket open but never push or error. Wait for the client to
            # disconnect (any inbound frame raises WebSocketDisconnect on close).
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            return

        # Enabled: seed the upstream subscription with the platform's tracked symbols, then
        # fan quotes out of a per-connection queue while concurrently honoring client
        # subscribe requests. Mirrors /ws/events (per-connection queue, clean disconnect).
        await stream.add_symbols(await _tracked_symbols(deps))
        queue = stream.subscribe()
        push = asyncio.ensure_future(_push_quotes(websocket, queue))
        recv = asyncio.ensure_future(_handle_quote_requests(websocket, stream))
        try:
            await asyncio.wait({push, recv}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (push, recv):
                task.cancel()
            stream.unsubscribe(queue)

    return app


async def _tracked_symbols(deps: AppDeps) -> list[str]:
    held = await deps.read_model.held_positions()
    symbols = sorted({p.symbol for p in held})
    return symbols or list(deps.default_watchlist)


async def _push_quotes(websocket: WebSocket, queue: "asyncio.Queue[Quote]") -> None:
    while True:
        quote = await queue.get()
        await websocket.send_json(
            {
                "type": "quote",
                "symbol": quote.symbol,
                "bid": str(quote.bid),
                "ask": str(quote.ask),
                "last": str(quote.last),
                "as_of": quote.as_of.isoformat(),
            }
        )


async def _handle_quote_requests(websocket: WebSocket, stream: QuoteStreamPort) -> None:
    while True:
        message = await websocket.receive_json()
        if isinstance(message, dict) and message.get("action") == "subscribe":
            requested = message.get("symbols")
            if isinstance(requested, list):
                await stream.add_symbols([str(s) for s in requested])


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _require_sleeve(deps: AppDeps, sleeve_id: str) -> None:
    try:
        deps.sleeve_registry.get(sleeve_id)
    except UnknownSleeveError:
        raise HTTPException(status_code=404, detail=f"unknown sleeve '{sleeve_id}'") from None


async def _reconcile_all(deps: AppDeps) -> None:
    for reconciler in deps.reconcilers:
        await reconciler.execute()


async def _sync_orders_all(deps: AppDeps) -> None:
    for sync in deps.order_syncs:
        await sync.execute()


def _scheduled_jobs(deps: AppDeps) -> list[tuple[Any, int]]:
    jobs: list[tuple[Any, int]] = []
    if deps.reconcile_interval_seconds > 0 and deps.reconcilers:
        jobs.append((lambda: _reconcile_all(deps), deps.reconcile_interval_seconds))
    if deps.ingest_interval_seconds > 0 and deps.political_cycle is not None:
        jobs.append((deps.political_cycle.execute, deps.ingest_interval_seconds))
    if deps.order_sync_interval_seconds > 0 and deps.order_syncs:
        jobs.append((lambda: _sync_orders_all(deps), deps.order_sync_interval_seconds))
    return jobs


async def _append(
    deps: AppDeps,
    stream_id: str,
    event_type: str,
    payload: dict[str, Any],
    correlation_id: str,
) -> None:
    await deps.event_store.append(
        DomainEvent(
            stream_id=stream_id,
            event_type=event_type,
            payload=payload,
            occurred_at=deps.clock.now(),
            correlation_id=correlation_id,
        )
    )
