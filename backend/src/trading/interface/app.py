"""FastAPI application. Receives fully wired dependencies from the composition root."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

from trading.application.backtest import BacktestResult
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.live_trading import (
    AssessLiveReadiness,
    LiveTradingError,
    RequestLiveTrading,
)
from trading.application.political_cycle import PoliticalCycle
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_bus import EventBroadcasterPort
from trading.application.ports.event_store import EventStorePort
from trading.application.projections import signals_from_events
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch
from trading.application.signal_recording import record_signals
from trading.application.sleeve_registry import SleeveRegistry, UnknownSleeveError
from trading.domain.conviction import ConvictionThresholds, compute_conviction
from trading.domain.events import DomainEvent, EventType
from trading.domain.sleeves import TradingMode
from trading.interface.schemas import (
    BacktestOut,
    ConvictionItem,
    EvaluateOut,
    EventOut,
    IngestOut,
    KillSwitchIn,
    KillSwitchOut,
    LiveReadinessOut,
    OrderOut,
    ReconcileOut,
    SignalFeedItem,
    SignalIn,
    SleeveModeIn,
    SleeveModeOut,
    SleeveOut,
)

_USER_ACTION_STREAM = "user-actions"


@dataclass
class AppDeps:
    event_store: EventStorePort
    kill_switch: KillSwitch
    sleeve_registry: SleeveRegistry
    evaluators: dict[str, EvaluateSignals]  # keyed by sleeve_id
    default_sleeve_id: str  # target for manual signal pushes
    reconcilers: list[Reconcile]  # one per sleeve (broker is per-sleeve)
    clock: ClockPort
    conviction_thresholds: ConvictionThresholds
    conviction_window_days: int
    political_cycle: PoliticalCycle | None = None
    backtest_runner: Callable[[], Awaitable[BacktestResult]] | None = None
    assess_readiness: AssessLiveReadiness | None = None
    request_live: RequestLiveTrading | None = None
    event_bus: EventBroadcasterPort | None = None  # live WebSocket fan-out (FR-API-1)
    reconcile_interval_seconds: int = 0  # 0 disables the scheduler (tests)
    ingest_interval_seconds: int = 0
    # Resources to release on shutdown (e.g. the shared HTTP client for data sources).
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


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
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)
            for aclose in deps.aclosers:
                await aclose()

    app = FastAPI(title="Trading Platform", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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

    @app.post("/signals", response_model=EvaluateOut)
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
        events = await deps.event_store.read_all()
        feed = [
            SignalFeedItem.from_stored(s)
            for s in events
            if s.event.event_type == EventType.SIGNAL_RECEIVED
        ]
        return feed[-limit:]

    @app.get("/conviction", response_model=list[ConvictionItem])
    async def conviction() -> list[ConvictionItem]:
        events = await deps.event_store.read_all()
        window = signals_from_events(
            events, now=deps.clock.now(), window_days=deps.conviction_window_days
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

    @app.post("/ingest", response_model=IngestOut)
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

    @app.post("/backtest", response_model=BacktestOut)
    async def run_backtest() -> BacktestOut:
        if deps.backtest_runner is None:
            raise HTTPException(status_code=409, detail="backtesting is not configured")
        return BacktestOut.from_result(await deps.backtest_runner())

    @app.get("/sleeves", response_model=list[SleeveOut])
    async def list_sleeves() -> list[SleeveOut]:
        return [SleeveOut.from_state(s) for s in deps.sleeve_registry.all()]

    @app.post("/sleeves/{sleeve_id}/readiness", response_model=LiveReadinessOut)
    async def assess_readiness(sleeve_id: str) -> LiveReadinessOut:
        if deps.assess_readiness is None:
            raise HTTPException(status_code=409, detail="live trading is not configured")
        _require_sleeve(deps, sleeve_id)
        report = await deps.assess_readiness.assess(sleeve_id)
        return LiveReadinessOut.from_report(report)

    @app.post("/sleeves/{sleeve_id}/mode", response_model=SleeveModeOut)
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

    @app.get("/kill-switch", response_model=KillSwitchOut)
    async def get_kill_switch() -> KillSwitchOut:
        return KillSwitchOut(engaged=deps.kill_switch.engaged)

    @app.post("/kill-switch", response_model=KillSwitchOut)
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

    @app.post("/reconcile", response_model=list[ReconcileOut])
    async def run_reconcile() -> list[ReconcileOut]:
        return [ReconcileOut.from_report(await r.execute()) for r in deps.reconcilers]

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

    return app


def _require_sleeve(deps: AppDeps, sleeve_id: str) -> None:
    try:
        deps.sleeve_registry.get(sleeve_id)
    except UnknownSleeveError:
        raise HTTPException(status_code=404, detail=f"unknown sleeve '{sleeve_id}'") from None


async def _reconcile_all(deps: AppDeps) -> None:
    for reconciler in deps.reconcilers:
        await reconciler.execute()


def _scheduled_jobs(deps: AppDeps) -> list[tuple[Any, int]]:
    jobs: list[tuple[Any, int]] = []
    if deps.reconcile_interval_seconds > 0 and deps.reconcilers:
        jobs.append((lambda: _reconcile_all(deps), deps.reconcile_interval_seconds))
    if deps.ingest_interval_seconds > 0 and deps.political_cycle is not None:
        jobs.append((deps.political_cycle.execute, deps.ingest_interval_seconds))
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
