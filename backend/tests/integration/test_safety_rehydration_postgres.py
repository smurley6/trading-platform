"""Cross-process rehydration of safety state over real Postgres (review F2/F3).

The unit tests cover the derivation logic on an in-memory log; these prove the composition root
actually seeds a FRESH process from the durable Postgres log — the real cron-vs-web / redeploy
scenario the review flagged. Require TEST_DATABASE_URL with migrations applied.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from trading.config import Settings
from trading.domain.events import DomainEvent, EventType
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.interface.app import AppDeps
from trading.main import build_dependencies

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL not set",
    ),
]

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "NEWS_FEED_URLS",
    "USASPENDING_ENABLED",
    "DISCLOSED_TRADE_LEG_ENABLED",
    "CONGRESS_TRADES_ENDPOINT",
    "FMP_API_KEY",
    "HOUSE_DISCLOSURES_ENABLED",
    "SENATE_EFD_ENABLED",
    "SEC_FORM4_ENABLED",
    "ENABLE_LIVE_QUOTES",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "ENABLE_TRADE_UPDATES_STREAM",
    "REQUIRE_REAL_DATA",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "ALPACA_LIVE_KEY_ID",
    "ALPACA_LIVE_SECRET_KEY",
    "API_KEY",
)


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])


async def _append(*events: DomainEvent) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres import PostgresEventStore

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresEventStore(engine)
    try:
        for event in events:
            await store.append(event)
    finally:
        await engine.dispose()


def _kill_switch(*, engaged: bool) -> DomainEvent:
    return DomainEvent(
        stream_id=f"risk-{uuid4()}",
        event_type=EventType.KILL_SWITCH_CHANGED,
        payload={"engaged": engaged},
        occurred_at=datetime.now(tz=UTC),
        correlation_id=str(uuid4()),
    )


def _snapshot(equity: str) -> DomainEvent:
    return DomainEvent(
        stream_id=f"sleeve-{uuid4()}",
        event_type=EventType.PORTFOLIO_SNAPSHOT,
        payload={
            "sleeve_id": "sleeve-political",
            "cash": equity,
            "equity": equity,
            "positions": {},
        },
        occurred_at=datetime.now(tz=UTC),
        correlation_id=str(uuid4()),
    )


@asynccontextmanager
async def _built_deps() -> AsyncIterator[AppDeps]:
    """Build a fresh composition over the shared Postgres log and release its engines after."""
    deps = build_dependencies(Settings(_env_file=None))
    try:
        yield deps
    finally:
        for aclose in deps.aclosers:
            await aclose()


async def test_engaged_kill_switch_survives_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    # A web process engaged the emergency stop; the fact is durable in the append-only log.
    await _append(_kill_switch(engaged=True))

    # A brand-new process (the daily cron / a redeploy) builds its own dependencies.
    async with _built_deps() as deps:
        assert deps.kill_switch.engaged is True

        # ...and the wired risk gate blocks a corroboration-worthy order end to end.
        now = deps.clock.now()
        signals = [
            _lmt_signal(now, SignalSource.STATEMENT, sentiment=0.9),
            _lmt_signal(now, SignalSource.CONTRACT, sentiment=0.8),
        ]
        result = await deps.evaluators["sleeve-political"].execute(signals)
        assert result.orders == ()
        rejections = [
            s.event.payload["reason"]
            for s in await deps.event_store.read_by_correlation(result.correlation_id)
            if s.event.event_type == EventType.RISK_EVALUATED
        ]
        assert rejections, "a proposal reached the risk gate"
        assert all("kill switch" in str(r) for r in rejections)


async def test_latest_disengage_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    # Engage then disengage: the fresh process must inherit the LATEST (disengaged) state.
    await _append(_kill_switch(engaged=True), _kill_switch(engaged=False))

    async with _built_deps() as deps:
        assert deps.kill_switch.engaged is False


async def test_redeploy_preserves_engaged_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    await _append(_kill_switch(engaged=True))

    # Two independent builds over the same durable log = two restarts; both stay engaged.
    for _ in range(2):
        async with _built_deps() as deps:
            assert deps.kill_switch.engaged is True


async def test_peak_equity_from_the_log_trips_the_drawdown_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    # A high-water mark far above any other test's snapshots (so max() is unambiguous) and far
    # above the simulator's starting equity => a huge drawdown. Disengage the kill switch last so
    # the rejection is unambiguously the drawdown breaker, not a leftover engaged switch.
    await _append(_snapshot("987654321.00"), _snapshot("1000"), _kill_switch(engaged=False))

    async with _built_deps() as deps:
        assert deps.kill_switch.engaged is False
        now = deps.clock.now()
        signals = [
            _lmt_signal(now, SignalSource.STATEMENT, sentiment=0.9),
            _lmt_signal(now, SignalSource.CONTRACT, sentiment=0.8),
        ]
        result = await deps.evaluators["sleeve-political"].execute(signals)
        assert result.orders == ()
        rejections = [
            s.event.payload["reason"]
            for s in await deps.event_store.read_by_correlation(result.correlation_id)
            if s.event.event_type == EventType.RISK_EVALUATED
        ]
        assert rejections, "a proposal reached the risk gate"
        assert all("drawdown" in str(r) for r in rejections)


def _lmt_signal(created_at: datetime, source_type: SignalSource, *, sentiment: float) -> Signal:
    return Signal(
        id=str(uuid4()),
        created_at=created_at,
        source_type=source_type,
        ticker="LMT",
        politician_id="pol-1",
        sentiment=sentiment,
        politician_weight=1.5,
        source_reliability=1.0,
        raw_payload={"source": "integration-fixture"},
        provenance=Provenance.SYNTHETIC,
        pit_confidence="exact",
    )
