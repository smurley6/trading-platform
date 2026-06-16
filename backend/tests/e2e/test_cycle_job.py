"""One-shot cycle job (``python -m trading.jobs.cycle``): runs ONE political cycle on the real
composition root and tears down cleanly. No network, no real DB — the default in-memory store.

Covers the two contract paths:
  * no data sources configured -> ``political_cycle`` is None -> clean no-op, exit 0;
  * a fake source producing corroborated signals -> the cycle ingests, evaluates, and trades,
    with state recorded in the (in-memory) event store.
Plus the teardown contract: every registered acloser is awaited, and ``main()`` maps a failing
run to a non-zero exit code.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tests.conftest import SignalFactory
from trading.application.ingest_signals import IngestSignals
from trading.application.political_cycle import PoliticalCycle
from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.orders import OrderStatus
from trading.domain.signals import Signal, SignalSource
from trading.interface.app import AppDeps
from trading.jobs import cycle as cycle_job
from trading.main import build_dependencies

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}

# Every external integration that the composition root reads from the environment. Cleared so a
# developer's .env / shell can never leak credentials (or a real DATABASE_URL) into these tests.
_CLEARED_ENV = (
    "DATABASE_URL",
    "NEWS_FEED_URLS",
    "USASPENDING_ENABLED",
    "DISCLOSED_TRADE_LEG_ENABLED",
    "CONGRESS_TRADES_ENDPOINT",
    "FMP_API_KEY",
    "ENABLE_LIVE_QUOTES",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "ENABLE_TRADE_UPDATES_STREAM",
    "REQUIRE_REAL_DATA",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_LIVE_KEY_ID",
    "ALPACA_LIVE_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)


class FakeSource:
    """Minimal in-memory ``DataSourcePort``: returns its canned signals on every poll."""

    def __init__(self, name: str, signals: list[Signal]) -> None:
        self._name = name
        self._signals = signals

    @property
    def name(self) -> str:
        return self._name

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        return self._signals


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)


def _settings() -> Settings:
    # _env_file=None so the test never reads a developer .env file.
    return Settings(_env_file=None)


async def test_run_cycle_no_sources_is_a_clean_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default settings configure NO data sources -> build_dependencies leaves political_cycle
    # None. run_cycle must log-and-return without raising (exit 0).
    _base_env(monkeypatch)
    monkeypatch.setattr(cycle_job, "Settings", _settings)

    await cycle_job.run_cycle()  # must not raise


def test_main_returns_zero_on_clean_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # main() drives its own event loop (asyncio.run), so this test stays SYNCHRONOUS — calling
    # asyncio.run from inside a running loop (an async test) would raise.
    _base_env(monkeypatch)
    monkeypatch.setattr(cycle_job, "Settings", _settings)

    assert cycle_job.main() == 0


async def test_run_cycle_ingests_and_trades_with_a_fake_source(
    monkeypatch: pytest.MonkeyPatch, make_signal: SignalFactory
) -> None:
    # Wire a real in-memory composition (NEWS_FEED_URLS turns the news source on), then swap the
    # source list for a fake producing two corroborated LMT signals -> a filled paper order.
    # This proves the job drives the existing ingest -> evaluate -> place-order path end to end.
    _base_env(monkeypatch)
    monkeypatch.setenv("NEWS_FEED_URLS", '["https://example.test/feed.xml"]')

    # Two corroborated LMT signals (statement + contract, strongly positive) -> conviction over
    # threshold -> a trade. Note: the real composition uses the wall-clock SystemClock, so we
    # re-stamp created_at at the deps clock's now (see below) to keep the test date-independent.
    base_signals = [
        make_signal(
            ticker="LMT", sentiment=0.9, politician_weight=1.5, source_type=SignalSource.STATEMENT
        ),
        make_signal(
            ticker="LMT", sentiment=0.8, politician_weight=1.5, source_type=SignalSource.CONTRACT
        ),
    ]

    captured: dict[str, AppDeps] = {}
    real_build = build_dependencies

    def build_with_fake_source(settings: Settings) -> AppDeps:
        deps = real_build(settings)
        assert deps.political_cycle is not None  # NEWS_FEED_URLS configured a source
        now = deps.clock.now()
        # Rebuild the cycle through its PUBLIC constructor, reusing the composition's real
        # event store / evaluator / clock but swapping ingestion for our fake source. Cleaner
        # than mutating private internals and keeps the rest of the wired path untouched. The
        # signals are stamped at the (real) clock so they fall inside the corroboration window
        # and after the lookback cursor regardless of the calendar date the job runs.
        signals: list[Signal] = [replace(s, created_at=now) for s in base_signals]
        deps.political_cycle = PoliticalCycle(
            ingest=IngestSignals(
                sources=[FakeSource("news", signals)],
                event_store=deps.event_store,
                clock=deps.clock,
                lookback_start=now - timedelta(days=1),
            ),
            evaluate=deps.evaluators["sleeve-political"],
            event_store=deps.event_store,
            clock=deps.clock,
            window_days=settings.corroboration_window_days,
        )
        captured["deps"] = deps
        return deps

    monkeypatch.setattr(cycle_job, "Settings", _settings)
    monkeypatch.setattr(cycle_job, "build_dependencies", build_with_fake_source)

    await cycle_job.run_cycle()

    deps = captured["deps"]
    stored = await deps.event_store.read_all()
    received = [s for s in stored if s.event.event_type == EventType.SIGNAL_RECEIVED]
    decisions = [s for s in stored if s.event.event_type == EventType.DECISION_MADE]
    assert len(received) == 2
    assert len(decisions) == 1

    # The cycle placed a real order through the wired execution path: the (in-memory) paper
    # broker now holds an LMT position and the order log shows a fill, recorded as events.
    evaluator = deps.evaluators["sleeve-political"]
    broker = evaluator._broker  # the per-sleeve ModeRoutingBroker (a BrokerPort)
    positions = {p.symbol: p for p in await broker.list_positions()}
    assert "LMT" in positions
    filled_events = [s for s in stored if s.event.event_type == EventType.ORDER_FILLED]
    assert filled_events  # an order filled during the cycle
    assert any(o.status is OrderStatus.FILLED for o in await broker.list_orders())


async def test_run_cycle_always_tears_down_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even on a clean no-op run, every registered acloser must be awaited (mirrors app shutdown).
    _base_env(monkeypatch)
    monkeypatch.setattr(cycle_job, "Settings", _settings)

    closed: list[str] = []
    real_build = build_dependencies

    def build_with_probe(settings: Settings) -> AppDeps:
        deps = real_build(settings)

        async def _record() -> None:
            closed.append("aclose")

        deps.aclosers.append(_record)
        return deps

    monkeypatch.setattr(cycle_job, "build_dependencies", build_with_probe)

    await cycle_job.run_cycle()

    assert closed == ["aclose"]


def test_main_returns_nonzero_when_run_cycle_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Synchronous for the same reason as above: main() owns the event loop via asyncio.run.
    async def boom() -> None:
        raise RuntimeError("composition exploded")

    monkeypatch.setattr(cycle_job, "run_cycle", boom)

    assert cycle_job.main() == 1


async def test_teardown_isolates_a_failing_acloser() -> None:
    # A failing close must not strand the others (the Postgres pool must still be disposed even
    # if an HTTP client close raises). _teardown swallows + logs per-resource errors.
    closed: list[str] = []

    async def boom() -> None:
        raise RuntimeError("close failed")

    async def ok() -> None:
        closed.append("ok")

    deps = AppDeps.__new__(AppDeps)  # bare instance; only aclosers is read by _teardown
    aclosers: list[Callable[[], Awaitable[None]]] = [boom, ok]
    deps.aclosers = aclosers

    await cycle_job._teardown(deps)  # must not raise

    assert closed == ["ok"]
