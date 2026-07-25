"""Real-time ingest -> broadcast pipeline (FR-API-1): the path that powers live signal
streaming. Two deterministic tests, neither of which waits on wall-clock APScheduler intervals:

  * scheduler wiring: the political-cycle ingest job is registered exactly when an interval is
    configured AND a cycle exists, and is absent when the interval is zeroed;
  * live push: a freshly ingested signal is fanned out over the /ws/events WebSocket as a
    ``signal.received`` event.

Both build the *real* composition root (trading.main.build_dependencies) and reuse the
test_cycle_job fixtures (FakeSource, _base_env, _settings) so the wired ingest -> record ->
broadcast path is exercised end to end, swapping only the data source for a fast fake.
"""

from dataclasses import replace
from datetime import timedelta

import pytest
from starlette.testclient import TestClient

from tests.conftest import SignalFactory
from tests.e2e.test_cycle_job import FakeSource, _base_env, _settings
from trading.application.ingest_signals import IngestSignals
from trading.application.political_cycle import PoliticalCycle
from trading.domain.events import EventType
from trading.domain.signals import Signal, SignalSource
from trading.interface.app import AppDeps, _loop_definitions, create_app
from trading.interface.system_status import ScheduledLoop
from trading.main import build_dependencies


def _build_with_fake_source(make_signal: SignalFactory) -> AppDeps:
    """Real composition with the news source swapped for a fast in-memory fake.

    Mirrors test_cycle_job.build_with_fake_source: NEWS_FEED_URLS turns ingestion on so
    build_dependencies wires a political_cycle, then we rebuild that cycle through its public
    constructor reusing the composition's real (broadcasting) event store / evaluator / clock.
    The two corroborated LMT signals are stamped at the real clock so they land inside the
    corroboration window and after the lookback cursor regardless of the calendar date.
    """
    deps = build_dependencies(_settings())
    assert deps.political_cycle is not None  # NEWS_FEED_URLS configured a source
    assert deps.event_bus is not None  # build_dependencies always wires the broadcaster
    now = deps.clock.now()
    base_signals = [
        make_signal(
            ticker="LMT", sentiment=0.9, politician_weight=1.5, source_type=SignalSource.STATEMENT
        ),
        make_signal(
            ticker="LMT", sentiment=0.8, politician_weight=1.5, source_type=SignalSource.CONTRACT
        ),
    ]
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
        window_days=_settings().corroboration_window_days,
    )
    return deps


def _ingest_loop(deps: AppDeps) -> ScheduledLoop:
    # The "ingest" loop is the political cycle; find its ScheduledLoop among the definitions.
    return next(loop for loop, _fn in _loop_definitions(deps) if loop.name == "ingest")


def test_ingest_job_scheduled_only_when_interval_and_cycle_present(
    monkeypatch: pytest.MonkeyPatch, make_signal: SignalFactory
) -> None:
    # Real deps with a fake source -> political_cycle is wired. The ingest loop is enabled only when
    # a positive interval AND a cycle are both present (the lifespan schedules only enabled loops).
    _base_env(monkeypatch)
    monkeypatch.setenv("NEWS_FEED_URLS", '["https://example.test/feed.xml"]')
    deps = _build_with_fake_source(make_signal)
    assert deps.political_cycle is not None

    # Enabled branch: a positive interval with a cycle present -> the ingest loop is enabled.
    deps.ingest_interval_seconds = 60
    ingest = _ingest_loop(deps)
    assert ingest.enabled is True
    assert ingest.interval_seconds == 60

    # Disabled branch: zeroing the interval disables the ingest loop even though the cycle exists.
    deps.ingest_interval_seconds = 0
    assert _ingest_loop(deps).enabled is False


def test_ingested_signal_is_pushed_over_events_websocket(
    monkeypatch: pytest.MonkeyPatch, make_signal: SignalFactory
) -> None:
    # TestClient drives its own event loop, so this test stays SYNCHRONOUS: subscribe to
    # /ws/events first, then drive a real ingest via POST /ingest (open because API_KEY is
    # cleared) and assert the SIGNAL_RECEIVED event arrives on the socket.
    _base_env(monkeypatch)
    monkeypatch.setenv("NEWS_FEED_URLS", '["https://example.test/feed.xml"]')
    deps = _build_with_fake_source(make_signal)
    app = create_app(deps)

    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        response = client.post("/ingest")
        assert response.status_code == 200
        assert response.json()["signals_ingested"] == 2

        # The broadcaster fans every appended event to the subscriber in order; the two
        # SIGNAL_RECEIVED frames lead. Read a bounded number of frames to find one.
        event_types: list[str] = []
        for _ in range(10):
            frame = ws.receive_json()
            event_types.append(frame["event_type"])
            if frame["event_type"] == EventType.SIGNAL_RECEIVED:
                break

    assert EventType.SIGNAL_RECEIVED in event_types
    assert EventType.SIGNAL_RECEIVED == "signal.received"
