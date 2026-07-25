import logging
from collections.abc import Sequence
from datetime import datetime, timedelta

import httpx
import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.ingest_signals import IngestSignals
from trading.application.projections import signals_from_events
from trading.application.signal_recording import record_signals
from trading.domain.events import EventType, StoredEvent
from trading.domain.signals import Signal
from trading.infrastructure.persistence.memory import InMemoryEventStore


class FakeSource:
    def __init__(self, name: str, batches: list[list[Signal]]) -> None:
        self._name = name
        self._batches = batches
        self._call = 0

    @property
    def name(self) -> str:
        return self._name

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        batch = self._batches[min(self._call, len(self._batches) - 1)]
        self._call += 1
        return batch


class BrokenSource:
    @property
    def name(self) -> str:
        return "broken"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        raise RuntimeError("provider exploded")


async def test_ingest_records_signals_as_events(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    signal = make_signal(ticker="AAPL")
    ingest = IngestSignals(
        sources=[FakeSource("news", [[signal]])],
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
    )

    result = await ingest.execute()

    assert len(result.signals) == 1
    recorded = [
        s for s in await event_store.read_all() if s.event.event_type == EventType.SIGNAL_RECEIVED
    ]
    assert len(recorded) == 1
    # The raw payload is preserved so signals can be re-derived later (spec section 9).
    assert recorded[0].event.payload["raw_payload"] == {"source": "test-fixture"}


async def test_cursor_prevents_reingesting_the_same_signal(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    signal = make_signal(ticker="AAPL")
    ingest = IngestSignals(
        sources=[FakeSource("news", [[signal], [signal]])],  # same signal returned twice
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
    )

    first = await ingest.execute()
    second = await ingest.execute()

    assert len(first.signals) == 1
    assert len(second.signals) == 0  # cursor advanced past it


async def test_one_failing_source_does_not_stop_the_others(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    ingest = IngestSignals(
        sources=[BrokenSource(), FakeSource("news", [[make_signal(ticker="AAPL")]])],
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
    )

    result = await ingest.execute()

    assert result.failed_sources == ("broken",)
    assert len(result.signals) == 1


def _signal_events(recorded: Sequence[StoredEvent]) -> list[StoredEvent]:
    return [s for s in recorded if s.event.event_type == EventType.SIGNAL_RECEIVED]


async def test_persisted_cursor_survives_a_fresh_process(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # Two SEQUENTIAL IngestSignals over one shared store, each a FRESH instance (the daily
    # cron is a fresh process every run) with its OWN stub source returning the SAME batch —
    # the worst case that produced up to ~21 log copies (review F1). The second run must
    # append ZERO duplicates because it rehydrates the persisted cursor before polling.
    event_store = InMemoryEventStore(clock)
    signal = make_signal(ticker="AAPL")
    lookback_start = clock.now() - timedelta(days=21)

    first_run = IngestSignals(
        sources=[FakeSource("news", [[signal]])],
        event_store=event_store,
        clock=clock,
        lookback_start=lookback_start,
    )
    first = await first_run.execute()
    assert len(first.signals) == 1
    assert len(_signal_events(await event_store.read_all())) == 1

    # Brand-new instance + brand-new source, same shared event log = simulated next cron run.
    second_run = IngestSignals(
        sources=[FakeSource("news", [[signal]])],
        event_store=event_store,
        clock=clock,
        lookback_start=lookback_start,
    )
    second = await second_run.execute()
    assert len(second.signals) == 0  # cursor rehydrated from the log; nothing is newer
    assert len(_signal_events(await event_store.read_all())) == 1  # still exactly one copy


async def test_append_guard_refuses_a_second_copy_of_an_existing_signal(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # Defence in depth (opt-in guard on the live ingest path): re-recording an already-logged
    # signal appends nothing, so a re-scan can never grow a second copy — while the guard is
    # DEFAULT OFF so backfill/backtest/manual-push keep appending as before.
    event_store = InMemoryEventStore(clock)
    signal = make_signal(ticker="AAPL")
    for _ in range(3):
        await record_signals(
            event_store=event_store,
            clock=clock,
            signals=[signal],
            correlation_id="corr",
            skip_existing_streams=True,
        )
    assert len(_signal_events(await event_store.read_all())) == 1

    # Default OFF: the shared path still appends duplicates (backfill has its own dedupe).
    await record_signals(
        event_store=event_store,
        clock=clock,
        signals=[signal],
        correlation_id="corr",
    )
    assert len(_signal_events(await event_store.read_all())) == 2


async def test_recorded_signals_round_trip_through_projection(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    signal = make_signal(ticker="AAPL", sentiment=0.7)
    ingest = IngestSignals(
        sources=[FakeSource("news", [[signal]])],
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
    )
    await ingest.execute()

    reconstructed = signals_from_events(await event_store.read_all())
    assert len(reconstructed) == 1
    assert reconstructed[0].ticker == "AAPL"
    assert reconstructed[0].sentiment == 0.7


class KeyLeakingSource:
    """A source whose fetch raises an httpx error embedding an API key in the URL (review F22)."""

    _URL = "https://api.fda.gov/drug/drugsfda.json?search=foo&api_key=SUPER-SECRET-KEY"

    @property
    def name(self) -> str:
        return "fda_approvals"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        request = httpx.Request("GET", self._URL)
        response = httpx.Response(429, request=request, text="Too Many Requests")
        raise httpx.HTTPStatusError(
            f"Client error '429 Too Many Requests' for url '{self._URL}'\nFor more information...",
            request=request,
            response=response,
        )


async def test_source_error_log_redacts_the_api_key(
    clock: FixedClock, caplog: pytest.LogCaptureFixture
) -> None:
    """F22: a logged 429 from a key-in-URL source must NOT contain the key."""
    event_store = InMemoryEventStore(clock)
    ingest = IngestSignals(
        sources=[KeyLeakingSource()],
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
    )

    with caplog.at_level(logging.INFO):
        result = await ingest.execute()

    assert result.failed_sources == ("fda_approvals",)
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "SUPER-SECRET-KEY" not in combined
    assert combined  # something WAS logged (the failure is still diagnosable)
    assert "REDACTED" in combined
    assert "fda_approvals" in combined
