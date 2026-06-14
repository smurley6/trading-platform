from collections.abc import Sequence
from datetime import datetime, timedelta

from tests.conftest import FixedClock, SignalFactory
from trading.application.ingest_signals import IngestSignals
from trading.application.projections import signals_from_events
from trading.domain.events import EventType
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
