import asyncio

from tests.conftest import FixedClock
from trading.domain.events import DomainEvent
from trading.infrastructure.persistence.broadcasting import BroadcastingEventStore
from trading.infrastructure.persistence.memory import InMemoryEventStore


def _event(clock: FixedClock, stream: str = "s1") -> DomainEvent:
    return DomainEvent(
        stream_id=stream,
        event_type="test.event",
        payload={"n": 1},
        occurred_at=clock.now(),
        correlation_id="corr-1",
    )


async def test_subscriber_receives_appended_events(clock: FixedClock) -> None:
    store = BroadcastingEventStore(InMemoryEventStore(clock))
    queue = store.subscribe()

    stored = await store.append(_event(clock))

    received = await asyncio.wait_for(queue.get(), timeout=1)
    assert received.event.event_id == stored.event.event_id
    assert received.seq == 1


async def test_append_still_persists_to_inner_store(clock: FixedClock) -> None:
    inner = InMemoryEventStore(clock)
    store = BroadcastingEventStore(inner)

    await store.append(_event(clock))

    assert len(await inner.read_all()) == 1


async def test_unsubscribe_stops_delivery(clock: FixedClock) -> None:
    store = BroadcastingEventStore(InMemoryEventStore(clock))
    queue = store.subscribe()
    store.unsubscribe(queue)

    await store.append(_event(clock))

    assert queue.empty()
