"""Opt-in compare-and-set append (review F9) on the in-memory store.

`expected_version` is the fill writers' optimistic-concurrency lever: an append at a stale
version is rejected (`ExpectedVersionError`) instead of silently reallocating a fresh seq.
The default path (no `expected_version`) keeps its retry-on-lost-race behaviour, so the
append-only seq-allocation invariant for normal writers is unchanged.
"""

from datetime import UTC, datetime

import pytest

from tests.conftest import FixedClock
from trading.application.ports.event_store import ExpectedVersionError
from trading.domain.events import DomainEvent
from trading.infrastructure.persistence.memory import InMemoryEventStore

_STREAM = "order-abc"


def _event() -> DomainEvent:
    return DomainEvent(
        stream_id=_STREAM,
        event_type="order.filled",
        payload={"quantity": "1"},
        occurred_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
        correlation_id="corr-1",
    )


async def test_cas_append_succeeds_at_the_expected_version(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    first = await store.append(_event())  # normal append -> seq 1
    # Stream version is now 1; a CAS append expecting 1 lands at seq 2.
    second = await store.append(_event(), expected_version=first.seq)
    assert (first.seq, second.seq) == (1, 2)


async def test_cas_append_conflicts_when_the_stream_advanced(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    # Two writers both observed version 0 and both try to write the first event.
    await store.append(_event(), expected_version=0)
    with pytest.raises(ExpectedVersionError):
        await store.append(_event(), expected_version=0)


async def test_default_append_never_raises_on_contention(clock: FixedClock) -> None:
    # Without expected_version the store allocates the next seq for every call (the normal-writer
    # contract): concurrent-looking appends all succeed with gap-free seqs, no conflict surfaced.
    store = InMemoryEventStore(clock)
    seqs = [(await store.append(_event())).seq for _ in range(3)]
    assert seqs == [1, 2, 3]
