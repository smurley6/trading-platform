"""Event store port: append-only system of record (spec section 9)."""

from collections.abc import Sequence
from typing import Protocol

from trading.domain.events import DomainEvent, StoredEvent


class ExpectedVersionError(Exception):
    """A compare-and-set append (an `expected_version` was supplied) found the target stream
    already advanced past that version — another writer appended first.

    This is the single-record concurrency signal the fill writers rely on (review F9): the
    caller MUST re-read the log and re-decide (the racing writer may have recorded the same
    fact), NOT blindly retry with a fresh sequence number. Only raised on the opt-in CAS path;
    the default `append` (no `expected_version`) never raises it.
    """


class EventStorePort(Protocol):
    async def append(
        self, event: DomainEvent, *, expected_version: int | None = None
    ) -> StoredEvent:
        """Durably append one event. Implementations MUST never update or delete.

        `expected_version` is OPT-IN optimistic concurrency for the fill writers (review F9).
        When ``None`` (every existing caller) the store allocates the next per-stream sequence,
        retrying on a lost race so a normal append always succeeds — the append-only
        seq-allocation invariant is unchanged. When given, it is the stream's max `seq` the
        caller last observed; the event is inserted at exactly ``expected_version + 1`` and, if
        that slot is already taken, `ExpectedVersionError` is raised WITHOUT retry so the
        caller can re-read rather than double-record.
        """
        ...

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]: ...

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]: ...

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        """All events in recording order; `limit` returns the most recent N."""
        ...
