"""Backfill dedupe port: the source-native ids already ingested by a backfill run.

Idempotency for backfill must cover the EVENT LOG, not just the read model: re-running a
window must not double-append ``signal.received`` events, or the projections and the future
labeler would double-count. So :class:`~trading.application.backfill_signals.BackfillSignals`
consults this port BEFORE appending, keyed on the source-native id
(``(source_id, source_uid)`` — EDGAR accession number, House DocID + row, …).

The dedupe store is a small dedicated ``backfill_seen (source_id, source_uid)`` table (chosen
over widening ``rm_signals`` so the read model's shape is untouched and the seen-check is an
O(1) primary-key lookup on the source-native id). ``source_uid`` is also carried inside each
signal's ``raw_payload`` so the table is fully rebuildable from the append-only event log.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol


class BackfillSeenPort(Protocol):
    """Records and queries the ``(source_id, source_uid)`` pairs a backfill run has ingested."""

    async def seen(self, source_id: str, source_uids: Sequence[str]) -> set[str]:
        """Return the subset of ``source_uids`` already recorded for ``source_id``."""
        ...

    async def mark(self, source_id: str, source_uids: Iterable[str]) -> None:
        """Record ``source_uids`` as ingested for ``source_id`` (idempotent on re-mark)."""
        ...
