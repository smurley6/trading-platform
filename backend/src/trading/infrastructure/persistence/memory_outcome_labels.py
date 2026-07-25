"""In-memory outcome-labels store (default / test path).

Mirrors :class:`~trading.infrastructure.persistence.postgres_outcome_labels.PostgresOutcomeLabels`
so the labeler behaves identically with or without a database configured. Keyed by
``label_run_id`` so ``replace_run`` is a self-cleaning overwrite (rerun never duplicates).
"""

import asyncio
from collections.abc import Sequence

from trading.application.ports.outcome_labels import OutcomeLabel


class InMemoryOutcomeLabels:
    """Stores derived outcome labels per label run, in memory (rebuildable, not an event fold)."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._by_run: dict[str, list[OutcomeLabel]] = {}

    async def replace_run(self, label_run_id: str, labels: Sequence[OutcomeLabel]) -> None:
        async with self._lock:
            # Delete-then-write: a rerun of the same run id overwrites rather than duplicates.
            # Pop first so the re-added key moves to the END of the dict, mirroring the Postgres
            # store where a rerun's rows get fresh (highest) identity ids — `latest_run_id` then
            # returns the most-recently-written run in either backend.
            self._by_run.pop(label_run_id, None)
            self._by_run[label_run_id] = list(labels)

    async def delete_run(self, label_run_id: str) -> None:
        async with self._lock:
            self._by_run.pop(label_run_id, None)

    async def count_for_run(self, label_run_id: str) -> int:
        async with self._lock:
            return len(self._by_run.get(label_run_id, ()))

    async def latest_run_id(self) -> str | None:
        async with self._lock:
            if not self._by_run:
                return None
            # Dict preserves insertion order; replace_run re-adds at the end, so the last key is
            # the most recently written run.
            return next(reversed(self._by_run))

    async def labels_for_signals(
        self, signal_ids: Sequence[str], *, label_run_id: str
    ) -> Sequence[OutcomeLabel]:
        wanted = set(signal_ids)
        async with self._lock:
            return [
                label for label in self._by_run.get(label_run_id, ()) if label.signal_id in wanted
            ]
