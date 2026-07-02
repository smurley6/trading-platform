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
            self._by_run[label_run_id] = list(labels)

    async def delete_run(self, label_run_id: str) -> None:
        async with self._lock:
            self._by_run.pop(label_run_id, None)

    async def count_for_run(self, label_run_id: str) -> int:
        async with self._lock:
            return len(self._by_run.get(label_run_id, ()))
