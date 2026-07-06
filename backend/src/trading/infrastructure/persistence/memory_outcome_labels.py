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
        # Insertion order of `replace_run` calls (Task 05). A plain dict preserves FIRST-insertion
        # order and does NOT move a re-inserted key to the end, so this list is the authoritative
        # "most recently written" record `latest_run_id` needs; a rerun of the same run id counts
        # as a fresh write, matching the Postgres impl's insertion-order (surrogate `id`) semantics.
        self._write_order: list[str] = []

    async def replace_run(self, label_run_id: str, labels: Sequence[OutcomeLabel]) -> None:
        async with self._lock:
            # Delete-then-write: a rerun of the same run id overwrites rather than duplicates.
            self._by_run[label_run_id] = list(labels)
            if label_run_id in self._write_order:
                self._write_order.remove(label_run_id)
            self._write_order.append(label_run_id)

    async def delete_run(self, label_run_id: str) -> None:
        async with self._lock:
            self._by_run.pop(label_run_id, None)
            if label_run_id in self._write_order:
                self._write_order.remove(label_run_id)

    async def count_for_run(self, label_run_id: str) -> int:
        async with self._lock:
            return len(self._by_run.get(label_run_id, ()))

    async def latest_run_id(self) -> str | None:
        """The most recently written run id (Task 05), or ``None`` if the store is empty."""
        async with self._lock:
            return self._write_order[-1] if self._write_order else None

    async def labels_for_run(self, label_run_id: str) -> Sequence[OutcomeLabel]:
        """All label rows for ``label_run_id`` (Task 05 read extension)."""
        async with self._lock:
            return list(self._by_run.get(label_run_id, ()))
