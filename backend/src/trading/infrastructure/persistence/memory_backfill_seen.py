"""In-memory backfill dedupe store (default / test path).

Mirrors :class:`~trading.infrastructure.persistence.postgres_backfill_seen.PostgresBackfillSeen`
so the backfill use case behaves identically with or without a database configured.
"""

import asyncio
from collections.abc import Iterable, Sequence


class InMemoryBackfillSeen:
    """Tracks the ``(source_id, source_uid)`` pairs a backfill run has ingested, in memory."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._seen: dict[str, set[str]] = {}

    async def seen(self, source_id: str, source_uids: Sequence[str]) -> set[str]:
        async with self._lock:
            known = self._seen.get(source_id, set())
            return {uid for uid in source_uids if uid in known}

    async def mark(self, source_id: str, source_uids: Iterable[str]) -> None:
        async with self._lock:
            self._seen.setdefault(source_id, set()).update(source_uids)
