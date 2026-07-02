"""Postgres backfill dedupe store, backed by the ``backfill_seen`` table (migration 0004).

Keeps the ``(source_id, source_uid)`` pairs a backfill run has already ingested so re-running a
window double-appends nothing. The table is a dedicated dedupe index (not part of the CQRS read
model) and is rebuildable from the event log, since each signal's ``source_uid`` also lives in
its ``raw_payload``.
"""

from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from trading.infrastructure.persistence.tables import backfill_seen_table


class PostgresBackfillSeen:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def seen(self, source_id: str, source_uids: Sequence[str]) -> set[str]:
        if not source_uids:
            return set()
        query = select(backfill_seen_table.c.source_uid).where(
            backfill_seen_table.c.source_id == source_id,
            backfill_seen_table.c.source_uid.in_(list(source_uids)),
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {row.source_uid for row in rows}

    async def mark(self, source_id: str, source_uids: Iterable[str]) -> None:
        rows = [{"source_id": source_id, "source_uid": uid} for uid in source_uids]
        if not rows:
            return
        # Idempotent: a re-mark of an already-seen pair is a no-op (the PK already holds).
        stmt = (
            pg_insert(backfill_seen_table)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["source_id", "source_uid"])
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
