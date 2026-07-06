"""Postgres outcome-labels store, backed by the ``outcome_labels`` table (migration 0005).

Persists the derived event-study outcome labels a labeler run computes. This is a REBUILDABLE
RESEARCH PROJECTION, not part of the CQRS event-fold read model: it is written by an explicit
labeler run and rebuilt by re-running the labeler. ``replace_run`` deletes the run's existing
rows then inserts the fresh batch inside one transaction, so a rerun of a given ``label_run_id``
overwrites rather than duplicates (each run id is self-cleaning).
"""

from collections.abc import Sequence

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from trading.application.ports.outcome_labels import OutcomeLabel
from trading.infrastructure.persistence.tables import outcome_labels_table

# `latest_run_id` (Task 05) orders by this surrogate identity column, NOT a `created_at` column
# (there isn't one — no migration was added for the diagnostic's read extension). Insertion
# order via `MAX(id)` is the "most recently written" row; if that run's most recent write was an
# empty `replace_run` (a rerun that produced zero rows), it has no rows left and so cannot be the
# max — this mirrors "a run with no rows can't be considered latest" rather than tracking writes
# independent of rows, which is an accepted, documented limitation (see the port docstring).
_LATEST_RUN_ID_QUERY = (
    select(outcome_labels_table.c.label_run_id).order_by(outcome_labels_table.c.id.desc()).limit(1)
)


class PostgresOutcomeLabels:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def replace_run(self, label_run_id: str, labels: Sequence[OutcomeLabel]) -> None:
        rows = [
            {
                "signal_id": label.signal_id,
                "ticker": label.ticker,
                "horizon_days": label.horizon_days,
                "abnormal_return": label.abnormal_return,
                "benchmark": label.benchmark,
                "status": label.status,
                "reason": label.reason,
                "label_run_id": label.label_run_id,
            }
            for label in labels
        ]
        async with self._engine.begin() as conn:
            # Delete-then-write in one transaction: a rerun of this run id overwrites, never
            # duplicates, and a partial failure rolls back to the prior state.
            await conn.execute(
                delete(outcome_labels_table).where(
                    outcome_labels_table.c.label_run_id == label_run_id
                )
            )
            if rows:
                await conn.execute(insert(outcome_labels_table), rows)

    async def delete_run(self, label_run_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                delete(outcome_labels_table).where(
                    outcome_labels_table.c.label_run_id == label_run_id
                )
            )

    async def count_for_run(self, label_run_id: str) -> int:
        query = select(func.count()).where(outcome_labels_table.c.label_run_id == label_run_id)
        async with self._engine.connect() as conn:
            return int((await conn.execute(query)).scalar_one())

    async def latest_run_id(self) -> str | None:
        """The `label_run_id` of the highest surrogate `id` row (most recently inserted)."""
        async with self._engine.connect() as conn:
            result = await conn.execute(_LATEST_RUN_ID_QUERY)
            row = result.first()
            return None if row is None else str(row[0])

    async def labels_for_run(self, label_run_id: str) -> Sequence[OutcomeLabel]:
        query = select(outcome_labels_table).where(
            outcome_labels_table.c.label_run_id == label_run_id
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).mappings().all()
        return [
            OutcomeLabel(
                signal_id=row["signal_id"],
                ticker=row["ticker"],
                horizon_days=row["horizon_days"],
                abnormal_return=row["abnormal_return"],
                benchmark=row["benchmark"],
                status=row["status"],
                reason=row["reason"],
                label_run_id=row["label_run_id"],
            )
            for row in rows
        ]
