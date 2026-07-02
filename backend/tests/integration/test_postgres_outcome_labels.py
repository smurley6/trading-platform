"""Postgres outcome-labels store integration tests (Task 03, migration 0005).

Require TEST_DATABASE_URL pointing at a Postgres with migrations applied (the integration job
runs `alembic upgrade head`). Prove the replace_run round-trip, delete_run, and the rerun
overwrite semantics (delete-then-write, so a rerun of a run id never duplicates) the labeler
depends on.
"""

import os
from uuid import uuid4

import pytest

from trading.application.ports.outcome_labels import OutcomeLabel
from trading.infrastructure.persistence.postgres_outcome_labels import PostgresOutcomeLabels

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL not set",
    ),
]


def _label(run_id: str, *, signal_id: str, horizon: int, status: str = "ok") -> OutcomeLabel:
    return OutcomeLabel(
        signal_id=signal_id,
        ticker="AAPL",
        horizon_days=horizon,
        abnormal_return=0.0123 if status == "ok" else None,
        benchmark="sector:XLK" if status == "ok" else "market",
        status="ok" if status == "ok" else "unresolvable",
        reason=None if status == "ok" else "insufficient_forward_bars",
        label_run_id=run_id,
    )


async def test_replace_run_round_trip() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_id = f"run-{uuid4()}"
    try:
        assert await store.count_for_run(run_id) == 0
        labels = [
            _label(run_id, signal_id="s1", horizon=1),
            _label(run_id, signal_id="s1", horizon=5),
            _label(run_id, signal_id="s1", horizon=20, status="unresolvable"),
        ]
        await store.replace_run(run_id, labels)
        assert await store.count_for_run(run_id) == 3
    finally:
        await store.delete_run(run_id)
        await engine.dispose()


async def test_rerun_overwrites_rather_than_duplicates() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_id = f"run-{uuid4()}"
    try:
        await store.replace_run(run_id, [_label(run_id, signal_id="s1", horizon=1)])
        assert await store.count_for_run(run_id) == 1
        # A rerun with the SAME run id (more rows) replaces, not appends.
        await store.replace_run(
            run_id,
            [
                _label(run_id, signal_id="s1", horizon=1),
                _label(run_id, signal_id="s1", horizon=5),
            ],
        )
        assert await store.count_for_run(run_id) == 2
    finally:
        await store.delete_run(run_id)
        await engine.dispose()


async def test_delete_run_removes_only_its_rows() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_a = f"run-a-{uuid4()}"
    run_b = f"run-b-{uuid4()}"
    try:
        await store.replace_run(run_a, [_label(run_a, signal_id="s1", horizon=1)])
        await store.replace_run(run_b, [_label(run_b, signal_id="s2", horizon=1)])
        await store.delete_run(run_a)
        assert await store.count_for_run(run_a) == 0
        assert await store.count_for_run(run_b) == 1
    finally:
        await store.delete_run(run_b)
        await engine.dispose()


async def test_replace_run_with_empty_clears() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_id = f"run-{uuid4()}"
    try:
        await store.replace_run(run_id, [_label(run_id, signal_id="s1", horizon=1)])
        await store.replace_run(run_id, [])  # empty rerun clears the run
        assert await store.count_for_run(run_id) == 0
    finally:
        await store.delete_run(run_id)
        await engine.dispose()
