"""Postgres outcome-labels store integration tests (Task 03, migration 0005; Task 05 read ext).

Require TEST_DATABASE_URL pointing at a Postgres with migrations applied (the integration job
runs `alembic upgrade head`). Prove the replace_run round-trip, delete_run, and the rerun
overwrite semantics (delete-then-write, so a rerun of a run id never duplicates) the labeler
depends on, plus (Task 05) `latest_run_id`/`labels_for_run`, the minimal read extension the
diagnostic job needs (no migration: "latest" orders by the surrogate `id` primary key, since
there is no `created_at` column).
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


async def test_latest_run_id_is_the_most_recently_inserted_row() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_a = f"run-a-{uuid4()}"
    run_b = f"run-b-{uuid4()}"
    try:
        await store.replace_run(run_a, [_label(run_a, signal_id="s1", horizon=1)])
        await store.replace_run(run_b, [_label(run_b, signal_id="s2", horizon=1)])
        # run_b was inserted after run_a -> latest, regardless of the run ids' string ordering.
        assert await store.latest_run_id() == run_b
    finally:
        await store.delete_run(run_a)
        await store.delete_run(run_b)
        await engine.dispose()


async def test_latest_run_id_rerun_of_earlier_run_becomes_latest_again() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_a = f"run-a-{uuid4()}"
    run_b = f"run-b-{uuid4()}"
    try:
        await store.replace_run(run_a, [_label(run_a, signal_id="s1", horizon=1)])
        await store.replace_run(run_b, [_label(run_b, signal_id="s2", horizon=1)])
        assert await store.latest_run_id() == run_b
        # Re-running run_a re-inserts its rows (delete-then-write), making it latest again.
        await store.replace_run(run_a, [_label(run_a, signal_id="s1", horizon=1)])
        assert await store.latest_run_id() == run_a
    finally:
        await store.delete_run(run_a)
        await store.delete_run(run_b)
        await engine.dispose()


async def test_labels_for_run_round_trips_and_empty_for_unknown_run() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresOutcomeLabels(engine)
    run_id = f"run-{uuid4()}"
    try:
        labels = [
            _label(run_id, signal_id="s1", horizon=1),
            _label(run_id, signal_id="s1", horizon=5, status="unresolvable"),
        ]
        await store.replace_run(run_id, labels)
        result = {
            (label.signal_id, label.horizon_days): label
            for label in await store.labels_for_run(run_id)
        }
        assert result[("s1", 1)].status == "ok"
        assert result[("s1", 1)].abnormal_return == 0.0123
        assert result[("s1", 5)].status == "unresolvable"
        assert result[("s1", 5)].reason == "insufficient_forward_bars"

        assert await store.labels_for_run(f"unknown-{uuid4()}") == []
    finally:
        await store.delete_run(run_id)
        await engine.dispose()
