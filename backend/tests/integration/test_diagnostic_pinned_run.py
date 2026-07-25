"""Pinned-run regression: the episode evaluator reproduces the Phase-7 ex-LUV h=1 buy edge.

Acceptance test 5 (task-13). Against the READ-ONLY research corpus (label run
``a38d3848-f1eb-4a55-8a6d-15e8f1464f7a``), :func:`form4_episode_stats` must reproduce the
pre-registered primary cell — ex-LUV per-EPISODE h=1 = **127 episodes / 61.4% / +0.41%** — within
rounding, and the label-level pooled hit must be materially higher (the inflation the task fixes).

OPT-IN / read-only. This test needs the FULL research corpus, which the hermetic unit suite and the
``trading_test`` integration DB do not carry, so it is gated on its OWN env var
``PINNED_RUN_DATABASE_URL`` (point it at
``postgresql+asyncpg://trading:trading@localhost:55432/trading_research``). It is skipped
everywhere that var is unset — the normal unit suite and the ``TEST_DATABASE_URL`` integration run
stay green and hermetic. The test only READS (``all_signals`` / ``labels_for_run``); it never
writes, never mutates a label, never appends an event.
"""

import os

import pytest

from trading.application.diagnostic import form4_episode_stats

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PINNED_RUN_DATABASE_URL"),
        reason="PINNED_RUN_DATABASE_URL not set (research corpus not available)",
    ),
]

_RUN_ID = "a38d3848-f1eb-4a55-8a6d-15e8f1464f7a"


async def test_pinned_run_reproduces_ex_luv_episode_h1_edge() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres_outcome_labels import PostgresOutcomeLabels
    from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

    engine = create_async_engine(os.environ["PINNED_RUN_DATABASE_URL"])
    try:
        read_model = PostgresReadModel(engine)
        labels_store = PostgresOutcomeLabels(engine)
        signals = await read_model.all_signals()
        labels = await labels_store.labels_for_run(_RUN_ID)
    finally:
        await engine.dispose()

    assert labels, "pinned label run not found in the research corpus"

    rows = {(r.variant, r.horizon_days): r for r in form4_episode_stats(signals, labels)}
    ex_luv_h1 = rows[("ex_luv", 1)]

    # The pre-registered primary cell, within rounding.
    assert ex_luv_h1.n_episodes == 127
    assert ex_luv_h1.episode_hit_rate is not None
    assert round(ex_luv_h1.episode_hit_rate * 100, 1) == 61.4
    assert ex_luv_h1.episode_mean is not None
    assert round(ex_luv_h1.episode_mean * 100, 2) == 0.41

    # The whole point: the pooled LABEL-level hit is materially inflated above the episode hit.
    assert ex_luv_h1.label_hit_rate is not None
    assert ex_luv_h1.label_hit_rate > ex_luv_h1.episode_hit_rate

    # The decay narrative holds on the same episode set: by h=20 the edge is gone.
    ex_luv_h20 = rows[("ex_luv", 20)]
    assert ex_luv_h20.episode_hit_rate is not None
    assert ex_luv_h20.episode_hit_rate < 0.50
    assert ex_luv_h20.episode_median is not None
    assert ex_luv_h20.episode_median < 0.0
