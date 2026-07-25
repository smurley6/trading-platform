"""In-memory outcome-labels store: the UI Phase D read methods (latest_run_id, labels_for_signals).

The write/replace/count semantics are covered by the Postgres integration tests and the labeler
use-case tests; this file covers the two read methods the Signals research surface (§5.5) added,
plus the parity contract that a rerun re-orders a run to "latest".
"""

from trading.application.ports.outcome_labels import OutcomeLabel
from trading.infrastructure.persistence.memory_outcome_labels import InMemoryOutcomeLabels


def _label(run_id: str, *, signal_id: str, horizon: int) -> OutcomeLabel:
    return OutcomeLabel(
        signal_id=signal_id,
        ticker="AAPL",
        horizon_days=horizon,
        abnormal_return=0.0123,
        benchmark="sector:XLK",
        status="ok",
        reason=None,
        label_run_id=run_id,
    )


async def test_latest_run_id_is_none_when_empty() -> None:
    store = InMemoryOutcomeLabels()
    assert await store.latest_run_id() is None


async def test_latest_run_id_tracks_most_recent_write() -> None:
    store = InMemoryOutcomeLabels()
    await store.replace_run("run-a", [_label("run-a", signal_id="s1", horizon=1)])
    await store.replace_run("run-b", [_label("run-b", signal_id="s2", horizon=1)])
    assert await store.latest_run_id() == "run-b"
    # A rerun of an OLDER run makes it the latest again (parity with the Postgres id ordering).
    await store.replace_run("run-a", [_label("run-a", signal_id="s1", horizon=5)])
    assert await store.latest_run_id() == "run-a"


async def test_labels_for_signals_scopes_to_run_and_ids() -> None:
    store = InMemoryOutcomeLabels()
    await store.replace_run(
        "run-a",
        [
            _label("run-a", signal_id="s1", horizon=1),
            _label("run-a", signal_id="s1", horizon=5),
            _label("run-a", signal_id="s2", horizon=1),
        ],
    )
    await store.replace_run("run-b", [_label("run-b", signal_id="s1", horizon=1)])

    got = await store.labels_for_signals(["s1"], label_run_id="run-a")
    assert {(label.signal_id, label.horizon_days) for label in got} == {("s1", 1), ("s1", 5)}
    # Wrong run id yields nothing for s1 in run-a's absence there.
    assert await store.labels_for_signals(["s2"], label_run_id="run-b") == []
    # Empty request short-circuits.
    assert await store.labels_for_signals([], label_run_id="run-a") == []
