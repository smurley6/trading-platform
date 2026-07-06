"""In-memory outcome-labels store: `latest_run_id`/`labels_for_run` read extension (Task 05).

Covers the new read methods added alongside the existing `replace_run`/`delete_run`/
`count_for_run` (Task 03): empty-store `latest_run_id` is None, insertion order (not label-run-id
string order) determines "latest", a RERUN of an existing run id counts as a fresh write (moves
it back to "latest"), `delete_run` removes a run from latest-selection, and `labels_for_run`
round-trips exactly what was written (including returning `()` for an unknown run id).
"""

from trading.application.ports.outcome_labels import OutcomeLabel
from trading.infrastructure.persistence.memory_outcome_labels import InMemoryOutcomeLabels


def _label(run_id: str, signal_id: str) -> OutcomeLabel:
    return OutcomeLabel(
        signal_id=signal_id,
        ticker="AAPL",
        horizon_days=1,
        abnormal_return=0.01,
        benchmark="market",
        status="ok",
        reason=None,
        label_run_id=run_id,
    )


async def test_latest_run_id_none_when_empty() -> None:
    store = InMemoryOutcomeLabels()
    assert await store.latest_run_id() is None


async def test_latest_run_id_is_most_recently_written_not_alphabetical() -> None:
    store = InMemoryOutcomeLabels()
    await store.replace_run("run-z", [_label("run-z", "s1")])
    await store.replace_run("run-a", [_label("run-a", "s2")])  # written AFTER "run-z"
    assert await store.latest_run_id() == "run-a"


async def test_rerun_of_existing_run_id_counts_as_a_fresh_write() -> None:
    store = InMemoryOutcomeLabels()
    await store.replace_run("run-a", [_label("run-a", "s1")])
    await store.replace_run("run-b", [_label("run-b", "s2")])
    assert await store.latest_run_id() == "run-b"
    # Re-running run-a (even with the same content) makes it the latest write again.
    await store.replace_run("run-a", [_label("run-a", "s1")])
    assert await store.latest_run_id() == "run-a"


async def test_delete_run_removes_it_from_latest_selection() -> None:
    store = InMemoryOutcomeLabels()
    await store.replace_run("run-a", [_label("run-a", "s1")])
    await store.replace_run("run-b", [_label("run-b", "s2")])
    await store.delete_run("run-b")
    assert await store.latest_run_id() == "run-a"


async def test_labels_for_run_round_trips() -> None:
    store = InMemoryOutcomeLabels()
    labels = [_label("run-a", "s1"), _label("run-a", "s2")]
    await store.replace_run("run-a", labels)
    result = await store.labels_for_run("run-a")
    assert list(result) == labels


async def test_labels_for_run_empty_for_unknown_run_id() -> None:
    store = InMemoryOutcomeLabels()
    assert await store.labels_for_run("does-not-exist") == []
