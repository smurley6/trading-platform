"""Outcome-labels port: the rebuildable research projection of event-study outcome labels.

Outcome labels (Task 03, ``docs/RESEARCH-PLAN.md`` §2-§3) are DERIVED, not truth: they live in
a rebuildable projection, NEVER the immutable event log. This port is the small write/read
surface the labeler run uses to persist its results, keeping the application layer free of any
concrete store (in-memory for dev/tests, Postgres in production; wired only in ``main.py``).

Unlike the CQRS read model (:mod:`trading.application.ports.read_model`), this table is NOT
folded from events on append — it is populated by an explicit labeler run and rebuilt by
re-running the labeler. ``replace_run`` gives rebuild semantics: a rerun of a given
``label_run_id`` deletes that run's rows first, then writes the fresh batch, so re-running never
duplicates (each run id is self-cleaning).

``latest_run_id``/``labels_for_run`` (Task 05, docs/RESEARCH-PLAN.md §2-§3) are the minimal READ
extension the diagnostic job needs; no migration was required (no `created_at` column exists on
the `outcome_labels` table). "Latest" is defined by insertion order (the surrogate `id` primary
key in Postgres; append order in memory) since a `label_run_id` is a UUID with no embedded
ordering of its own.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

# A label is "ok" when a forward abnormal return was computed, or "unresolvable" when required
# bars were missing (delisted / too recent / no data). Unresolvable labels are recorded WITH a
# reason (never silently dropped — RESEARCH-PLAN §3.4 survivorship awareness); the reason must be
# queryable, so it is a persisted column, not just a log line.
LabelStatus = Literal["ok", "unresolvable"]


@dataclass(frozen=True, slots=True)
class OutcomeLabel:
    """One (signal, horizon) outcome-study row, as persisted in the ``outcome_labels`` table."""

    signal_id: str
    ticker: str
    horizon_days: int
    # The forward abnormal return (ticker return - benchmark return) over the horizon window, as
    # a float. None when status is "unresolvable" (no return could be computed).
    abnormal_return: float | None
    # The benchmark the abnormal return was computed against: "sector:<ETF>" for a sector proxy
    # (e.g. "sector:XLK") or "market" for the SPY fallback. The Task-05 diagnostic splits on the
    # "sector:"/"market" prefix, so most non-directory tickers (SPY fallback) group together.
    benchmark: str
    status: LabelStatus
    # Set when status is "unresolvable" (e.g. "insufficient_forward_bars", "no_bars",
    # "benchmark_bars_missing"); None for an "ok" label.
    reason: str | None
    label_run_id: str


class OutcomeLabelsPort(Protocol):
    """Persist and query the derived outcome-labels projection (rebuildable, not an event fold)."""

    async def replace_run(self, label_run_id: str, labels: Sequence[OutcomeLabel]) -> None:
        """Atomically replace ``label_run_id``'s rows with ``labels`` (rerun = self-cleaning).

        Deletes any existing rows for ``label_run_id`` first, then inserts ``labels`` (all of
        which must carry that same run id). Re-running a labeler pass with the same run id thus
        overwrites rather than duplicates.
        """
        ...

    async def delete_run(self, label_run_id: str) -> None:
        """Delete every row for ``label_run_id`` (e.g. to discard a bad run)."""
        ...

    async def count_for_run(self, label_run_id: str) -> int:
        """The number of label rows currently stored for ``label_run_id``."""
        ...

    async def latest_run_id(self) -> str | None:
        """The most recently WRITTEN ``label_run_id`` (by insertion order), or ``None`` if empty.

        "Most recent" is insertion order, not a timestamp column (there is none) and not the
        `label_run_id` string's own ordering (it is an opaque UUID) — the last row written, by
        whichever run wrote it last (a rerun's `replace_run` counts as a fresh write for this
        purpose, since it deletes-then-reinserts that run's rows).
        """
        ...

    async def labels_for_run(self, label_run_id: str) -> Sequence[OutcomeLabel]:
        """Every label row stored for ``label_run_id`` (empty if the run id is unknown/empty)."""
        ...
