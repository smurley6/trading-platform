"""System-screen status types (UI Phase D, docs/UI-DESIGN.md §5.5).

Plain, transport-agnostic descriptors for the operator console's System screen:

  * :class:`ScheduledLoop` / :class:`SchedulerState` — the scheduler/job-status surface. The app
    lifespan owns the actual APScheduler; it records each loop's last run and binds a next-run
    provider into a shared :class:`SchedulerState`, which the ``GET /system/status`` route reads.
    This keeps the interface free of any APScheduler type (the lifespan injects a plain callable).
  * :class:`DataSourceStatus` — a read-only mirror of one ingest adapter's config (enabled state +
    reliability prior). Built in the composition root (``main.py``), where the enable predicates and
    the per-adapter reliability constants already live, so this stays a mirror and never drifts.

All are decoupled from the Pydantic DTOs in :mod:`trading.interface.schemas` — the route maps these
into ``*Out`` models, so no domain/runtime type leaks onto the API surface.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ScheduledLoop:
    """One scheduled background loop's static definition (independent of whether it is running)."""

    name: str
    description: str
    interval_seconds: int
    # Whether this loop is actually scheduled: its interval is > 0 AND its dependency is wired.
    # Disabled loops are still reported (with their interval) so the operator sees the full set.
    enabled: bool


@dataclass
class _LoopRuntime:
    last_run: datetime | None = None
    # Injected by the lifespan from the live APScheduler job (``lambda: job.next_run_time``); None
    # until bound, and for loops that are not scheduled.
    next_run_provider: Callable[[], datetime | None] | None = None


@dataclass
class SchedulerState:
    """Mutable runtime status of the scheduled loops (last/next run), shared with the lifespan.

    Constructed in the composition root and stored on ``AppDeps``. The app lifespan records each
    run and binds a next-run provider; ``GET /system/status`` reads the snapshot. Empty/None values
    are the honest default before the scheduler starts (e.g. in tests, where all intervals are 0).
    """

    _runtime: dict[str, _LoopRuntime] = field(default_factory=dict)

    def _entry(self, name: str) -> _LoopRuntime:
        return self._runtime.setdefault(name, _LoopRuntime())

    def record_run(self, name: str, when: datetime) -> None:
        """Mark ``name`` as having just run at ``when`` (called by the lifespan's job wrapper)."""
        self._entry(name).last_run = when

    def bind_next_run(self, name: str, provider: Callable[[], datetime | None]) -> None:
        """Bind ``name``'s live next-run provider (the APScheduler job's ``next_run_time``)."""
        self._entry(name).next_run_provider = provider

    def last_run(self, name: str) -> datetime | None:
        entry = self._runtime.get(name)
        return entry.last_run if entry is not None else None

    def next_run(self, name: str) -> datetime | None:
        entry = self._runtime.get(name)
        if entry is None or entry.next_run_provider is None:
            return None
        return entry.next_run_provider()


@dataclass(frozen=True, slots=True)
class DataSourceStatus:
    """A read-only mirror of one ingest adapter's configuration (§5.5 Data sources pane).

    ``enabled`` mirrors the composition root's own enable predicate for the adapter (the same
    boolean that decides whether it is constructed), so the pane never drifts from env truth.
    ``reliability_prior`` is the adapter's ``source_reliability`` term used in conviction scoring.
    """

    source_id: str
    label: str
    enabled: bool
    reliability_prior: float
