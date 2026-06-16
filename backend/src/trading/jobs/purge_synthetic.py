"""Purge synthetic data so the running app is REAL-only: ``python -m trading.jobs.purge_synthetic``.

Directive: the platform should operate on REAL data ONLY. Anything SYNTHETIC (fabricated,
seeded, placeholder, or un-stamped — fail-closed; see :mod:`trading.domain.provenance`) must
stop driving the live app. It is fine for a source to go blank until real ingestion fills it;
the holes are reported so they are VISIBLE rather than silently hidden.

WHAT THIS TOUCHES (and what it deliberately does NOT):

  * The ``events`` log is the append-only system of record (the Postgres adapter even installs
    a trigger rejecting UPDATE/DELETE). We NEVER delete from it — history is preserved.
  * Instead we PURGE the SYNTHETIC signals from the CQRS read side (the ``rm_*`` projection
    tables / the in-memory read model) that every hot endpoint and the coverage report query.
    Mechanism: rebuild the read model from the event log with SYNTHETIC ``signal.received``
    events filtered OUT. Both ``InMemoryReadModel.rebuild`` and ``PostgresReadModel.rebuild``
    accept a ``Sequence[StoredEvent]`` and repopulate from scratch, so the rebuilt read side is
    real-only. Order fills / statuses / snapshots are replayed unchanged.
  * CAVEAT: because the event log is immutable, a later full ``rebuild`` from the UNFILTERED
    log would resurrect the synthetic signals. The durable real-only guarantee is the
    ingestion gate (``require_real_data``), which drops synthetic at the door going forward;
    this job cleans what already landed in the read side. To make the purge permanent at the
    log level you would compact/rewrite the event log (an offline migration) — out of scope
    here and intentionally not attempted against the append-only store.

USAGE:

  * Default is DRY-RUN: prints the current coverage (counts per source) and exactly what WOULD
    be removed, then exits 0 WITHOUT changing anything.
  * ``--confirm`` performs the purge, then prints a before/after summary and the post-purge
    coverage report (so the holes are visible) and exits 0.

Composition mirrors ``trading.jobs.cycle``: build the same dependencies, run, always tear down
the async resources (which disposes the Postgres engine).
"""

import asyncio
import logging
import sys
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from trading.application.data_coverage import CoverageReport, coverage_report
from trading.application.data_integrity import non_real_signals
from trading.application.projections import signals_from_events
from trading.config import Settings
from trading.domain.events import EventType, StoredEvent
from trading.domain.provenance import Provenance
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging

logger = logging.getLogger(__name__)


@runtime_checkable
class _RebuildableReadModel(Protocol):
    """The read-model capability this job needs beyond :class:`ReadModelPort`.

    Both concrete read models (in-memory and Postgres) implement ``rebuild`` — replaying a
    ``Sequence[StoredEvent]`` to repopulate every projection from scratch. We re-declare it here
    rather than edit the port; the runtime check fails loudly if a read model cannot rebuild.
    """

    async def rebuild(self, events: Sequence[StoredEvent]) -> None: ...


def _is_synthetic_signal_event(stored: StoredEvent) -> bool:
    """True iff this is a ``signal.received`` event carrying a non-REAL signal.

    Reuses the projection's payload decoding so legacy events (no ``provenance`` key) are
    treated as SYNTHETIC, exactly as the rest of the system does (fail-closed).
    """
    if stored.event.event_type != EventType.SIGNAL_RECEIVED:
        return False
    provenance = Provenance(str(stored.event.payload.get("provenance", Provenance.SYNTHETIC.value)))
    return provenance is not Provenance.REAL


def _filtered_events(events: Sequence[StoredEvent]) -> list[StoredEvent]:
    """The event log with SYNTHETIC signal events removed (everything else preserved)."""
    return [s for s in events if not _is_synthetic_signal_event(s)]


def _format_report(report: CoverageReport) -> str:
    """Render a CoverageReport as a human-readable block for the CLI summary."""
    lines = [
        f"  total REAL signals:      {report.total_real}",
        f"  total SYNTHETIC signals: {report.total_synthetic}",
        "  per source:",
    ]
    for cov in report.per_source:
        last = cov.last_real_at.isoformat() if cov.last_real_at is not None else "never"
        lines.append(
            f"    - {cov.source_type.value:<16} real={cov.real_count:<5} "
            f"synthetic={cov.synthetic_count:<5} last_real={last}"
        )
    if report.empty_sources:
        holes = ", ".join(s.value for s in report.empty_sources)
        lines.append(f"  HOLES (sources with NO real data yet): {holes}")
    else:
        lines.append("  HOLES: none — every source has real data")
    return "\n".join(lines)


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    print(text)  # noqa: T201 — summary output is the whole point of this command


async def run_purge(*, confirm: bool) -> int:
    """Build dependencies, report/purge synthetic data, tear down. Returns a process exit code.

    DRY-RUN (default) prints the current coverage and what would be removed and returns 0.
    With ``confirm=True`` it rebuilds the read side without the synthetic signals, prints a
    before/after summary plus the post-purge coverage, and returns 0. Any failure propagates to
    :func:`main`, which maps it to a non-zero exit; teardown always runs via ``finally``.
    """
    settings = Settings()
    configure_logging(settings.log_level)
    deps = build_dependencies(settings)
    try:
        events = list(await deps.event_store.read_all())
        before = signals_from_events(events)
        before_report = coverage_report(before)
        to_purge = non_real_signals(before)

        _emit("=== Synthetic-data purge ===")
        _emit(f"mode: {'CONFIRM (will purge)' if confirm else 'DRY-RUN (no changes)'}")
        _emit("")
        _emit("BEFORE:")
        _emit(_format_report(before_report))
        _emit("")
        _emit(f"SYNTHETIC signals identified for purge: {len(to_purge)}")

        if not confirm:
            _emit("")
            _emit("DRY-RUN: nothing was changed. Re-run with --confirm to purge.")
            return 0

        read_model = deps.read_model
        if not isinstance(read_model, _RebuildableReadModel):
            raise RuntimeError(
                "configured read model does not support rebuild(); cannot purge the read side"
            )
        # Rebuild the read side from the log with SYNTHETIC signal events filtered out. The
        # append-only event log is left fully intact (system of record); only the projections
        # the running app reads are made real-only.
        await read_model.rebuild(_filtered_events(events))

        after = list(await deps.read_model.all_signals())
        after_report = coverage_report(after)
        remaining_fake = non_real_signals(after)

        _emit("")
        _emit("AFTER (read-model / live view):")
        _emit(_format_report(after_report))
        _emit("")
        _emit(
            f"purged {len(to_purge)} synthetic signal(s); "
            f"read side now has {after_report.total_real} real, "
            f"{after_report.total_synthetic} synthetic"
        )
        if remaining_fake:
            raise RuntimeError(
                f"purge incomplete: {len(remaining_fake)} synthetic signal(s) still in the "
                "read model after rebuild"
            )
        _emit("read side is now REAL-only.")
        return 0
    finally:
        await _teardown(deps)


async def _teardown(deps: AppDeps) -> None:
    """Release every async resource the composition root registered (mirrors app shutdown).

    Each acloser is awaited independently so one failing close cannot strand the others (e.g.
    the Postgres pool is still disposed even if an HTTP client close raised).
    """
    for aclose in deps.aclosers:
        try:
            await aclose()
        except Exception:
            logger.exception("error closing a resource during job teardown")


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entrypoint: run the purge and return a process exit code.

    ``--confirm`` (anywhere in argv) performs the deletion; otherwise it is a dry run. Returns 0
    on success and 1 on any unhandled failure (logged before returning). Wired to ``sys.exit``
    under ``__main__`` so ``python -m trading.jobs.purge_synthetic`` exits with this code.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    confirm = "--confirm" in args
    try:
        return asyncio.run(run_purge(confirm=confirm))
    except Exception:
        logger.exception("synthetic-data purge job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
