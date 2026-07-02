"""Backfill historical signals from ONE source over a date window (Task 02).

Reconstructs real historical signals (Provenance.BACKFILLED, point-in-time disclosure
timestamps) from a source's archive and records them as ``signal.received`` events. This is the
research program's ingestion arm (docs/RESEARCH-PLAN.md §2-§4); backfilled signals feed the
labeler/learner but never order placement (Task 01's execution wall).

USAGE::

    python -m trading.jobs.backfill --source sec_form4 --start 2024-07-01 --end 2026-07-01
    python -m trading.jobs.backfill --source sec_form4 --start 2024-07-01 --end 2026-07-01 --confirm

  * DRY-RUN (default): fetch the window and report what WOULD be ingested (per-source counts —
    fetched / new / duplicate / refused) WITHOUT writing anything. Always allowed.
  * ``--confirm``: append ``signal.received`` events (one correlation id for the run) + mark the
    source-native ids seen + record a ``user.action`` event with the run parameters and counts.
    Requires ``BACKFILL_ENABLED=true``; a window whose span exceeds the source's max-lookback
    guard is refused.

Composition mirrors ``trading.jobs.cycle`` / ``trading.jobs.purge_synthetic``: build the same
dependencies via the composition root, run, and ALWAYS tear down the async resources (disposing
the Postgres engine). Exit 0 on a clean run, 1 on any failure.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from trading.application.backfill_signals import (
    BackfillError,
    BackfillReport,
    BackfillSignals,
    supported_sources,
)
from trading.config import Settings
from trading.domain.events import DomainEvent, EventType
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging

logger = logging.getLogger(__name__)

_USER_ACTION_STREAM = "user-actions"


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    print(text)  # noqa: T201 — summary output is the whole point of this command


def _parse_date(value: str) -> datetime:
    """Parse an ISO date (or datetime) as a UTC-aware datetime for the window bounds."""
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.backfill",
        description="Backfill historical signals from one source over a date window.",
    )
    parser.add_argument("--source", required=True, help="source id, e.g. sec_form4")
    parser.add_argument("--start", required=True, help="window start (ISO date, inclusive)")
    parser.add_argument("--end", required=True, help="window end (ISO date, exclusive)")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="write events (default is a dry run); requires BACKFILL_ENABLED=true",
    )
    return parser.parse_args(list(argv))


def _format_report(report: BackfillReport, *, dry_run: bool) -> str:
    lines = [
        f"  source:    {report.source_id}",
        f"  window:    {report.start.isoformat()} .. {report.end.isoformat()}",
        f"  fetched:   {report.fetched}",
        f"  new:       {report.new}",
        f"  duplicate: {report.duplicate}",
        f"  refused:   {report.refused}  (no source_uid — cannot dedupe, not recorded)",
    ]
    if dry_run:
        lines.append("  committed: NO (dry-run)")
    else:
        lines.append(f"  committed: {'YES' if report.committed else 'NO (nothing new)'}")
        if report.correlation_id is not None:
            lines.append(f"  correlation_id: {report.correlation_id}")
    return "\n".join(lines)


async def run_backfill(*, source_id: str, start: datetime, end: datetime, confirm: bool) -> int:
    """Build dependencies, backfill the window (or dry-run), tear down. Returns an exit code.

    Guards, in order: the window must be non-empty; ``--confirm`` requires BACKFILL_ENABLED; the
    source must be a wired historical source; a ``--confirm`` window may not exceed the source's
    max-lookback guard. Any guard failure returns non-zero with a clear message. Teardown always
    runs via ``finally``.
    """
    settings = Settings()
    configure_logging(settings.log_level)

    if end <= start:
        _emit(f"error: --end ({end.isoformat()}) must be after --start ({start.isoformat()}).")
        return 1

    deps = build_dependencies(settings)
    try:
        source = deps.historical_sources.get(source_id)
        if source is None:
            supported = supported_sources(list(deps.historical_sources.values()))
            listed = ", ".join(supported) if supported else "(none wired yet)"
            _emit(
                f"error: unknown/unsupported --source '{source_id}'. "
                f"Supported historical sources: {listed}."
            )
            return 1

        if confirm and not settings.backfill_enabled:
            _emit(
                "error: --confirm requires BACKFILL_ENABLED=true (a dry-run is always allowed). "
                "Set the env var to write events."
            )
            return 1

        if confirm:
            cap_days = settings.backfill_max_lookback_days(source_id)
            span_days = (end - start).days
            if cap_days is not None and span_days > cap_days:
                _emit(
                    f"error: window span {span_days}d exceeds the max-lookback guard for "
                    f"'{source_id}' ({cap_days}d). Narrow the window or raise "
                    f"BACKFILL_{source_id.upper()}_MAX_LOOKBACK_DAYS."
                )
                return 1

        if deps.backfill_seen is None:  # defensive: the composition root always wires this
            raise RuntimeError("backfill dedupe store is not configured")

        backfill = BackfillSignals(
            source=source,
            event_store=deps.event_store,
            seen=deps.backfill_seen,
            clock=deps.clock,
        )
        correlation_id = str(uuid4())
        report = await backfill.execute(
            start=start, end=end, correlation_id=correlation_id, dry_run=not confirm
        )

        _emit("=== Backfill ===")
        _emit(f"mode: {'CONFIRM (writes events)' if confirm else 'DRY-RUN (no changes)'}")
        _emit("")
        _emit(_format_report(report, dry_run=not confirm))

        if not confirm:
            _emit("")
            _emit("DRY-RUN: nothing was written. Re-run with --confirm to record.")
            return 0

        # A user.action event records the run parameters + counts + active filters, ONLY on a
        # --confirm run (dry runs write nothing). One event per run for auditability.
        await deps.event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": "backfill",
                    "source": source_id,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "counts": {
                        "fetched": report.fetched,
                        "new": report.new,
                        "duplicate": report.duplicate,
                        "refused": report.refused,
                    },
                    "filters": {},
                },
                occurred_at=deps.clock.now(),
                correlation_id=correlation_id,
            )
        )
        _emit("")
        _emit(f"recorded {report.new} new backfilled signal(s) for '{source_id}'.")
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
    """Synchronous entrypoint: parse args, run the backfill, return a process exit code.

    Returns 0 on success (including a clean dry run) and 1 on any failure — a bad window, a
    disallowed --confirm, an unknown source, or an unhandled exception (logged before returning).
    """
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = _parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    try:
        start = _parse_date(args.start)
        end = _parse_date(args.end)
    except ValueError as exc:
        _emit(f"error: could not parse --start/--end as ISO dates ({exc}).")
        return 1
    try:
        return asyncio.run(
            run_backfill(source_id=args.source, start=start, end=end, confirm=bool(args.confirm))
        )
    except BackfillError as exc:
        _emit(f"error: {exc}")
        return 1
    except Exception:
        logger.exception("backfill job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
