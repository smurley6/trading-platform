"""Label outcome event-study returns for ticker-resolved signals (Task 03).

Computes forward abnormal returns (1/5/20 trading days from the first market session AFTER each
signal's disclosure timestamp) for every REAL+BACKFILLED ticker-resolved signal, sector-relative
where the sector map resolves the ticker and SPY-relative otherwise, and writes them to the
rebuildable ``outcome_labels`` research projection (docs/RESEARCH-PLAN.md §2-§3). Labels are
DERIVED, not truth — they never enter the immutable event log.

USAGE::

    python -m trading.jobs.label_outcomes
    python -m trading.jobs.label_outcomes --confirm
    python -m trading.jobs.label_outcomes --exclude-tickers NAN
    python -m trading.jobs.label_outcomes --exclude-tickers NAN,XYZ --confirm

  * DRY-RUN (default): compute labels and report the ok / unresolvable split WITHOUT writing.
    Always allowed; may run against whatever historical-prices provider is wired (incl. synthetic).
  * ``--confirm``: replace this run's rows in the outcome_labels projection (delete-then-write,
    so a rerun overwrites rather than duplicates) and append ONE ``user.action`` audit event.
    Requires the REAL historical-bars adapter (ENABLE_HISTORICAL_DATA + resolvable Alpaca data
    credentials) — the job refuses to STORE labels computed from the synthetic price provider, so
    fake bars never produce stored research labels (mirrors the platform's data-integrity stance).
  * ``--exclude-tickers``: comma-separated ticker(s) (repeatable OR single comma-joined, case-
    insensitive) to drop BEFORE labeling — e.g. known-bad signals admitted by a since-fixed
    upstream defect (Task 04c: 44 leaked sec_form4 signals ticker NAN) that must stay in the
    append-only event log but should not feed the research projection. Excluded signals are
    counted (``excluded`` in the report) and, on ``--confirm``, recorded in the run's
    ``user.action`` audit payload.

No HTTP endpoint is added (the API surface stays stable; documented in MEMORY.md). Composition
mirrors ``trading.jobs.backfill`` / ``trading.jobs.cycle``: build dependencies via the composition
root, run, and ALWAYS tear down async resources. Exit 0 clean / 1 on failure.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from uuid import uuid4

from trading.application.outcome_labels import LabelReport
from trading.config import Settings
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging

logger = logging.getLogger(__name__)


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    print(text)  # noqa: T201 — summary output is the whole point of this command


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.label_outcomes",
        description="Label ticker-resolved signals with forward event-study abnormal returns.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="write labels (default is a dry run); requires the REAL historical-bars adapter",
    )
    parser.add_argument(
        "--exclude-tickers",
        action="append",
        default=[],
        metavar="TICKER[,TICKER...]",
        help=(
            "ticker(s) to exclude from labeling (case-insensitive); comma-separated, and/or "
            "repeat the flag, e.g. --exclude-tickers NAN,XYZ or --exclude-tickers NAN "
            "--exclude-tickers XYZ"
        ),
    )
    return parser.parse_args(list(argv))


def _parse_exclude_tickers(raw: Sequence[str]) -> frozenset[str] | None:
    """Flatten repeated/comma-joined ``--exclude-tickers`` values into an upper-cased set.

    Returns ``None`` (not an empty frozenset) when nothing was passed, so ``execute``'s
    ``exclude_tickers`` stays a clean "absent" signal rather than a vacuous empty set.
    """
    tickers = {
        ticker.strip().upper() for group in raw for ticker in group.split(",") if ticker.strip()
    }
    return frozenset(tickers) if tickers else None


def _format_report(report: LabelReport, *, dry_run: bool) -> str:
    lines = [
        f"  label_run_id:  {report.label_run_id}",
        f"  as_of:         {report.as_of.isoformat()}",
        f"  signals:       {report.signal_count}  (ticker-resolved, REAL+BACKFILLED)",
        f"  excluded:      {report.excluded}  (dropped by --exclude-tickers, not labeled)",
        f"  labels:        {report.labels}  (one per signal x horizon)",
        f"  ok:            {report.ok}",
        f"  unresolvable:  {report.unresolvable}",
    ]
    if dry_run:
        lines.append("  committed:     NO (dry-run)")
    else:
        lines.append(f"  committed:     {'YES' if report.committed else 'NO'}")
    return "\n".join(lines)


async def run_labeling(*, confirm: bool, exclude_tickers: frozenset[str] | None = None) -> int:
    """Build dependencies, label (or dry-run), tear down. Returns an exit code.

    Guard: ``--confirm`` requires the REAL historical-bars adapter to be wired
    (``deps.real_historical_prices``). A dry run may use whatever provider is configured. Teardown
    always runs via ``finally``.
    """
    settings = Settings()
    configure_logging(settings.log_level)

    deps = build_dependencies(settings)
    try:
        if deps.label_outcomes is None:  # defensive: the composition root always wires this
            raise RuntimeError("outcome labeler is not configured")

        if confirm and not deps.real_historical_prices:
            _emit(
                "error: --confirm requires the REAL historical-bars adapter (set "
                "ENABLE_HISTORICAL_DATA=true with resolvable Alpaca data credentials). Refusing "
                "to store labels computed from the synthetic price provider — a dry run is always "
                "allowed."
            )
            return 1

        label_run_id = str(uuid4())
        report = await deps.label_outcomes.execute(
            label_run_id=label_run_id, dry_run=not confirm, exclude_tickers=exclude_tickers
        )

        _emit("=== Label outcomes ===")
        _emit(f"mode: {'CONFIRM (writes labels)' if confirm else 'DRY-RUN (no changes)'}")
        if exclude_tickers:
            _emit(f"exclude_tickers: {', '.join(sorted(exclude_tickers))}")
        _emit("")
        _emit(_format_report(report, dry_run=not confirm))

        if not confirm:
            _emit("")
            _emit("DRY-RUN: nothing was written. Re-run with --confirm to store labels.")
            return 0

        _emit("")
        _emit(f"stored {report.labels} outcome label(s) for run {label_run_id}.")
        return 0
    finally:
        await _teardown(deps)


async def _teardown(deps: AppDeps) -> None:
    """Release every async resource the composition root registered (mirrors app shutdown).

    Each acloser is awaited independently so one failing close cannot strand the others (e.g. the
    Postgres pool is still disposed even if an HTTP client close raised).
    """
    for aclose in deps.aclosers:
        try:
            await aclose()
        except Exception:
            logger.exception("error closing a resource during job teardown")


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entrypoint: parse args, run the labeler, return a process exit code.

    Returns 0 on success (including a clean dry run) and 1 on any failure — a disallowed
    --confirm, or an unhandled exception (logged before returning).
    """
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = _parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    try:
        exclude_tickers = _parse_exclude_tickers(args.exclude_tickers)
        return asyncio.run(
            run_labeling(confirm=bool(args.confirm), exclude_tickers=exclude_tickers)
        )
    except Exception:
        logger.exception("label-outcomes job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
