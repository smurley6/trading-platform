"""Diagnostic report job (Task 05, docs/RESEARCH-PLAN.md §2-§3).

Reports, per data source: signal resolution rates, a monthly disclosure-date histogram, forward
abnormal-return stats (per horizon x benchmark class, and per horizon x POINT-IN-TIME conviction
bucket), the hard-coded reliability prior next to the measured hit rate / mean return, and an
unresolvable-label reason breakdown. Strictly READ-ONLY over the read model (signals) and the
``outcome_labels`` projection for a single label run — the only write in the whole flow is ONE
``user.action`` audit event recording that the report ran.

USAGE::

    python -m trading.jobs.diagnostic
    python -m trading.jobs.diagnostic --run-id <label_run_id>
    python -m trading.jobs.diagnostic --out report.json

  * Default label run: the LATEST ``label_run_id`` in the outcome-labels store
    (:meth:`~trading.application.ports.outcome_labels.OutcomeLabelsPort.latest_run_id`).
    ``--run-id`` overrides this.
  * Output: the full report as JSON to stdout (or ``--out <path>`` when given) AND a
    human-readable printed table — both always produced (the JSON is the machine-readable
    record; the table is for an operator reading the terminal).
  * No HTTP endpoint (deliberate, matches ``jobs/backfill.py``/``jobs/label_outcomes.py`` — the
    API surface stays stable and research runs stay operator-driven; documented in MEMORY.md §7).

Composition mirrors ``trading.jobs.label_outcomes``: build dependencies via the composition
root, run, and ALWAYS tear down async resources. Exit 0 clean / 1 on failure (including "no
label run found").
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from trading.application.diagnostic import (
    DiagnosticReport,
    EpisodeHorizonStats,
    ReturnStats,
    build_report,
)
from trading.config import Settings
from trading.domain.events import DomainEvent, EventType
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging

logger = logging.getLogger(__name__)

_USER_ACTION_STREAM = "user-actions"

# The report's own section names, recorded in the audit event so a run's provenance says what it
# covered (clarification 6's `sections` field) without hard-coding a duplicate list elsewhere.
_SECTIONS: tuple[str, ...] = (
    "resolution_by_source",
    "monthly_histogram",
    "horizon_benchmark_stats",
    "horizon_conviction_stats",
    "prior_vs_measured",
    "unresolvable_reasons",
    "form4_episode_stats",
)


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    print(text)  # noqa: T201 — summary output is the whole point of this command


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.diagnostic",
        description=(
            "Read-only diagnostic report over a label run: resolution rates, disclosure-date "
            "histogram, abnormal-return stats by benchmark/conviction bucket, prior-vs-measured "
            "reliability, and unresolvable-label reasons."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="LABEL_RUN_ID",
        help="label run to report on (default: the latest run in the outcome-labels store)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="write the full JSON report to this path (default: stdout)",
    )
    return parser.parse_args(list(argv))


def _report_to_json(report: DiagnosticReport) -> dict[str, object]:
    """Convert the dataclass tree to a plain JSON-able dict (ISO-format the one datetime)."""
    payload = asdict(report)
    payload["as_of"] = report.as_of.isoformat()
    return payload


def _format_table(report: DiagnosticReport) -> str:
    """Human-readable printed table covering every report section, in spec order."""
    lines = [
        "=== Diagnostic report ===",
        f"label_run_id: {report.label_run_id}",
        f"as_of:        {report.as_of.isoformat()}",
        f"signals:      {report.signal_count}",
        f"labels:       {report.label_count}",
        "",
        "-- 1. Resolution by source --",
    ]
    for resolution_row in report.resolution:
        rate = (
            "n/a"
            if resolution_row.resolution_rate is None
            else f"{resolution_row.resolution_rate:.1%}"
        )
        lines.append(
            f"  {resolution_row.source:<20} total={resolution_row.total_signals:<6} "
            f"ticker_resolved={resolution_row.ticker_resolved:<6} resolution_rate={rate}"
        )
    if not report.resolution:
        lines.append("  (no signals)")

    lines.append("")
    lines.append("-- 2. Monthly disclosure-date histogram --")
    for source, buckets in report.monthly_histogram.items():
        lines.append(f"  {source}:")
        for month, count in buckets.items():
            lines.append(f"    {month}: {count}")
    if not report.monthly_histogram:
        lines.append("  (no signals)")

    lines.append("")
    lines.append("-- 3. Abnormal return by source x horizon x benchmark class --")
    for benchmark_row in report.horizon_benchmark_stats:
        lines.append(
            f"  {benchmark_row.source:<20} h={benchmark_row.horizon_days:<3} "
            f"bench={benchmark_row.benchmark_class:<7} {_format_stats(benchmark_row.stats)}"
        )
    if not report.horizon_benchmark_stats:
        lines.append("  (no ok labels)")

    lines.append("")
    lines.append("-- 4. Abnormal return by source x horizon x conviction bucket --")
    for conviction_row in report.horizon_conviction_stats:
        lines.append(
            f"  {conviction_row.source:<20} h={conviction_row.horizon_days:<3} "
            f"bucket={conviction_row.conviction_bucket:<6} {_format_stats(conviction_row.stats)}"
        )
    if not report.horizon_conviction_stats:
        lines.append("  (no ok labels with a conviction bucket)")

    lines.append("")
    lines.append("-- 5. Prior vs measured reliability --")
    for prior_row in report.prior_vs_measured:
        prior = (
            "n/a" if prior_row.prior_reliability is None else f"{prior_row.prior_reliability:.2f}"
        )
        hit = "n/a" if prior_row.measured_hit_rate is None else f"{prior_row.measured_hit_rate:.1%}"
        mean = (
            "n/a"
            if prior_row.measured_mean_abnormal_return is None
            else f"{prior_row.measured_mean_abnormal_return:+.4f}"
        )
        lines.append(
            f"  {prior_row.source:<20} prior={prior:<6} measured_hit_rate={hit:<7} "
            f"measured_mean_abnormal_return={mean:<9} n={prior_row.n}"
        )
    if not report.prior_vs_measured:
        lines.append("  (no data)")

    lines.append("")
    lines.append("-- 6. Unresolvable-label reasons --")
    for reason, count in report.unresolvable_reasons.items():
        lines.append(f"  {reason}: {count}")
    if not report.unresolvable_reasons:
        lines.append("  (no unresolvable labels)")

    lines.append("")
    lines.append("-- 7. Form 4 buy edge: episode-level vs label-level (task-13) --")
    curve = " ".join(
        f"{day}d={weight:.3f}" for day, weight in sorted(report.form4_buy_decay_curve.items())
    )
    lines.append(
        f"  anchored expected edge (h=1): {report.form4_expected_edge_h1:+.4f} "
        f"(episode ex-LUV mean; OLD label-level +0.0100 via FORM4_EXPECTED_EDGE_H1=0.01)"
    )
    lines.append(
        f"  buy decay half-life: {report.form4_buy_decay_half_life_days:g}d  weights [{curve}]  "
        f"(0 recovers the OLD flat holding via FORM4_BUY_DECAY_HALF_LIFE_DAYS=0)"
    )
    lines.append(
        "  primary cell = ex_luv per-EPISODE h=1 (pre-registered); label-level shown alongside."
    )
    for episode_row in report.form4_episode_stats:
        lines.append(_format_episode_stats(episode_row))
    if not report.form4_episode_stats:
        lines.append("  (no Form 4 buy labels)")

    return "\n".join(lines)


def _format_episode_stats(row: EpisodeHorizonStats) -> str:
    """One episode-vs-label line: episode counts/stats then the pooled label-level counterpart."""

    def _pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    def _num(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.4f}"

    def _se(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    low = " LOW-N" if row.low_n else ""
    return (
        f"  {row.variant:<8} h={row.horizon_days:<3} "
        f"EPISODE n_ep={row.n_episodes:<4} n_tick={row.n_tickers:<4} "
        f"hit={_pct(row.episode_hit_rate):<7} mean={_num(row.episode_mean):<9} "
        f"med={_num(row.episode_median):<9} se={_se(row.episode_clustered_se):<7} | "
        f"LABEL n={row.n_labels:<5} hit={_pct(row.label_hit_rate):<7} "
        f"mean={_num(row.label_mean):<9} se={_se(row.label_clustered_se):<7}{low}"
    )


def _format_stats(stats: ReturnStats) -> str:
    mean_s = "n/a" if stats.mean is None else f"{stats.mean:+.4f}"
    median_s = "n/a" if stats.median is None else f"{stats.median:+.4f}"
    hit_s = "n/a" if stats.hit_rate is None else f"{stats.hit_rate:.1%}"
    stdev_s = "n/a" if stats.stdev is None else f"{stats.stdev:.4f}"
    return (
        f"n={stats.n:<4} mean={mean_s:<9} median={median_s:<9} hit_rate={hit_s:<7} stdev={stdev_s}"
    )


async def run_diagnostic(*, run_id: str | None, out: str | None) -> int:
    """Build dependencies, compute the report (read-only), append the audit event, tear down.

    Returns 1 (no writes attempted) when no label run is found: an explicit ``--run-id`` that
    does not exist, or (when defaulting) an empty outcome-labels store.
    """
    settings = Settings()
    configure_logging(settings.log_level)

    deps = build_dependencies(settings)
    try:
        if deps.outcome_labels is None:  # defensive: the composition root always wires this
            raise RuntimeError("outcome-labels store is not configured")

        label_run_id = run_id if run_id is not None else await deps.outcome_labels.latest_run_id()
        if label_run_id is None:
            _emit(
                "error: no label run found (the outcome-labels store is empty and no --run-id "
                "was given). Run `python -m trading.jobs.label_outcomes --confirm` first."
            )
            return 1

        labels = await deps.outcome_labels.labels_for_run(label_run_id)
        if not labels:
            _emit(f"error: no labels found for run '{label_run_id}'.")
            return 1

        signals = list(await deps.read_model.all_signals())
        as_of = deps.clock.now()

        report = build_report(
            label_run_id=label_run_id,
            as_of=as_of,
            signals=signals,
            labels=labels,
            window_days=deps.conviction_window_days,
            thresholds=deps.conviction_thresholds,
            expected_edge_h1=settings.form4_expected_edge_h1,
            decay_half_life_days=settings.form4_buy_decay_half_life_days,
        )

        payload = _report_to_json(report)
        rendered = json.dumps(payload, indent=2, sort_keys=False)
        if out is not None:
            Path(out).write_text(rendered + "\n")
        else:
            _emit(rendered)
        _emit("")
        _emit(_format_table(report))

        # The ONE write (clarification 6): a single user.action audit event recording that this
        # report ran, over which run, and what it covered — everything else is strictly read-only.
        await deps.event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": "diagnostic",
                    "label_run_id": label_run_id,
                    "as_of": as_of.isoformat(),
                    "sections": list(_SECTIONS),
                    "signal_count": report.signal_count,
                    "label_count": report.label_count,
                },
                occurred_at=as_of,
                correlation_id=label_run_id,
            )
        )
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
    """Synchronous entrypoint: parse args, run the diagnostic, return a process exit code.

    Returns 0 on success and 1 on any failure — no label run found, or an unhandled exception
    (logged before returning).
    """
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = _parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    try:
        return asyncio.run(run_diagnostic(run_id=args.run_id, out=args.out))
    except Exception:
        logger.exception("diagnostic job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
