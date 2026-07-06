"""Diagnostic report (Task 05, docs/RESEARCH-PLAN.md §2-§3): the program's first learning output.

Read-only over the CQRS read model (signals) and the rebuildable ``outcome_labels`` projection.
Computes, per data source:

  1. Total signals, ticker-resolved count, resolution rate.
  2. A monthly (YYYY-MM) histogram of disclosure dates (``Signal.created_at``).
  3. Per horizon (1/5/20 trading days), split by benchmark class (``sector:`` vs ``market``):
     n labeled, mean/median abnormal return, hit rate (fraction > 0), stdev.
  4. The same stats, split by a POINT-IN-TIME conviction bucket (0-40 / 40-70 / 70+).
  5. The hard-coded reliability prior (``infrastructure/data_sources``) next to the measured hit
     rate / mean abnormal return.
  6. A breakdown of unresolvable-label reasons.

Everything here is pure computation over data already fetched by the caller (the job); the ONE
write in the whole diagnostic flow — a single ``user.action`` audit event — is the job's
responsibility, not this module's (mirrors ``application/data_coverage.py``: this module is the
computation, the job/use-case class does the I/O). Stats use only the stdlib ``statistics``
module (no numpy/pandas — this module stays dependency-free by design).
"""

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from trading.application.ports.outcome_labels import OutcomeLabel
from trading.domain.conviction import ConvictionParams, ConvictionThresholds, compute_conviction
from trading.domain.signals import Signal

# --- Source key derivation (clarification 3) ---------------------------------------------------

# `Signal.id` prefix -> diagnostic source key, verified against the actual live/backfill id
# formats emitted by `infrastructure/data_sources/*.py` (see the grep audit in the task report):
#   sec_form4.py:        id=f"sec-form4-{...}"       (live)
#                         id=f"sec-form4-hist-{...}"  (historical/backfill — ALSO starts with
#                                                       "sec-form4-")
#   house_disclosures.py: id=f"house-{...}"
#   usaspending.py:       id=f"contract-{...}"
#   news.py:              id=f"news-{...}"
# Order matters: checked longest/most-specific prefix first so e.g. "sec-form4-hist-" (which also
# starts with "sec-form4-") still maps to the same "sec_form4" key without needing its own entry.
_PREFIX_TO_SOURCE_KEY: tuple[tuple[str, str], ...] = (
    ("sec-form4-", "sec_form4"),
    ("house-", "house_disclosures"),
    ("contract-", "usaspending"),
    ("news-", "news"),
)


def source_key_for_signal_id(signal_id: str) -> str:
    """Map a ``Signal.id`` to its diagnostic source key by prefix, else ``"unknown"``.

    Deliberately a single small helper (clarification 3) so the report's source grouping and any
    future consumer share one mapping. Other live sources (senate_efd, lobbying, fda_approvals,
    federal_register, congress_bills, regulations_gov, clinical_trials, congress_trades,
    sec_edgar_atom) are not in the RESEARCH-PLAN §3.1 diagnostic scope's explicit source list and
    fall through to "unknown" rather than silently miscounting into an unrelated bucket.
    """
    for prefix, key in _PREFIX_TO_SOURCE_KEY:
        if signal_id.startswith(prefix):
            return key
    return "unknown"


# --- Section 5: hard-coded reliability priors ---------------------------------------------------

# The reliability prior each diagnostic source key trades against (clarification 4, section 5).
# Values are the SAME module-level constants the adapters import when stamping
# `Signal.source_reliability` — duplicated here (not imported) because they live in per-source
# modules with import-time side effects/heavy deps (httpx clients, pdfplumber) that a read-only
# report has no business pulling in; each value is annotated with its origin for auditability.
RELIABILITY_PRIORS: Mapping[str, float] = {
    # infrastructure/data_sources/sec_form4.py::RELIABILITY_INSIDER_TRADE
    "sec_form4": 0.75,
    # infrastructure/data_sources/__init__.py::RELIABILITY_DISCLOSED_TRADE (house_disclosures.py
    # imports this shared constant, it has no reliability constant of its own)
    "house_disclosures": 0.7,
    # infrastructure/data_sources/__init__.py::RELIABILITY_CONTRACT (usaspending.py imports this)
    "usaspending": 0.9,
    # infrastructure/data_sources/__init__.py::RELIABILITY_STATEMENT (news.py imports this)
    "news": 0.6,
}

# Conviction buckets (clarification 5): thresholds mirror the domain's TRADE/WATCH cut points
# (`ConvictionThresholds` defaults: watch=40, trade=70) so "0-40 / 40-70 / 70+" lines up with the
# same ignore/watch/trade action boundaries the live engine uses.
ConvictionBucket = str  # "0-40" | "40-70" | "70+"


def conviction_bucket_for_score(score: float, thresholds: ConvictionThresholds) -> ConvictionBucket:
    """Map a 0-100 conviction score to its display bucket using the domain's own thresholds."""
    if score >= thresholds.trade:
        return "70+"
    if score >= thresholds.watch:
        return "40-70"
    return "0-40"


# --- Section 1: per-source resolution --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceResolution:
    source: str
    total_signals: int
    ticker_resolved: int

    @property
    def resolution_rate(self) -> float | None:
        """``ticker_resolved / total_signals``, or ``None`` when there are no signals at all."""
        if self.total_signals == 0:
            return None
        return self.ticker_resolved / self.total_signals


def resolution_by_source(signals: Sequence[Signal]) -> tuple[SourceResolution, ...]:
    """Section 1: total / ticker-resolved / resolution rate, one row per source key present."""
    totals: dict[str, int] = defaultdict(int)
    resolved: dict[str, int] = defaultdict(int)
    for signal in signals:
        key = source_key_for_signal_id(signal.id)
        totals[key] += 1
        if signal.ticker is not None and signal.ticker.strip():
            resolved[key] += 1
    return tuple(
        SourceResolution(source=key, total_signals=totals[key], ticker_resolved=resolved[key])
        for key in sorted(totals)
    )


# --- Section 2: monthly disclosure-date histogram -----------------------------------------------


def monthly_histogram(signals: Sequence[Signal]) -> dict[str, dict[str, int]]:
    """Section 2: {source -> {"YYYY-MM" -> count}} bucketed on ``Signal.created_at``."""
    histogram: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for signal in signals:
        key = source_key_for_signal_id(signal.id)
        bucket = f"{signal.created_at.year:04d}-{signal.created_at.month:02d}"
        histogram[key][bucket] += 1
    return {source: dict(sorted(buckets.items())) for source, buckets in sorted(histogram.items())}


# --- Section 3 / 4: horizon x benchmark [x conviction bucket] abnormal-return stats -------------


def _benchmark_class(benchmark: str) -> str:
    """ "sector:<ETF>" -> "sector"; "market" -> "market"; anything else -> "other" (defensive)."""
    if benchmark.startswith("sector:"):
        return "sector"
    if benchmark == "market":
        return "market"
    return "other"


@dataclass(frozen=True, slots=True)
class ReturnStats:
    """Stats over one group's abnormal returns. ``stdev`` is ``None`` for a single sample."""

    n: int
    mean: float | None
    median: float | None
    hit_rate: float | None  # fraction of ok labels with abnormal_return > 0
    stdev: float | None


def _return_stats(values: Sequence[float]) -> ReturnStats:
    """Plain-Python stats (stdlib ``statistics`` only): stdev of <2 samples is None, not a crash."""
    n = len(values)
    if n == 0:
        return ReturnStats(n=0, mean=None, median=None, hit_rate=None, stdev=None)
    mean = statistics.mean(values)
    median = statistics.median(values)
    hit_rate = sum(1 for v in values if v > 0) / n
    stdev = statistics.stdev(values) if n >= 2 else None
    return ReturnStats(n=n, mean=mean, median=median, hit_rate=hit_rate, stdev=stdev)


@dataclass(frozen=True, slots=True)
class HorizonBenchmarkStats:
    source: str
    horizon_days: int
    benchmark_class: str  # "sector" | "market" | "other"
    stats: ReturnStats


def stats_by_source_horizon_benchmark(
    labels: Sequence[OutcomeLabel], signal_source: Mapping[str, str]
) -> tuple[HorizonBenchmarkStats, ...]:
    """Section 3: per (source, horizon, benchmark-class), stats over "ok" abnormal returns.

    ``signal_source`` maps ``signal_id -> source key`` (derived once by the caller via
    :func:`source_key_for_signal_id` over the signals batch) so labels — which do not carry the
    original signal id's full context — can still be grouped by source. A label whose signal id
    is unknown to the map (should not happen: labels are only ever produced for read-model
    signals) falls back to "unknown".
    """
    groups: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for label in labels:
        if label.status != "ok" or label.abnormal_return is None:
            continue
        source = signal_source.get(label.signal_id, "unknown")
        key = (source, label.horizon_days, _benchmark_class(label.benchmark))
        groups[key].append(label.abnormal_return)
    return tuple(
        HorizonBenchmarkStats(
            source=source, horizon_days=horizon, benchmark_class=bench, stats=_return_stats(values)
        )
        for (source, horizon, bench), values in sorted(groups.items())
    )


@dataclass(frozen=True, slots=True)
class HorizonConvictionStats:
    source: str
    horizon_days: int
    conviction_bucket: ConvictionBucket
    stats: ReturnStats


def stats_by_source_horizon_conviction_bucket(
    labels: Sequence[OutcomeLabel],
    signal_source: Mapping[str, str],
    conviction_bucket: Mapping[str, ConvictionBucket],
) -> tuple[HorizonConvictionStats, ...]:
    """Section 4: per (source, horizon, conviction bucket), stats over "ok" abnormal returns.

    ``conviction_bucket`` maps ``signal_id -> bucket`` for every ticker-resolved signal (computed
    point-in-time by the caller, see :func:`point_in_time_conviction_buckets`). A label whose
    signal has no entry (should not happen for a ticker-resolved, labeled signal) is skipped —
    "where conviction exists" (clarification 5) means ticker-resolved signals only.
    """
    groups: dict[tuple[str, int, ConvictionBucket], list[float]] = defaultdict(list)
    for label in labels:
        if label.status != "ok" or label.abnormal_return is None:
            continue
        bucket = conviction_bucket.get(label.signal_id)
        if bucket is None:
            continue
        source = signal_source.get(label.signal_id, "unknown")
        key = (source, label.horizon_days, bucket)
        groups[key].append(label.abnormal_return)
    return tuple(
        HorizonConvictionStats(
            source=source,
            horizon_days=horizon,
            conviction_bucket=bucket,
            stats=_return_stats(values),
        )
        for (source, horizon, bucket), values in sorted(groups.items())
    )


# --- Clarification 5: POINT-IN-TIME conviction bucketing ----------------------------------------


def point_in_time_conviction_buckets(
    signals: Sequence[Signal],
    *,
    window_days: int,
    thresholds: ConvictionThresholds,
    params: ConvictionParams | None = None,
) -> dict[str, ConvictionBucket]:
    """Bucket every ticker-resolved signal by conviction computed AT ITS OWN ``created_at``.

    Anti-lookahead core (clarification 5, mirrors the labeler's anti-lookahead session
    resolution): a signal's conviction bucket is computed from the trailing
    ``window_days``-day corroboration window ENDING at that signal's own ``created_at``, using
    only signals for the SAME ticker created no later than it — a later signal must NEVER change
    an earlier signal's bucket. Ticker-less signals are excluded entirely (no ticker to score).

    Implementation: one pass over signals in `created_at` order, grouped by ticker, with a
    sliding window (a per-ticker list of already-seen signals, trimmed to the trailing
    `window_days` before each new signal is scored) — O(n) signals x O(window) comparisons per
    ticker, not O(n^2) over the whole signal set.
    """
    resolved = [s for s in signals if s.ticker is not None and s.ticker.strip()]
    ordered = sorted(resolved, key=lambda s: s.created_at)

    # Per-ticker trailing window of signals already processed (chronological), so scoring signal
    # `i` only ever sees signals with created_at <= signal i's created_at for the SAME ticker.
    window_by_ticker: dict[str, list[Signal]] = defaultdict(list)
    buckets: dict[str, ConvictionBucket] = {}

    for signal in ordered:
        ticker = signal.ticker
        assert ticker is not None  # narrowed by the `resolved` filter above
        history = window_by_ticker[ticker]
        cutoff = signal.created_at.timestamp() - window_days * 86_400.0
        # Trim signals that fell out of the trailing window (older than `window_days` before
        # THIS signal); this keeps the list from growing unboundedly across a long ticker history
        # while never admitting a future signal (nothing is ever appended ahead of its turn).
        history[:] = [s for s in history if s.created_at.timestamp() >= cutoff]
        # Include the current signal itself in its own scoring window (mirrors
        # `signals_in_window` + `compute_conviction`'s live-path semantics: the just-arrived
        # signal contributes to its own conviction).
        in_window = [*history, signal]
        result = compute_conviction(
            ticker, in_window, window_days=window_days, thresholds=thresholds, params=params
        )
        buckets[signal.id] = conviction_bucket_for_score(result.score, thresholds)
        history.append(signal)

    return buckets


# --- Section 5: prior vs measured ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PriorVsMeasured:
    source: str
    prior_reliability: float | None  # None when the source has no documented prior
    measured_hit_rate: float | None
    measured_mean_abnormal_return: float | None
    n: int


def prior_vs_measured(
    labels: Sequence[OutcomeLabel], signal_source: Mapping[str, str]
) -> tuple[PriorVsMeasured, ...]:
    """Section 5: per source, the hard-coded prior next to the measured hit rate / mean return.

    Measured over ALL horizons pooled (the prior is a single per-source scalar, not per-horizon)
    so there is one comparable number per source. Sources with zero "ok" labels still appear
    (n=0, measured fields None) so an entirely-unlabeled source is visible, not silently omitted.
    """
    values_by_source: dict[str, list[float]] = defaultdict(list)
    for label in labels:
        if label.status != "ok" or label.abnormal_return is None:
            continue
        source = signal_source.get(label.signal_id, "unknown")
        values_by_source[source].append(label.abnormal_return)

    all_sources = sorted(set(RELIABILITY_PRIORS) | set(values_by_source))
    rows = []
    for source in all_sources:
        values = values_by_source.get(source, [])
        n = len(values)
        hit_rate = (sum(1 for v in values if v > 0) / n) if n else None
        mean = statistics.mean(values) if n else None
        rows.append(
            PriorVsMeasured(
                source=source,
                prior_reliability=RELIABILITY_PRIORS.get(source),
                measured_hit_rate=hit_rate,
                measured_mean_abnormal_return=mean,
                n=n,
            )
        )
    return tuple(rows)


# --- Section 6: unresolvable-reason breakdown -----------------------------------------------------


def unresolvable_reason_breakdown(labels: Sequence[OutcomeLabel]) -> dict[str, int]:
    """Section 6: {reason -> count} over "unresolvable" labels (reason defaults to "unknown")."""
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        if label.status != "unresolvable":
            continue
        counts[label.reason or "unknown"] += 1
    return dict(sorted(counts.items()))


# --- Whole-report assembly -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """The full Task-05 diagnostic, ready to serialize to JSON or print as a table."""

    label_run_id: str
    as_of: datetime
    signal_count: int
    label_count: int
    resolution: tuple[SourceResolution, ...] = field(default_factory=tuple)
    monthly_histogram: dict[str, dict[str, int]] = field(default_factory=dict)
    horizon_benchmark_stats: tuple[HorizonBenchmarkStats, ...] = field(default_factory=tuple)
    horizon_conviction_stats: tuple[HorizonConvictionStats, ...] = field(default_factory=tuple)
    prior_vs_measured: tuple[PriorVsMeasured, ...] = field(default_factory=tuple)
    unresolvable_reasons: dict[str, int] = field(default_factory=dict)


def build_report(
    *,
    label_run_id: str,
    as_of: datetime,
    signals: Sequence[Signal],
    labels: Sequence[OutcomeLabel],
    window_days: int,
    thresholds: ConvictionThresholds,
    params: ConvictionParams | None = None,
) -> DiagnosticReport:
    """Assemble every report section (pure; no I/O) from a signals batch + a label-run batch."""
    signal_source = {s.id: source_key_for_signal_id(s.id) for s in signals}
    conviction_buckets = point_in_time_conviction_buckets(
        signals, window_days=window_days, thresholds=thresholds, params=params
    )
    return DiagnosticReport(
        label_run_id=label_run_id,
        as_of=as_of,
        signal_count=len(signals),
        label_count=len(labels),
        resolution=resolution_by_source(signals),
        monthly_histogram=monthly_histogram(signals),
        horizon_benchmark_stats=stats_by_source_horizon_benchmark(labels, signal_source),
        horizon_conviction_stats=stats_by_source_horizon_conviction_bucket(
            labels, signal_source, conviction_buckets
        ),
        prior_vs_measured=prior_vs_measured(labels, signal_source),
        unresolvable_reasons=unresolvable_reason_breakdown(labels),
    )
