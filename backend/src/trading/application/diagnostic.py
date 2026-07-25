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

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from trading.application.form4_corroboration import form4_buy_votes
from trading.application.ports.outcome_labels import OutcomeLabel
from trading.domain.conviction import (
    ConvictionParams,
    ConvictionThresholds,
    compute_conviction,
    group_form4_vote_streams,
)
from trading.domain.form4_edge import (
    DEFAULT_FORM4_BUY_DECAY_HALF_LIFE_DAYS,
    DEFAULT_FORM4_EXPECTED_EDGE_H1,
    form4_buy_decay_weight,
)
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
#
# Task 10 (docs/RESEARCH-PLAN.md; Phase 6 label run a38d3848-f1eb-4a55-8a6d-15e8f1464f7a, 68,242 ok
# labels, 2026-07-07) refuted the flat sec_form4 (0.75) and house_disclosures (0.70) priors and
# made sec_form4 SIDE-AWARE at the adapter (purchase 0.60, sale 0.0 — see
# infrastructure/data_sources/sec_form4.py::_RELIABILITY_BY_CODE). This section groups `ok` labels
# by SOURCE KEY only (`OutcomeLabel`/`signal_source` carry no buy/sell side), so it cannot split
# sec_form4's row by side without a broader schema change (out of scope here) — the row below is
# therefore a BLEND, documented rather than side-split per the task brief:
#   sec_form4 blend rule: a per-signal reliability-weighted blend would need the actual
#   purchase/sale mix in the labeled set, which this section does not carry; we instead print the
#   PURCHASE-side prior (0.60) since purchases are the only verified directional edge (ex-LUV
#   60.9% aligned / +1.01% mean, n=724, monotone by horizon) and sales contribute zero conviction
#   by construction, making the purchase prior the operative "does the measured number back the
#   number we actually trade on" comparison. The measured hit-rate/mean columns are UNCHANGED
#   (still pooled over both sides — a labeling/measurement fact, not a conviction weight) so this
#   remains a comparison of "prior we act on" vs "what actually happened to every disclosed trade".
RELIABILITY_PRIORS: Mapping[str, float] = {
    # infrastructure/data_sources/sec_form4.py::RELIABILITY_INSIDER_PURCHASE (side-split; see the
    # blend-rule comment above — sale side is 0.0 and not represented by this single scalar)
    "sec_form4": 0.60,
    # infrastructure/data_sources/house_disclosures.py::RELIABILITY_HOUSE_DISCLOSURE — benched to
    # 0.0 (Phase 6: measured 0.504 direction-aligned vs the refuted 0.70 prior; see the re-review
    # trigger in docs/RESEARCH-PLAN.md, "Re-review triggers")
    "house_disclosures": 0.0,
    # infrastructure/data_sources/__init__.py::RELIABILITY_CONTRACT (usaspending.py imports this).
    # Untouched by Task 10 — still flagged untested (deferred Task 06 FMP spot-check).
    "usaspending": 0.9,
    # infrastructure/data_sources/__init__.py::RELIABILITY_STATEMENT (news.py imports this).
    # Untouched by Task 10 — still flagged untested.
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


# --- Episode-level Form 4 buy edge (task-13, docs/RESEARCH-PLAN.md §2-§3) ------------------------

# The mega-ticker whose same-day insider stacking inflated the pooled label-level buy edge (194
# same-day LUV rows across 2 owners in the Phase-7 analysis). The evaluator reports BOTH a with-LUV
# and an ex-LUV variant so the LUV distortion is always visible; the PRE-REGISTERED primary cell is
# ex-LUV per-episode h=1 (docs/RESEARCH-PLAN.md — "No new splits count as confirmation").
FORM4_EPISODE_EXCLUDED_TICKER = "LUV"

# Episode cells thinner than this are flagged low_n: too few independent clusters/episodes for the
# cluster-robust read to be trustworthy.
_MIN_EPISODE_CELL_N = 30


def _hit_rate(values: Sequence[float]) -> float | None:
    return (sum(1 for v in values if v > 0) / len(values)) if values else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def clustered_standard_error(values_by_cluster: Mapping[str, Sequence[float]]) -> float | None:
    """CR0 cluster-robust standard error of the sample mean, clustered by the mapping's keys.

    Regressing the outcome on a constant makes the point estimate the sample mean; the CR0 sandwich
    variance of that mean is ``(1 / n^2) * sum_g (sum_{i in g} (x_i - xbar))^2`` -- the sum over
    clusters ``g`` of the SQUARED cluster residual totals, with NO small-sample (``G/(G-1)``)
    correction (that is what makes it CR0). Unlike the iid ``sigma / sqrt(n)`` estimator it does NOT
    shrink toward zero as correlated observations pile into a handful of clusters -- 194 same-ticker
    Form 4 rows count as roughly one independent draw, not 194 -- which is exactly why the honest
    report clusters by ticker. Returns ``None`` for an empty input; ``0.0`` when every cluster's
    residual total is 0 (e.g. a single observation, whose residual is 0).
    """
    values = [v for group in values_by_cluster.values() for v in group]
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    meat = sum(sum(v - mean for v in group) ** 2 for group in values_by_cluster.values())
    return math.sqrt(meat) / n


@dataclass(frozen=True, slots=True)
class EpisodeHorizonStats:
    """Form 4 buy edge at one horizon, per-EPISODE reported ALONGSIDE label-level (task-13).

    An *episode* is one ``(ticker, normalized owner)`` filer-on-name relationship — every Form 4
    buy that owner filed on that ticker — collapsed to a single unit whose outcome is the MEAN of
    its member abnormal returns at the horizon. Collapsing to episodes removes the same-day /
    same-filer stacking that inflates the naive pooled label-level number (Phase 7: LUV's 194
    same-day rows drove the with-LUV label hit to 80% while the honest per-episode hit is ~62%).

    Every cell carries a ticker-clustered (CR0) SE and the honest unit counts at BOTH levels
    (``n_episodes`` / ``n_tickers`` for the episode read, ``n_labels`` for the pooled read), and
    ``low_n`` flags an episode cell with fewer than 30 episodes.
    """

    horizon_days: int
    variant: str  # "with_luv" | "ex_luv"
    # Pooled label-level (one row per member label — the inflated, naive number).
    n_labels: int
    label_hit_rate: float | None
    label_mean: float | None
    label_median: float | None
    label_clustered_se: float | None
    # Episode-level ((ticker, owner) unit; outcome = mean member abnormal return).
    n_episodes: int
    n_tickers: int
    episode_hit_rate: float | None
    episode_mean: float | None
    episode_median: float | None
    episode_clustered_se: float | None
    low_n: bool  # n_episodes < _MIN_EPISODE_CELL_N


def form4_episode_stats(
    signals: Sequence[Signal],
    labels: Sequence[OutcomeLabel],
    *,
    excluded_ticker: str = FORM4_EPISODE_EXCLUDED_TICKER,
) -> tuple[EpisodeHorizonStats, ...]:
    """Per (horizon x {with-LUV, ex-LUV}): the Form 4 buy edge at episode AND label level (task-13).

    Read-only over ``signals`` (for the ``(ticker, owner)`` episode grouping — the owner lives in
    ``raw_payload``, parsed by :func:`form4_buy_votes`) and ``labels`` (the abnormal returns).

    An episode is one ``(ticker, owner)`` filer stream with NO gap splitting (the whole filer-on-
    name relationship is one edge unit — this is the definition the pinned Phase-7 run measured,
    reusing :func:`group_form4_vote_streams` with ``stream_window_days=None``). An episode
    contributes to horizon ``h`` when it has at least one ``ok`` member label at ``h``; its outcome
    is the MEAN of its member abnormal returns at ``h``. The label-level row pools those same member
    labels one-per-label (the inflated number). Both levels cluster by ticker for the CR0 SE.
    """
    votes = form4_buy_votes(signals)
    streams = group_form4_vote_streams(votes, stream_window_days=None)
    buy_signal_ids = {vote.signal_id for vote in votes}

    # ok abnormal returns keyed by (signal_id, horizon), Form 4 buys only.
    ars_by_signal_horizon: dict[tuple[str, int], list[float]] = defaultdict(list)
    horizons: set[int] = set()
    for label in labels:
        if label.status != "ok" or label.abnormal_return is None:
            continue
        if label.signal_id not in buy_signal_ids:
            continue
        ars_by_signal_horizon[(label.signal_id, label.horizon_days)].append(label.abnormal_return)
        horizons.add(label.horizon_days)

    rows: list[EpisodeHorizonStats] = []
    for horizon in sorted(horizons):
        for variant, drop_luv in (("with_luv", False), ("ex_luv", True)):
            episode_means: list[float] = []
            episode_by_ticker: dict[str, list[float]] = defaultdict(list)
            label_values: list[float] = []
            label_by_ticker: dict[str, list[float]] = defaultdict(list)
            for stream in streams:
                ticker = stream[0].ticker
                if drop_luv and ticker == excluded_ticker:
                    continue
                member_ars = [
                    ar
                    for vote in stream
                    for ar in ars_by_signal_horizon.get((vote.signal_id, horizon), ())
                ]
                if not member_ars:
                    continue
                episode_outcome = statistics.mean(member_ars)
                episode_means.append(episode_outcome)
                episode_by_ticker[ticker].append(episode_outcome)
                label_values.extend(member_ars)
                label_by_ticker[ticker].extend(member_ars)
            rows.append(
                EpisodeHorizonStats(
                    horizon_days=horizon,
                    variant=variant,
                    n_labels=len(label_values),
                    label_hit_rate=_hit_rate(label_values),
                    label_mean=_mean_or_none(label_values),
                    label_median=_median_or_none(label_values),
                    label_clustered_se=clustered_standard_error(label_by_ticker),
                    n_episodes=len(episode_means),
                    n_tickers=len(episode_by_ticker),
                    episode_hit_rate=_hit_rate(episode_means),
                    episode_mean=_mean_or_none(episode_means),
                    episode_median=_median_or_none(episode_means),
                    episode_clustered_se=clustered_standard_error(episode_by_ticker),
                    low_n=len(episode_means) < _MIN_EPISODE_CELL_N,
                )
            )
    return tuple(rows)


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
    # Task-13 episode-level Form 4 buy edge + the calibration constants the report anchors on.
    form4_episode_stats: tuple[EpisodeHorizonStats, ...] = field(default_factory=tuple)
    form4_expected_edge_h1: float = DEFAULT_FORM4_EXPECTED_EDGE_H1
    form4_buy_decay_half_life_days: float = DEFAULT_FORM4_BUY_DECAY_HALF_LIFE_DAYS
    # Decay weight sampled at the study horizons (day -> weight), from the anchored half-life.
    form4_buy_decay_curve: dict[int, float] = field(default_factory=dict)


def build_report(
    *,
    label_run_id: str,
    as_of: datetime,
    signals: Sequence[Signal],
    labels: Sequence[OutcomeLabel],
    window_days: int,
    thresholds: ConvictionThresholds,
    params: ConvictionParams | None = None,
    expected_edge_h1: float = DEFAULT_FORM4_EXPECTED_EDGE_H1,
    decay_half_life_days: float = DEFAULT_FORM4_BUY_DECAY_HALF_LIFE_DAYS,
) -> DiagnosticReport:
    """Assemble every report section (pure; no I/O) from a signals batch + a label-run batch.

    ``expected_edge_h1`` / ``decay_half_life_days`` are the task-13 Form 4 buy calibration constants
    the report anchors on (defaults = the new episode-level values; the job threads the config
    overrides through, so the OLD values stay recoverable without a code change).
    """
    signal_source = {s.id: source_key_for_signal_id(s.id) for s in signals}
    conviction_buckets = point_in_time_conviction_buckets(
        signals, window_days=window_days, thresholds=thresholds, params=params
    )
    decay_curve = {
        day: form4_buy_decay_weight(float(day), half_life_days=decay_half_life_days)
        for day in (1, 5, 20)
    }
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
        form4_episode_stats=form4_episode_stats(signals, labels),
        form4_expected_edge_h1=expected_edge_h1,
        form4_buy_decay_half_life_days=decay_half_life_days,
        form4_buy_decay_curve=decay_curve,
    )
