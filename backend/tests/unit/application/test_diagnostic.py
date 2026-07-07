"""Diagnostic report unit tests (Task 05, docs/RESEARCH-PLAN.md §2-§3).

Pure computation over synthetic signals + label rows (in-memory, no I/O, hand-computed
answers): source-key derivation, per-source resolution, monthly histogram, section-3 stats
(1e-9 exact), POINT-IN-TIME conviction bucketing (lookahead explicitly tested), benchmark-class
split, prior-vs-measured, and unresolvable-reason breakdown.
"""

import math
from datetime import UTC, datetime
from typing import Literal

from trading.application.diagnostic import (
    build_report,
    conviction_bucket_for_score,
    monthly_histogram,
    point_in_time_conviction_buckets,
    prior_vs_measured,
    resolution_by_source,
    source_key_for_signal_id,
    stats_by_source_horizon_benchmark,
    stats_by_source_horizon_conviction_bucket,
    unresolvable_reason_breakdown,
)
from trading.application.ports.outcome_labels import OutcomeLabel
from trading.domain.conviction import ConvictionThresholds
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def _signal(
    *,
    signal_id: str,
    created_at: datetime,
    ticker: str | None = "AAPL",
    sentiment: float = 0.5,
    politician_weight: float = 1.0,
    source_reliability: float = 0.75,
    source_type: SignalSource = SignalSource.DISCLOSED_TRADE,
) -> Signal:
    return Signal(
        id=signal_id,
        created_at=created_at,
        source_type=source_type,
        ticker=ticker,
        politician_id=None,
        sentiment=sentiment,
        politician_weight=politician_weight,
        source_reliability=source_reliability,
        provenance=Provenance.BACKFILLED,
    )


def _label(
    *,
    signal_id: str,
    horizon_days: int,
    abnormal_return: float | None,
    benchmark: str = "market",
    status: Literal["ok", "unresolvable"] = "ok",
    reason: str | None = None,
    label_run_id: str = "run-1",
    ticker: str = "AAPL",
) -> OutcomeLabel:
    return OutcomeLabel(
        signal_id=signal_id,
        ticker=ticker,
        horizon_days=horizon_days,
        abnormal_return=abnormal_return,
        benchmark=benchmark,
        status=status,
        reason=reason,
        label_run_id=label_run_id,
    )


# --- source_key_for_signal_id ---------------------------------------------------------------


def test_source_key_sec_form4_live_prefix() -> None:
    assert source_key_for_signal_id("sec-form4-0000320193-2025-01-02-P") == "sec_form4"


def test_source_key_sec_form4_historical_prefix_maps_to_same_key() -> None:
    # "sec-form4-hist-..." also starts with "sec-form4-", so it must map identically (not to
    # a separate/unknown bucket) -- this is the actual historical id format in sec_form4.py.
    assert source_key_for_signal_id("sec-form4-hist-0000320193:0001-1") == "sec_form4"


def test_source_key_house_prefix() -> None:
    assert source_key_for_signal_id("house-20040002-0-AAPL-1234567") == "house_disclosures"


def test_source_key_contract_prefix() -> None:
    assert source_key_for_signal_id("contract-AWARD123-LMT") == "usaspending"


def test_source_key_news_prefix() -> None:
    assert source_key_for_signal_id("news-pol-1-AAPL-1234567") == "news"


def test_source_key_unmapped_prefix_is_unknown() -> None:
    # e.g. senate-efd-, lobbying-, fda-approval-, fr-document-, regulations-, clinical-trial-,
    # trade- (congress_trades), sec-edgar-atom- -- none are in the Task-05 explicit source list.
    assert source_key_for_signal_id("senate-efd-pol-1-AAPL-123") == "unknown"
    assert source_key_for_signal_id("lobbying-FIL-1-AAPL") == "unknown"
    assert source_key_for_signal_id("") == "unknown"


# --- resolution_by_source (section 1) --------------------------------------------------------


def test_resolution_by_source_counts_and_rate() -> None:
    signals = [
        _signal(signal_id="sec-form4-1", created_at=_dt("2025-01-01"), ticker="AAPL"),
        _signal(signal_id="sec-form4-2", created_at=_dt("2025-01-02"), ticker=None),
        _signal(signal_id="house-1", created_at=_dt("2025-01-01"), ticker="LMT"),
    ]
    rows = {r.source: r for r in resolution_by_source(signals)}
    assert rows["sec_form4"].total_signals == 2
    assert rows["sec_form4"].ticker_resolved == 1
    assert math.isclose(rows["sec_form4"].resolution_rate or 0.0, 0.5)
    assert rows["house_disclosures"].total_signals == 1
    assert rows["house_disclosures"].ticker_resolved == 1
    assert rows["house_disclosures"].resolution_rate == 1.0


def test_resolution_rate_none_when_zero_signals() -> None:
    # No signals at all for a key that never appears -> the key doesn't appear in the output
    # (only present sources are reported); this test just proves an empty batch yields nothing.
    assert resolution_by_source([]) == ()


def test_resolution_ticker_less_signal_not_double_counted() -> None:
    signals = [_signal(signal_id="news-1", created_at=_dt("2025-01-01"), ticker=None)]
    rows = {r.source: r for r in resolution_by_source(signals)}
    assert rows["news"].total_signals == 1
    assert rows["news"].ticker_resolved == 0
    assert rows["news"].resolution_rate == 0.0


# --- monthly_histogram (section 2) -----------------------------------------------------------


def test_monthly_histogram_buckets_by_year_month() -> None:
    signals = [
        _signal(signal_id="sec-form4-1", created_at=_dt("2025-01-05")),
        _signal(signal_id="sec-form4-2", created_at=_dt("2025-01-20")),
        _signal(signal_id="sec-form4-3", created_at=_dt("2025-02-01")),
        _signal(signal_id="house-1", created_at=_dt("2025-01-05")),
    ]
    hist = monthly_histogram(signals)
    assert hist["sec_form4"] == {"2025-01": 2, "2025-02": 1}
    assert hist["house_disclosures"] == {"2025-01": 1}


def test_monthly_histogram_empty_for_no_signals() -> None:
    assert monthly_histogram([]) == {}


# --- section 3: stats by source x horizon x benchmark class, 1e-9 exact ----------------------


def test_stats_by_source_horizon_benchmark_hand_computed() -> None:
    labels = [
        _label(signal_id="s1", horizon_days=1, abnormal_return=0.10, benchmark="sector:XLK"),
        _label(signal_id="s2", horizon_days=1, abnormal_return=-0.02, benchmark="sector:XLK"),
        _label(signal_id="s3", horizon_days=1, abnormal_return=0.04, benchmark="market"),
    ]
    signal_source = {"s1": "sec_form4", "s2": "sec_form4", "s3": "sec_form4"}
    rows = {
        (r.source, r.horizon_days, r.benchmark_class): r.stats
        for r in stats_by_source_horizon_benchmark(labels, signal_source)
    }
    sector_stats = rows[("sec_form4", 1, "sector")]
    assert sector_stats.n == 2
    assert sector_stats.mean is not None and math.isclose(sector_stats.mean, 0.04, abs_tol=1e-9)
    assert sector_stats.median is not None and math.isclose(sector_stats.median, 0.04, abs_tol=1e-9)
    assert sector_stats.hit_rate is not None and math.isclose(
        sector_stats.hit_rate, 0.5, abs_tol=1e-9
    )
    # stdev of [0.10, -0.02] (sample stdev): mean=0.04, deviations +-0.06 -> var = 2*0.06^2/1
    assert sector_stats.stdev is not None and math.isclose(
        sector_stats.stdev, 0.084852813742385, rel_tol=0, abs_tol=1e-9
    )

    market_stats = rows[("sec_form4", 1, "market")]
    assert market_stats.n == 1
    assert market_stats.mean is not None and math.isclose(market_stats.mean, 0.04, abs_tol=1e-9)
    assert market_stats.median == 0.04
    assert market_stats.hit_rate == 1.0
    assert market_stats.stdev is None  # single sample: no crash, explicit None


def test_stats_excludes_unresolvable_labels() -> None:
    labels = [
        _label(signal_id="s1", horizon_days=1, abnormal_return=None, status="unresolvable"),
        _label(signal_id="s2", horizon_days=1, abnormal_return=0.05, benchmark="market"),
    ]
    signal_source = {"s1": "sec_form4", "s2": "sec_form4"}
    rows = stats_by_source_horizon_benchmark(labels, signal_source)
    assert len(rows) == 1
    assert rows[0].stats.n == 1


def test_stats_empty_group_yields_no_row() -> None:
    assert stats_by_source_horizon_benchmark([], {}) == ()


def test_benchmark_class_other_fallback() -> None:
    labels = [_label(signal_id="s1", horizon_days=1, abnormal_return=0.01, benchmark="weird")]
    rows = stats_by_source_horizon_benchmark(labels, {"s1": "sec_form4"})
    assert rows[0].benchmark_class == "other"


# --- section 4: stats by source x horizon x conviction bucket -------------------------------


def test_stats_by_source_horizon_conviction_bucket_splits_correctly() -> None:
    labels = [
        _label(signal_id="s1", horizon_days=5, abnormal_return=0.10),
        _label(signal_id="s2", horizon_days=5, abnormal_return=0.20),
    ]
    signal_source = {"s1": "sec_form4", "s2": "sec_form4"}
    conviction = {"s1": "0-40", "s2": "70+"}
    rows = {
        (r.source, r.horizon_days, r.conviction_bucket): r.stats
        for r in stats_by_source_horizon_conviction_bucket(labels, signal_source, conviction)
    }
    assert rows[("sec_form4", 5, "0-40")].n == 1
    assert rows[("sec_form4", 5, "0-40")].mean == 0.10
    assert rows[("sec_form4", 5, "70+")].n == 1
    assert rows[("sec_form4", 5, "70+")].mean == 0.20


def test_stats_by_conviction_bucket_skips_labels_without_a_bucket() -> None:
    # "Where conviction exists" (clarification 5): a ticker-less/unresolved signal has no bucket
    # entry and must be skipped, not silently grouped under a default bucket.
    labels = [_label(signal_id="s1", horizon_days=1, abnormal_return=0.05)]
    rows = stats_by_source_horizon_conviction_bucket(labels, {"s1": "sec_form4"}, {})
    assert rows == ()


# --- conviction_bucket_for_score --------------------------------------------------------------


def test_conviction_bucket_thresholds() -> None:
    thresholds = ConvictionThresholds(trade=70.0, watch=40.0)
    assert conviction_bucket_for_score(0.0, thresholds) == "0-40"
    assert conviction_bucket_for_score(39.999, thresholds) == "0-40"
    assert conviction_bucket_for_score(40.0, thresholds) == "40-70"
    assert conviction_bucket_for_score(69.999, thresholds) == "40-70"
    assert conviction_bucket_for_score(70.0, thresholds) == "70+"
    assert conviction_bucket_for_score(100.0, thresholds) == "70+"


# --- point_in_time_conviction_buckets: THE anti-lookahead test -------------------------------


def test_point_in_time_conviction_is_computed_from_trailing_window_only() -> None:
    """A LATER signal must NOT change an EARLIER signal's conviction bucket (no lookahead)."""
    thresholds = ConvictionThresholds(trade=70.0, watch=40.0)

    early = _signal(
        signal_id="s-early",
        created_at=_dt("2025-01-01"),
        ticker="AAPL",
        sentiment=0.5,
        source_reliability=0.5,
    )
    # Compute the early signal's bucket ALONE (as if no later signal ever existed).
    buckets_alone = point_in_time_conviction_buckets([early], window_days=21, thresholds=thresholds)
    early_bucket_alone = buckets_alone["s-early"]

    # Now add a much later, high-conviction corroborating signal for the SAME ticker, outside
    # the window looking backward from `early` (it is AFTER `early`, so it must never affect
    # `early`'s own score) but well within 21 days of nothing before it.
    later = _signal(
        signal_id="s-later",
        created_at=_dt("2025-01-03"),
        ticker="AAPL",
        sentiment=1.0,
        source_reliability=1.0,
        politician_weight=5.0,
    )
    buckets_with_later = point_in_time_conviction_buckets(
        [early, later], window_days=21, thresholds=thresholds
    )

    # The earlier signal's bucket (and underlying score) is IDENTICAL whether or not the later,
    # much-stronger signal exists -- proving no lookahead.
    assert buckets_with_later["s-early"] == early_bucket_alone


def test_point_in_time_conviction_excludes_signals_outside_trailing_window() -> None:
    thresholds = ConvictionThresholds(trade=70.0, watch=40.0)
    old = _signal(
        signal_id="s-old",
        created_at=_dt("2025-01-01"),
        ticker="AAPL",
        sentiment=1.0,
        source_reliability=1.0,
    )
    # 30 days later -- outside a 21-day trailing window from `new`'s own created_at.
    new = _signal(
        signal_id="s-new",
        created_at=_dt("2025-01-31"),
        ticker="AAPL",
        sentiment=0.1,
        source_reliability=0.1,
    )
    buckets = point_in_time_conviction_buckets([old, new], window_days=21, thresholds=thresholds)
    solo_new = point_in_time_conviction_buckets([new], window_days=21, thresholds=thresholds)
    # `new`'s bucket is unaffected by `old` (which fell outside its trailing window) -- same
    # result as if `new` were scored completely alone.
    assert buckets["s-new"] == solo_new["s-new"]


def test_point_in_time_conviction_ignores_other_tickers() -> None:
    thresholds = ConvictionThresholds(trade=70.0, watch=40.0)
    aapl = _signal(signal_id="s-aapl", created_at=_dt("2025-01-01"), ticker="AAPL")
    lmt = _signal(
        signal_id="s-lmt",
        created_at=_dt("2025-01-01"),
        ticker="LMT",
        sentiment=1.0,
        source_reliability=1.0,
        politician_weight=10.0,
    )
    buckets = point_in_time_conviction_buckets([aapl, lmt], window_days=21, thresholds=thresholds)
    solo_aapl = point_in_time_conviction_buckets([aapl], window_days=21, thresholds=thresholds)
    assert buckets["s-aapl"] == solo_aapl["s-aapl"]


def test_point_in_time_conviction_excludes_ticker_less_signals() -> None:
    thresholds = ConvictionThresholds(trade=70.0, watch=40.0)
    signals = [
        _signal(signal_id="s1", created_at=_dt("2025-01-01"), ticker="AAPL"),
        _signal(signal_id="s2", created_at=_dt("2025-01-01"), ticker=None),
    ]
    buckets = point_in_time_conviction_buckets(signals, window_days=21, thresholds=thresholds)
    assert "s1" in buckets
    assert "s2" not in buckets


# --- prior_vs_measured (section 5) -----------------------------------------------------------


def test_prior_vs_measured_matches_known_priors() -> None:
    labels = [
        _label(signal_id="s1", horizon_days=1, abnormal_return=0.10),
        _label(signal_id="s2", horizon_days=5, abnormal_return=-0.10),
    ]
    signal_source = {"s1": "sec_form4", "s2": "sec_form4"}
    rows = {r.source: r for r in prior_vs_measured(labels, signal_source)}
    sec_form4 = rows["sec_form4"]
    # Task 10 (docs/RESEARCH-PLAN.md; Phase 6 label run a38d3848-f1eb-4a55-8a6d-15e8f1464f7a,
    # 2026-07-07) refuted the flat 0.75 prior asserted here before Task 10 and made sec_form4
    # side-aware at the adapter; this section prints the purchase-side prior (see
    # `RELIABILITY_PRIORS`'s blend-rule comment in `application/diagnostic.py`).
    assert sec_form4.prior_reliability == 0.60
    assert sec_form4.n == 2
    assert sec_form4.measured_hit_rate is not None and math.isclose(
        sec_form4.measured_hit_rate, 0.5, abs_tol=1e-9
    )
    assert sec_form4.measured_mean_abnormal_return is not None and math.isclose(
        sec_form4.measured_mean_abnormal_return, 0.0, abs_tol=1e-9
    )


def test_prior_vs_measured_includes_all_known_prior_sources_even_with_no_labels() -> None:
    rows = {r.source: r for r in prior_vs_measured([], {})}
    for source in ("sec_form4", "house_disclosures", "usaspending", "news"):
        assert rows[source].n == 0
        assert rows[source].measured_hit_rate is None
        assert rows[source].measured_mean_abnormal_return is None
        assert rows[source].prior_reliability is not None


def test_prior_vs_measured_unknown_source_has_no_prior() -> None:
    labels = [_label(signal_id="s1", horizon_days=1, abnormal_return=0.10)]
    rows = {r.source: r for r in prior_vs_measured(labels, {"s1": "unknown"})}
    assert rows["unknown"].prior_reliability is None
    assert rows["unknown"].n == 1


# --- unresolvable_reason_breakdown (section 6) ------------------------------------------------


def test_unresolvable_reason_breakdown_counts_by_reason() -> None:
    labels = [
        _label(
            signal_id="s1",
            horizon_days=1,
            abnormal_return=None,
            status="unresolvable",
            reason="no_bars",
        ),
        _label(
            signal_id="s2",
            horizon_days=1,
            abnormal_return=None,
            status="unresolvable",
            reason="no_bars",
        ),
        _label(
            signal_id="s3",
            horizon_days=1,
            abnormal_return=None,
            status="unresolvable",
            reason="insufficient_forward_bars",
        ),
        _label(signal_id="s4", horizon_days=1, abnormal_return=0.01, status="ok"),
    ]
    breakdown = unresolvable_reason_breakdown(labels)
    assert breakdown == {"insufficient_forward_bars": 1, "no_bars": 2}


def test_unresolvable_reason_defaults_to_unknown_when_reason_missing() -> None:
    labels = [
        _label(
            signal_id="s1", horizon_days=1, abnormal_return=None, status="unresolvable", reason=None
        ),
    ]
    assert unresolvable_reason_breakdown(labels) == {"unknown": 1}


def test_unresolvable_reason_breakdown_empty_for_no_labels() -> None:
    assert unresolvable_reason_breakdown([]) == {}


# --- build_report: whole-report assembly ------------------------------------------------------


def test_build_report_assembles_all_sections() -> None:
    thresholds = ConvictionThresholds(trade=70.0, watch=40.0)
    signals = [
        _signal(signal_id="sec-form4-1", created_at=_dt("2025-01-01"), ticker="AAPL"),
        _signal(signal_id="house-1", created_at=_dt("2025-01-01"), ticker=None),
    ]
    labels = [
        _label(signal_id="sec-form4-1", horizon_days=1, abnormal_return=0.05, benchmark="market"),
        _label(
            signal_id="sec-form4-1",
            horizon_days=5,
            abnormal_return=None,
            status="unresolvable",
            reason="insufficient_forward_bars",
        ),
    ]
    report = build_report(
        label_run_id="run-xyz",
        as_of=_dt("2025-02-01"),
        signals=signals,
        labels=labels,
        window_days=21,
        thresholds=thresholds,
    )
    assert report.label_run_id == "run-xyz"
    assert report.signal_count == 2
    assert report.label_count == 2
    assert {r.source for r in report.resolution} == {"sec_form4", "house_disclosures"}
    assert report.monthly_histogram["sec_form4"] == {"2025-01": 1}
    assert len(report.horizon_benchmark_stats) == 1
    assert report.unresolvable_reasons == {"insufficient_forward_bars": 1}
    assert any(r.source == "sec_form4" for r in report.prior_vs_measured)
