"""task-rl-10 acceptance tests for the pure walk-forward evaluation (no extras: stdlib + domain).

Results are hand-built ``PolicyFoldResult`` rows, so every gate is exercised on numbers whose
statistics are known in closed form — the env never runs here (the job's e2e test covers that).
"""

import dataclasses
import itertools
import math
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from trading.application.diagnostic import clustered_standard_error
from trading.application.rl_evaluation import (
    BEATS_BH_NO_CI_INCLUDES_ZERO,
    CI_NOT_APPLICABLE,
    CORR_NOTE,
    GATE_LEARN,
    GATE_NAMES,
    GATE_NOTDEGENERATE,
    GATE_OVERFIT,
    VS_BUY_AND_HOLD,
    GateVerdict,
    LeakageError,
    PolicyFoldResult,
    RlEvaluationError,
    RlEvaluationReport,
    RlEvaluationRequest,
    WalkForwardFold,
    build_rl_report,
    buy_and_hold_line,
    evaluate_gates,
    fold_sign_flip_statistic,
    interquartile_mean,
    make_folds,
    mean_net_log_growth,
    paired_fold_diffs,
    pooled_break_even_cost_bps,
    sign_flip_permutation_p,
    t_critical,
    two_stage_bootstrap_iqm_ci,
)
from trading.domain.rl_features import MAX_LOOKBACK, WARMUP_BARS

_AS_OF = datetime(2026, 10, 22, tzinfo=UTC)
AGENT = "ppo"
NULL = "random_matched[ppo]"
BH = "buy_and_hold"


def _row(
    policy_id: str,
    fold: int,
    seed: int,
    sharpe: float,
    *,
    cost: float = 2.0,
    exposure: float = 0.5,
    entropy: float = 0.6,
    corr: float | None = 0.3,
    train_sharpe: float | None = None,
    growth: float | None = 1.0,
    fills: tuple[float, ...] = (),
) -> PolicyFoldResult:
    return PolicyFoldResult(
        policy_id=policy_id,
        fold_index=fold,
        seed=seed,
        cost_bps=cost,
        total_return_pct=0.0,
        annualized_return_pct=0.0,
        sharpe=sharpe,
        sortino=sharpe,
        volatility=0.1,
        max_drawdown_pct=1.0,
        win_rate=None,
        annualized_turnover=1.0,
        exposure_fraction=exposure,
        n_trades=len(fills),
        n_decisions=63,
        risk_clipped_fraction=0.0,
        train_sharpe=train_sharpe,
        bh_position_corr=corr,
        action_entropy=entropy,
        frictionless_growth=growth,
        fill_sizes=fills,
    )


def _folds(k: int) -> tuple[WalkForwardFold, ...]:
    # Tiny geometry (train 10 / val 0 / purge 50 / test 5): test.lo = i + 60, so 61 + 5k bars.
    folds = make_folds(
        n_bars=61 + 5 * k, train_bars=10, val_bars=0, test_bars=5, step_bars=5, purge_bars=50
    )
    assert len(folds) == k
    return folds


def _results(
    k: int,
    agent_sharpe: Callable[[int, int], float],
    *,
    seeds: int = 5,
    null_sharpe: float = 0.0,
    bh_sharpe: float = 1.0,
    exposure: float = 0.5,
    entropy: float = 0.6,
    corr: float | None = 0.3,
    train_sharpe: float | None = None,
    growth: float | None = 1.0,
    fills: tuple[float, ...] = (),
) -> list[PolicyFoldResult]:
    rows: list[PolicyFoldResult] = []
    for fold in range(k):
        for seed in range(seeds):
            rows.append(
                _row(
                    AGENT,
                    fold,
                    seed,
                    agent_sharpe(fold, seed),
                    exposure=exposure,
                    entropy=entropy,
                    corr=corr,
                    train_sharpe=train_sharpe,
                    growth=growth,
                    fills=fills,
                )
            )
        for seed in range(3):
            rows.append(_row(NULL, fold, seed, null_sharpe + (seed - 1) * 0.1))
        rows.append(_row(BH, fold, 0, bh_sharpe))
    return rows


def _request(k: int, rows: list[PolicyFoldResult], *, looks: int = 1) -> RlEvaluationRequest:
    return RlEvaluationRequest(
        symbol="SPY",
        adjustment="all",
        dataset_sha256="0" * 64,
        folds=_folds(k),
        results=tuple(rows),
        agent_ids=(AGENT,),
        matched_random_ids={AGENT: NULL},
        buy_and_hold_id=BH,
        model_ids=("spy-ppo-test",),
        artifact_sha256s={"a.json": "1" * 64},
        cost_bps_grid=(2.0,),
        n_configurations_evaluated=looks,
        parity_ties=0,
        parity_observations=1000,
        evaluation_mode="test",
    )


def _gate(report: RlEvaluationReport, name: str) -> GateVerdict:
    return next(v for v in report.gates if v.name == name and v.policy_id == AGENT)


# --- 1-2: fold geometry -------------------------------------------------------------------------


def test_make_folds_test_windows_are_non_overlapping_and_chronological() -> None:
    folds = make_folds(n_bars=2010)
    # The frozen SPY file: K = 21 rolling folds, fold 0 exactly program doc §5.3.
    assert len(folds) == 21
    first = folds[0]
    assert (first.train, first.val, first.test) == ((0, 504), (504, 630), (680, 743))
    assert first.purge == (630, 680)
    for fold in folds:
        assert fold.train[1] <= fold.val[0] <= fold.val[1] <= fold.test[0] < fold.test[1]
        assert fold.test[1] <= 2010 - 1  # the last decision's move marks close test.hi
    for earlier, later in itertools.pairwise(folds):
        assert later.index == earlier.index + 1
        assert later.test[0] == earlier.test[1]  # step == test: contiguous, never shared
        assert later.train[0] == earlier.train[0] + 63  # rolling, not anchored
        assert later.train[1] - later.train[0] == 504
    with pytest.raises(RlEvaluationError, match="share test bars"):
        make_folds(n_bars=2010, test_bars=63, step_bars=21)
    assert make_folds(n_bars=743) == ()  # bar 743 must exist to mark the last move


def test_make_folds_purge_gap_is_at_least_the_obs_window() -> None:
    for fold in make_folds(n_bars=2010):
        assert fold.test[0] - fold.val[1] >= WARMUP_BARS
        # The first test decision's feature window [test.lo - 49, test.lo] stays past val/train.
        assert fold.test[0] - MAX_LOOKBACK > fold.val[1] - 1
        fold.assert_no_leakage(obs_window=WARMUP_BARS)
        start, length = fold.train_replay_segment()
        assert start - MAX_LOOKBACK == fold.train[0]  # IS replay reads train bars only
        assert start + length == fold.train[1] - 1


def test_assert_no_leakage_raises_when_purge_is_too_small() -> None:
    folds = make_folds(n_bars=2010, purge_bars=10)
    with pytest.raises(LeakageError, match="observation window"):
        folds[0].assert_no_leakage(obs_window=WARMUP_BARS)
    overlapping = WalkForwardFold(
        index=0, train=(0, 510), val=(504, 630), test=(680, 743), purge_bars=50
    )
    with pytest.raises(LeakageError, match="overlaps val"):
        overlapping.assert_no_leakage(obs_window=WARMUP_BARS)
    undeclared = WalkForwardFold(
        index=0, train=(0, 504), val=(504, 630), test=(660, 723), purge_bars=50
    )
    with pytest.raises(LeakageError, match="declared purge"):
        undeclared.assert_no_leakage(obs_window=WARMUP_BARS)
    request = _request(8, _results(8, lambda f, s: 1.0))
    leaky = dataclasses.replace(request, folds=(overlapping,))
    with pytest.raises(LeakageError):
        build_rl_report(leaky, as_of=_AS_OF)


# --- 3-4: clustering -----------------------------------------------------------------------------


def test_paired_fold_diffs_clusters_by_fold_not_by_seed() -> None:
    rows = _results(10, lambda f, s: 0.1 * f + 0.01 * s)
    diffs = paired_fold_diffs(rows, agent_id=AGENT, baseline_id=NULL)
    assert list(diffs) == [str(f) for f in range(10)]  # 10 clusters ...
    assert all(len(group) == 5 for group in diffs.values())  # ... of 5 seeds each
    # Each value is agent[k, s] minus the fold's MEAN null Sharpe (the 3 null seeds average 0).
    assert diffs["3"] == pytest.approx([0.3 + 0.01 * s for s in range(5)])
    # Fed to the house CR0 primitive, the fold clustering is what sets the SE: the per-fold
    # residual totals, not 50 independent draws.
    values = [v for g in diffs.values() for v in g]
    mean = sum(values) / 50
    manual = math.sqrt(sum(sum(v - mean for v in g) ** 2 for g in diffs.values())) / 50
    clustered = clustered_standard_error(diffs)
    assert clustered == pytest.approx(manual)
    iid = math.sqrt(sum((v - mean) ** 2 for v in values) / 50) / math.sqrt(50)
    assert clustered is not None and clustered > 2 * iid  # seeds are NOT independent draws
    # Rows at another cost never leak into the 2 bps pairing.
    noisy = [*rows, _row(AGENT, 0, 9, 99.0, cost=0.0)]
    assert paired_fold_diffs(noisy, agent_id=AGENT, baseline_id=NULL) == diffs


def test_clustered_se_with_one_cluster_is_not_used_as_a_result() -> None:
    report = build_rl_report(_request(1, _results(1, lambda f, s: 2.0 + s)), as_of=_AS_OF)
    assert report.low_n and report.n_folds == 1
    assert all(v.passed is not True for v in report.gates)  # no PASS from any gate
    learn = _gate(report, GATE_LEARN)
    assert learn.passed is False
    # One cluster: CR0 = 0 (the lone residual total is 0) — never turned into a t statistic.
    assert learn.standard_error == 0.0
    assert learn.statistic is None and learn.t_critical is None
    assert any(w.startswith("LOW_N") for w in report.warnings)


# --- 5-7: gates -----------------------------------------------------------------------------------


def test_gate_notdegenerate_vetoes_an_always_flat_agent() -> None:
    # Always flat: Sharpe 0.0 (not NaN) "beats" a null at -0.5, mean(d) = +0.5 > 0 ...
    rows = _results(8, lambda f, s: 0.0, null_sharpe=-0.5, exposure=0.0, entropy=0.0, corr=None)
    report = build_rl_report(_request(8, rows), as_of=_AS_OF)
    notdegenerate, learn = _gate(report, GATE_NOTDEGENERATE), _gate(report, GATE_LEARN)
    assert notdegenerate.passed is False
    assert "exposure 0.000 outside" in notdegenerate.detail
    assert "mean(d)=+0.500" in learn.detail
    assert learn.passed is None  # VOID, never PASS
    assert "VOID" in learn.detail


def test_gate_notdegenerate_vetoes_an_always_long_agent() -> None:
    rows = _results(
        8, lambda f, s: 1.5 + 0.01 * s, null_sharpe=0.0, exposure=0.98, entropy=0.08, corr=1.0
    )
    report = build_rl_report(_request(8, rows), as_of=_AS_OF)
    notdegenerate = _gate(report, GATE_NOTDEGENERATE)
    assert notdegenerate.passed is False
    assert "corr(bh)" in notdegenerate.detail and "1.000 >= 0.98" in notdegenerate.detail
    assert _gate(report, GATE_LEARN).passed is None


def test_gate_learn_uses_t_critical_not_1_96() -> None:
    # K = 8 folds x 5 identical seeds: d_k = m + a * e_k with e = (1,1,1,1,1,1,-3,-3) (sum 0),
    # so CR0 SE = 5a * sqrt(sum e^2) / 40 = a * sqrt(24) / 8 and t = m / SE. Pick t = 2.05.
    a = 0.1
    se = a * math.sqrt(24) / 8
    m = 2.05 * se
    offsets = (1, 1, 1, 1, 1, 1, -3, -3)
    rows = _results(8, lambda f, s: m + a * offsets[f], null_sharpe=0.0)
    learn = evaluate_gates(rows, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[1]
    assert learn.standard_error == pytest.approx(se)
    assert learn.statistic == pytest.approx(2.05)
    assert learn.statistic is not None and learn.statistic > 1.96  # would PASS against 1.96 ...
    assert learn.t_critical == t_critical(7) == pytest.approx(2.365, abs=1e-3)
    assert learn.passed is False  # ... but FAILS against t(K-1 = 7)
    assert "t >= t_crit" in learn.detail.split("failed:")[1]
    assert "beats fold random mean" not in learn.detail.split("failed:")[1]  # 6/8 = 75% ok


def test_t_critical_table_is_conservative_between_entries() -> None:
    assert t_critical(1) == pytest.approx(12.706, abs=1e-3)
    assert t_critical(16) == pytest.approx(2.120, abs=1e-3)
    assert t_critical(20) == pytest.approx(2.086, abs=1e-3)
    assert t_critical(35) == t_critical(30)  # next-LOWER df: a larger, conservative value
    assert t_critical(10_000) == t_critical(120) > 1.96
    values = [t_critical(df) for df in range(1, 200)]
    assert values == sorted(values, reverse=True)
    with pytest.raises(RlEvaluationError):
        t_critical(0)


def test_gate_learn_passes_only_when_every_condition_holds() -> None:
    rows = _results(10, lambda f, s: 1.0 + 0.05 * f + 0.01 * s, null_sharpe=0.0)
    report = build_rl_report(_request(10, rows), as_of=_AS_OF)
    assert _gate(report, GATE_NOTDEGENERATE).passed is True
    learn = _gate(report, GATE_LEARN)
    assert learn.passed is True
    assert learn.p_value is not None and learn.p_value < 0.05
    assert not report.low_n


def test_gate_overfit_warns_on_the_is_oos_inversion() -> None:
    rows = _results(8, lambda f, s: -0.2, train_sharpe=0.8)
    overfit = evaluate_gates(rows, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[2]
    assert overfit.name == GATE_OVERFIT
    assert overfit.passed is False  # gap 1.0 is not > 1.0, but IS 0.8 > 0.5 while OOS <= 0
    assert overfit.statistic == pytest.approx(1.0)
    assert "REPLAYED over its own train window" in overfit.detail
    calm = evaluate_gates(
        _results(8, lambda f, s: 0.3 + 0.01 * f, train_sharpe=0.5),
        agent_id=AGENT,
        random_id=NULL,
        buy_and_hold_id=BH,
    )[2]
    assert calm.passed is True


def test_vs_buy_and_hold_prints_the_literal_line_unless_the_ci_excludes_zero() -> None:
    noisy = _results(8, lambda f, s: 1.0 + (0.5 if f % 2 else -0.5), bh_sharpe=1.0)
    verdict = evaluate_gates(noisy, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[3]
    assert verdict.name == VS_BUY_AND_HOLD and verdict.passed is False
    assert buy_and_hold_line(verdict) == "beats buy-and-hold: NO (CI includes 0)"
    assert buy_and_hold_line(verdict) == BEATS_BH_NO_CI_INCLUDES_ZERO
    worse = _results(8, lambda f, s: 0.0 + 0.01 * f, bh_sharpe=1.0)
    verdict = evaluate_gates(worse, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[3]
    assert buy_and_hold_line(verdict).startswith("beats buy-and-hold: NO (CI excludes 0")
    better = _results(8, lambda f, s: 2.0 + 0.01 * f, bh_sharpe=1.0)
    verdict = evaluate_gates(better, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[3]
    assert verdict.passed is True
    assert buy_and_hold_line(verdict) == "beats buy-and-hold: YES (CI excludes 0)"


# --- 8: the permutation test respects clustering -------------------------------------------------


def test_sign_flip_permutation_respects_fold_clustering() -> None:
    diffs = {"0": [1.0, 2.0], "1": [3.0, 4.0, 5.0], "2": [-1.0]}
    # Flipping fold "1" negates ALL of its residuals together (and nothing else).
    flipped = fold_sign_flip_statistic(diffs, {"0": 1, "1": -1, "2": 1})
    assert flipped == pytest.approx((1 + 2 - 3 - 4 - 5 - 1) / 6)
    assert fold_sign_flip_statistic(diffs, {"0": 1, "1": 1, "2": 1}) == pytest.approx(14 / 6)
    # Two folds of 50 identical positive residuals: only 4 fold-sign patterns exist and half are
    # as extreme as the data, so p ~ 0.5. Flipping seeds independently would give p ~ 1e-4.
    two_folds = {"0": [0.3] * 50, "1": [0.3] * 50}
    assert sign_flip_permutation_p(two_folds, n_draws=2_000) > 0.4
    # Many folds all positive: a small p, deterministic given the seed.
    many = {str(k): [0.2 + 0.01 * k] * 5 for k in range(12)}
    p = sign_flip_permutation_p(many, n_draws=4_000, seed=3)
    assert p < 0.01
    assert p == sign_flip_permutation_p(many, n_draws=4_000, seed=3)


# --- 9: seeds -------------------------------------------------------------------------------------


def test_report_refuses_fewer_than_five_seeds() -> None:
    rows = _results(8, lambda f, s: 1.0, seeds=4)
    with pytest.raises(RlEvaluationError, match="at least 5 training seeds are required"):
        build_rl_report(_request(8, rows), as_of=_AS_OF)


# --- 10: break-even -------------------------------------------------------------------------------


def test_break_even_cost_bps_is_reported_and_monotone() -> None:
    def rows_with(growth: float) -> list[PolicyFoldResult]:
        return _results(8, lambda f, s: 0.5 + 0.01 * f, growth=growth, fills=(1.0, 1.0, 1.0, 1.0))

    rows = rows_with(1.002)
    report = build_rl_report(_request(8, rows), as_of=_AS_OF)
    be = report.break_even_cost_bps[AGENT]
    assert be is not None and be > 0
    # Same per-episode equation as rl-07: ln g + n ln(1 - c BPS) = 0 => c = (1 - g^(-1/n)) / BPS.
    assert be == pytest.approx((1 - 1.002 ** (-1 / 4)) / 1e-4, rel=1e-9)
    # Monotone: the mean net log-growth edge over always-flat strictly falls with cost, is
    # positive below the break-even, zero at it and negative above.
    edges = [
        mean_net_log_growth(rows, policy_id=AGENT, at_cost_bps=c) for c in (0.0, 2.0, 4.0, be, 6.0)
    ]
    assert edges[0] is not None and edges[0] > 0
    assert edges[3] == pytest.approx(0.0, abs=1e-12)
    assert edges[4] is not None and edges[4] < 0
    finite = [e for e in edges if e is not None]
    assert sorted(finite, reverse=True) == [edges[0], edges[1], edges[2], edges[3], edges[4]]
    # A bigger frictionless edge has a bigger break-even; no edge => 0.0; ruin => None.
    bigger = pooled_break_even_cost_bps(rows_with(1.004), policy_id=AGENT)
    assert bigger is not None and bigger > be
    assert pooled_break_even_cost_bps(rows_with(0.99), policy_id=AGENT) == 0.0
    ruined = [*rows, _row(AGENT, 0, 7, 0.0, growth=None)]
    assert pooled_break_even_cost_bps(ruined, policy_id=AGENT) is None
    # The report carries the edge at every grid cost (the monotone curve the break-even solves).
    assert report.edge_over_flat_log_growth is not None
    assert report.edge_over_flat_log_growth[AGENT]["2"] == pytest.approx(edges[1])


# --- report plumbing ------------------------------------------------------------------------------


def test_report_carries_iqm_bootstrap_ci_and_all_four_gates() -> None:
    rows = _results(8, lambda f, s: 0.1 * s, train_sharpe=0.4)
    report = build_rl_report(_request(8, rows, looks=6), as_of=_AS_OF)
    assert [v.name for v in report.gates] == list(GATE_NAMES)
    summary = next(s for s in report.summaries if s.policy_id == AGENT)
    assert summary.n_folds == 8 and summary.n_seeds == 5 and summary.n_results == 40
    assert summary.sharpe_iqm == pytest.approx(0.2)  # seeds 0.0..0.4: the middle 50% mean
    assert summary.sharpe_iqm_ci_low is not None and summary.sharpe_iqm_ci_high is not None
    assert summary.sharpe_iqm_ci_low <= summary.sharpe_iqm <= summary.sharpe_iqm_ci_high
    bh = next(s for s in report.summaries if s.policy_id == BH)
    # One run per fold still gets a CI: the two-stage bootstrap's inner stage is a no-op there,
    # so it degenerates to the fold-only bootstrap of the same estimator (review ruling, look #2).
    assert bh.sharpe_iqm_ci_low is not None and bh.sharpe_iqm_ci_high is not None
    # This fixture's buy-and-hold sharpe is the same in every fold, so the fold-only bootstrap's
    # zero width is genuine zero variance, not a short-circuit; the varying case is pinned in
    # test_iqm_and_two_stage_bootstrap_are_deterministic.
    assert bh.sharpe_iqm_ci_low <= bh.sharpe_iqm <= bh.sharpe_iqm_ci_high
    assert bh.sharpe_iqm_ci_note == ""
    assert CI_NOT_APPLICABLE == "n/a: fewer than 2 folds"
    assert summary.train_sharpe_iqm == pytest.approx(0.4)
    assert any("Deflated Sharpe" in w for w in report.warnings)  # 6 looks > 5


def test_risk_clipped_rows_raise_the_invalid_comparison_warning() -> None:
    rows = _results(8, lambda f, s: 0.1 * s)
    rows[0] = dataclasses.replace(rows[0], risk_clipped_fraction=0.2)
    report = build_rl_report(_request(8, rows), as_of=_AS_OF)
    assert any("INVALID" in w and "risk gate" in w for w in report.warnings)
    # None = n/a (no risk gate on the producing path): never a measurement, never a warning.
    unmeasured = [dataclasses.replace(r, risk_clipped_fraction=None) for r in rows]
    report = build_rl_report(_request(8, unmeasured), as_of=_AS_OF)
    assert not any("INVALID" in w for w in report.warnings)


def test_iqm_and_two_stage_bootstrap_are_deterministic() -> None:
    assert interquartile_mean([1.0, 2.0, 3.0, 4.0, 100.0, -100.0, 2.5, 3.5]) == pytest.approx(2.75)
    groups = {"0": [0.1, 0.2, 0.3], "1": [1.0, 1.1, 1.2]}
    assert two_stage_bootstrap_iqm_ci(groups, seed=1) == two_stage_bootstrap_iqm_ci(groups, seed=1)
    ci = two_stage_bootstrap_iqm_ci(groups)
    assert ci is not None
    assert ci[0] <= interquartile_mean([v for g in groups.values() for v in g]) <= ci[1]
    # One run per fold: no CI at all (a zero-width interval would be false precision).
    # One value per fold degenerates to the fold-only bootstrap of the same estimator: a real,
    # nonzero-width CI spanning the fold values (review ruling, look #2) — None only under 2 folds.
    single = two_stage_bootstrap_iqm_ci({"0": [1.3], "1": [1.4], "2": [1.2]})
    assert single is not None
    assert 1.2 <= single[0] < single[1] <= 1.4
    assert two_stage_bootstrap_iqm_ci({"0": [1.3]}) is None


# --- review B2: one never-traded episode must not veto ----------------------------------------


def test_one_constant_episode_among_healthy_ones_passes_notdegenerate() -> None:
    rows = _results(10, lambda f, s: 1.0 + 0.05 * f + 0.01 * s, null_sharpe=0.0)
    rows[3] = dataclasses.replace(rows[3], bh_position_corr=None)  # one flat (fold, seed)
    report = build_rl_report(_request(10, rows), as_of=_AS_OF)
    notdegenerate = _gate(report, GATE_NOTDEGENERATE)
    assert notdegenerate.passed is True
    assert "corr(bh)=0.30 (1/50 undefined" in notdegenerate.detail
    assert CORR_NOTE in notdegenerate.detail
    assert _gate(report, GATE_LEARN).passed is True


def test_all_constant_paths_fail_notdegenerate_on_the_corr_clause() -> None:
    rows = _results(8, lambda f, s: 1.0 + 0.05 * f, corr=None)
    report = build_rl_report(_request(8, rows), as_of=_AS_OF)
    notdegenerate = _gate(report, GATE_NOTDEGENERATE)
    assert notdegenerate.passed is False
    assert "corr(bh) undefined everywhere" in notdegenerate.detail
    assert "40/40 undefined" in notdegenerate.detail
    assert _gate(report, GATE_LEARN).passed is None


# --- review: pin the six surviving mutations ------------------------------------------------------


def test_bootstrap_is_two_stage_not_stratified_or_pooled() -> None:
    # Two folds, 20 identical runs each. Stratified (folds fixed) would always give IQM 0.5; a
    # pooled resample of 40 values almost never reaches 0 or 1. Resampling FOLDS first draws
    # (fold 0, fold 0) a quarter of the time, so the 95% interval is exactly [0, 1].
    groups = {"0": [0.0] * 20, "1": [1.0] * 20}
    assert two_stage_bootstrap_iqm_ci(groups, n_draws=2_000, seed=0) == (0.0, 1.0)


def test_gate_learn_null_is_the_matched_random_not_any_other_random() -> None:
    rows = _results(10, lambda f, s: 1.0 + 0.05 * f + 0.01 * s, null_sharpe=0.0)
    rows += [_row("random", f, s, -3.0) for f in range(10) for s in range(3)]  # uniform null
    learn = _gate(build_rl_report(_request(10, rows), as_of=_AS_OF), GATE_LEARN)
    expected = paired_fold_diffs(rows, agent_id=AGENT, baseline_id=NULL)
    mean_d = sum(v for g in expected.values() for v in g) / 50
    assert f"mean(d)={mean_d:+.3f}" in learn.detail  # vs the matched null, not the -3.0 one
    assert learn.ci_low is not None and learn.ci_high is not None
    assert (learn.ci_low + learn.ci_high) / 2 == pytest.approx(mean_d)


def test_gate_learn_fails_on_the_beat_fraction_alone() -> None:
    # 30 folds: 12 at +1.0, 18 at -0.05 -> t ~ 3.9, perm p ~ 0.001, but beats random in 40%.
    rows = _results(30, lambda f, s: 1.0 if f < 12 else -0.05, null_sharpe=0.0)
    learn = evaluate_gates(rows, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[1]
    failed = learn.detail.split("failed:")[1]
    assert learn.passed is False
    assert "beats fold random mean in >= 60% of folds" in failed
    assert "t >= t_crit" not in failed and "perm p" not in failed and "mean(d)" not in failed


def test_gate_learn_fails_when_the_permutation_p_is_not_small() -> None:
    rows = _results(10, lambda f, s: 1.0 + 0.05 * f + 0.01 * s, null_sharpe=0.0)
    passing = evaluate_gates(rows, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[1]
    assert passing.passed is True
    # One draw: p = (1 + extreme) / 2 >= 0.5 — every other condition still holds.
    one_draw = evaluate_gates(
        rows, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH, permutation_draws=1
    )[1]
    assert one_draw.p_value is not None and one_draw.p_value >= 0.5
    assert one_draw.passed is False
    assert one_draw.detail.split("failed:")[1].strip() == "perm p < 0.05"


def test_gate_overfit_flags_a_gap_above_one_on_its_own() -> None:
    # IS 0.4 (not > 0.5, so the inversion clause is off), OOS -0.7: gap 1.1 > 1.0.
    rows = _results(8, lambda f, s: -0.7 + 0.01 * f, train_sharpe=0.4)
    overfit = evaluate_gates(rows, agent_id=AGENT, random_id=NULL, buy_and_hold_id=BH)[2]
    assert overfit.statistic is not None and overfit.statistic > 1.0
    assert overfit.passed is False
