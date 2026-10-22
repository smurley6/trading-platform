"""Walk-forward evaluation, the "did it learn?" gates and the report (task-rl-10; program doc §5).

PURE application code: stdlib (``math``, ``random``, ``statistics``, ``dataclasses``) +
``trading.domain`` + the house clustering primitive
:func:`trading.application.diagnostic.clustered_standard_error` (REUSED, never reimplemented). No
numpy, no ``trading.rl`` (the forbidden-deps contract covers ``trading.application``), so this
module type-checks under ``mypy --strict`` and runs without any optional extra. It CONSUMES replay
results; it never steps an environment. ``jobs/rl_eval.py`` produces them through
``trading.rl.scoring.rollout`` (Tier B, program doc §5.1) and hands them to :func:`build_rl_report`
through the composition root's ``rl_evaluator``.

**Design decision — per-fold RETRAINING, not a fixed-policy replay.** Program doc §5.3 says
"Retrain PPO once per fold", and §5.1 (Tier A on "the fold's validation slice"), §5.5 ("Five seeds
trained on fold *k*") and §5.6 (GATE-OVERFIT replays "the deterministic policy over its *own* train
slice") all assume it; the rl-08 PIT amendment calls ``--train-end 2020-10-22`` "the first fold's
train bars ``[0, 504)``". So every fold *k* has its OWN seeded policies trained on exactly
``fold.train`` (``jobs/rl_eval.py --retrain`` trains + exports them; evaluation itself only loads
the torch-free JSON exports). Consequences, all deliberate:

- Fold *k*'s TEST window is out-of-sample for fold *k*'s policies by construction:
  :meth:`WalkForwardFold.assert_no_leakage` proves ``train < val < purge < test`` with a purge of at
  least the observation window, and the job additionally checks every loaded policy's recorded
  train window equals its fold's ``train`` (an in-sample window can never masquerade as OOS).
- ``PolicyFoldResult.train_sharpe`` IS in-sample evidence for exactly the policy in that row: the
  same deterministic policy replayed over its own train window (GATE-OVERFIT's "IS"). It is a
  REPLAY of the frozen policy, not the training curve, and it is never a result.
- The fold is a valid CR0 cluster: seeds within a fold share one price path (and one train
  window); different folds have different models and disjoint test returns. A fixed policy replayed
  across folds would correlate every fold through the shared model, which fold clustering misses.

**Statistics.** Cluster key = the walk-forward FOLD, one value per ``(fold, seed)`` (never the
ticker: a single-ticker study would be G = 1). :func:`t_critical` supplies ``t(G - 1)`` at 0.975
because ``clustered_standard_error`` is CR0 with no ``G / (G - 1)`` correction (anti-conservative at
G ~ 8-21). Every headline is reported as an interquartile mean (Agarwal et al. 2021, rliable)
with a TWO-STAGE bootstrap 95% CI — resample folds with replacement, then seeds within each drawn
fold — so fold-level (market-path) uncertainty is in the interval, ALONGSIDE mean +/- clustered SE
(the one the gates read). A policy with one run per fold (buy-and-hold, SMA, always-flat) gets a
fold-only CI: the two-stage bootstrap's inner stage degenerates to a no-op there, so it is the same
estimator, and hiding it would make deterministic baselines read as exact. Never the best seed or
fold.

**Gates** (:func:`evaluate_gates`, exactly program doc §5.6): GATE-NOTDEGENERATE first (a veto —
failing it makes GATE-LEARN **VOID**, never PASS); GATE-LEARN (``K >= 8``, ``seeds >= 5``,
``mean(d) > 0``, ``t >= t_crit(K - 1)``, fold-level sign-flip ``p < 0.05``, beats the fold's random
mean in ``>= 60%`` of folds; ``d[k, s] = S_agent[k, s] - S_rand[k]`` at 2 bps, ``S_rand`` the
TURNOVER-MATCHED random policy — rl-07's ``matched_trade_probability``, built for this gate);
GATE-OVERFIT (a warning); vs BUY-AND-HOLD (measured, NOT gated: the literal line
``beats buy-and-hold: NO (CI includes 0)`` unless the CI strictly excludes zero). A ``low_n``
report (``K < 8`` or ``seeds < 5``) emits no PASS from any gate.

**Break-even cost** (task-rl-07's definition, vs ALWAYS-FLAT): the one-way cost at which a policy's
mean net log-growth over its ``(fold, seed)`` test episodes equals always-flat's (zero at every
cost) — rl-07's per-episode equation summed over episodes. Exact because a position path is
cost-invariant; the net edge is strictly decreasing in cost, so it is positive below the
break-even and negative above (:func:`mean_net_log_growth`, reported per cost as
``edge_over_flat_log_growth``).

``risk_clipped_fraction`` is part of every row; the Tier B env path has no ``RiskGate`` (only the
cut Tier B' ``Backtest`` path does), so it is ``None`` ("n/a on this path") there — never 0.0,
which would read as a measurement. The > 10% warning fires on any producer that reports one.

**corr(bh) carries no timing information here.** Both paths are 0/1 and buy-and-hold is flat for
exactly one step then long, so the agent's correlation with it is a closed-form function of its
exposure (``sqrt(x / (1 - x)) / sqrt(n - 1)``; the PIT review verified it on 1,378 rows to
1.8e-15). ``|corr| >= 0.98`` is therefore unreachable while exposure <= 0.95; the clause is kept
as spec'd (its live role is the undefined case: a never-traded path) and is labeled as
exposure-determined wherever it is shown.
"""

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from trading.application.diagnostic import clustered_standard_error
from trading.domain.rl_features import MAX_LOOKBACK, WARMUP_BARS

_MIN_FOLDS: Final = 8
_MIN_SEEDS: Final = 5
_MAX_BH_POSITION_CORR: Final = 0.98
_MIN_EXPOSURE: Final = 0.05
_MAX_EXPOSURE: Final = 0.95
_MIN_ACTION_ENTROPY: Final = 0.2
# GATE-LEARN: the agent must beat its fold's random mean in at least this share of folds.
_MIN_BEAT_FRACTION: Final = 0.60
_PERMUTATION_ALPHA: Final = 0.05
# GATE-OVERFIT flags: gap > 1.0, or IS Sharpe > 0.5 while OOS Sharpe <= 0 (§5.6).
_OVERFIT_MAX_GAP: Final = 1.0
_OVERFIT_IS_SHARPE: Final = 0.5
# Past this many looks at the same test span the report prints the Deflated-Sharpe caveat (§5.6);
# past _STOP_TUNING_LOOKS the program's stop condition applies (§9).
_DEFLATED_SHARPE_LOOKS: Final = 5
_STOP_TUNING_LOOKS: Final = 10
_MAX_RISK_CLIPPED: Final = 0.10
_IQM_TRIM: Final = 0.25
_BPS: Final = 1e-4
_BISECTION_STEPS: Final = 200
# Relative slack when counting permutation statistics "at least as extreme" (float summation order).
_PERMUTATION_TIE_SLACK: Final = 1e-12

GATE_COST_BPS: Final = 2.0
PERMUTATION_DRAWS: Final = 10_000
BOOTSTRAP_DRAWS: Final = 2_000
# The gate-rule revision every look records (the look ledger's config hash includes it).
GATE_RULES_VERSION: Final = (
    "rl-10 v2: GATE-NOTDEGENERATE corr(bh) = mean |corr| over episodes where it is defined; the "
    "corr clause vetoes only when NO episode has it defined (v1, look #1: any undefined vetoed)"
)
CI_NOT_APPLICABLE: Final = "n/a: fewer than 2 folds"
CORR_NOTE: Final = "corr(bh) is determined by exposure here; no timing information"

GATE_NOTDEGENERATE: Final = "GATE-NOTDEGENERATE"
GATE_LEARN: Final = "GATE-LEARN"
GATE_OVERFIT: Final = "GATE-OVERFIT"
VS_BUY_AND_HOLD: Final = "vs BUY-AND-HOLD"
GATE_NAMES: Final = (GATE_NOTDEGENERATE, GATE_LEARN, GATE_OVERFIT, VS_BUY_AND_HOLD)
BEATS_BH_NO_CI_INCLUDES_ZERO: Final = "beats buy-and-hold: NO (CI includes 0)"
BEATS_BH_NO_CI_BELOW_ZERO: Final = "beats buy-and-hold: NO (CI excludes 0: the agent is worse)"
BEATS_BH_YES: Final = "beats buy-and-hold: YES (CI excludes 0)"

# Two-sided 97.5% quantiles of Student's t, t(df, 0.975), to 4 decimals. Source: standard
# critical-value tables (e.g. NIST/SEMATECH e-Handbook of Statistical Methods §1.3.6.7.2,
# "Critical values of the Student's t distribution"); re-derived for this module by numerically
# integrating the t density (Simpson, 20k panels) and bisecting — every entry agreed to 4 dp.
# The stdlib has no t inverse, hence the table. A df between entries uses the NEXT-LOWER tabulated
# df (a larger critical value: conservative); df > 120 uses df = 120's value, never 1.96.
_T_CRIT_975: Final[Mapping[int, float]] = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
    11: 2.2010,
    12: 2.1788,
    13: 2.1604,
    14: 2.1448,
    15: 2.1314,
    16: 2.1199,
    17: 2.1098,
    18: 2.1009,
    19: 2.0930,
    20: 2.0860,
    21: 2.0796,
    22: 2.0739,
    23: 2.0687,
    24: 2.0639,
    25: 2.0595,
    26: 2.0555,
    27: 2.0518,
    28: 2.0484,
    29: 2.0452,
    30: 2.0423,
    40: 2.0211,
    50: 2.0086,
    60: 2.0003,
    80: 1.9901,
    100: 1.9840,
    120: 1.9799,
}


class LeakageError(ValueError):
    """A walk-forward fold would let training/validation information reach a test decision."""


class RlEvaluationError(ValueError):
    """The evaluation inputs cannot support an honest report (fails closed, never guesses)."""


def _require_int(value: object, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RlEvaluationError(f"{what} must be an int, got {value!r}")
    return value


# --- fold geometry --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One rolling walk-forward fold over bar INDICES. Every window is HALF-OPEN ``[lo, hi)``.

    ``train`` is what the fold's policies were trained on; ``val`` is reserved for Tier A
    model selection (unused by the fixed-budget training — never quoted); the purge
    ``[val.hi, test.lo)`` keeps every test observation window clear of val/train; ``test`` holds
    the DECISION bars. A test replay of ``test = [a, b)`` makes ``b - a`` decisions at bars
    ``a .. b-1``, reads observation windows back to bar ``a - MAX_LOOKBACK`` and marks its last
    step at close ``b`` (so the fold's test RETURNS are the moves ``C_a -> C_{a+1}`` ..
    ``C_{b-1} -> C_b``, disjoint across folds when ``step == test``).
    """

    index: int
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    purge_bars: int

    @property
    def purge(self) -> tuple[int, int]:
        return (self.val[1], self.test[0])

    def assert_no_leakage(self, *, obs_window: int) -> None:
        """Raise :class:`LeakageError` unless ``train`` < ``val`` < purge < ``test`` are ordered,
        non-overlapping and the purge gap is at least ``obs_window`` bars (``WARMUP_BARS`` = 50:
        the feature window ending at the first test decision covers bars
        ``[test.lo - 49, test.lo]``, so a 50-bar gap keeps it — and everything later — out of
        val and train with one bar to spare)."""
        _require_int(obs_window, what="obs_window")
        if obs_window < 1:
            raise LeakageError(f"obs_window must be >= 1, got {obs_window}")
        (train_lo, train_hi), (val_lo, val_hi), (test_lo, test_hi) = self.train, self.val, self.test
        where = f"fold {self.index}"
        if not 0 <= train_lo < train_hi:
            raise LeakageError(f"{where}: train window {self.train} is empty or negative")
        if not val_lo <= val_hi:
            raise LeakageError(f"{where}: val window {self.val} is inverted")
        if not test_lo < test_hi:
            raise LeakageError(f"{where}: test window {self.test} is empty")
        if train_hi > val_lo:
            raise LeakageError(f"{where}: train {self.train} overlaps val {self.val}")
        if val_hi > test_lo:
            raise LeakageError(f"{where}: val {self.val} overlaps test {self.test}")
        gap = test_lo - val_hi
        if gap < self.purge_bars:
            raise LeakageError(
                f"{where}: declared purge {self.purge_bars} bars but the gap between val and "
                f"test is {gap}"
            )
        if gap < obs_window:
            raise LeakageError(
                f"{where}: purge gap {gap} bars < the {obs_window}-bar observation window — the "
                f"first test decision's features (bars [{test_lo - obs_window + 1}, {test_lo}]) "
                f"would read bars at or before val/train's end ({val_hi})"
            )

    def test_segment(self) -> tuple[int, int]:
        """``(start, length)`` of the out-of-sample test replay: decisions at ``test``."""
        return self.test[0], self.test[1] - self.test[0]

    def train_replay_segment(self, *, lookback: int = MAX_LOOKBACK) -> tuple[int, int]:
        """``(start, length)`` of the IN-SAMPLE replay over the policy's own train window, kept
        entirely inside ``train``: the first decision's observation reaches back exactly to
        ``train.lo`` and the last step marks close ``train.hi - 1``."""
        start = self.train[0] + lookback
        length = self.train[1] - 1 - start
        if length < 1:
            raise RlEvaluationError(
                f"fold {self.index}: train window {self.train} cannot hold a replay after the "
                f"{lookback}-bar lookback"
            )
        return start, length


def make_folds(
    *,
    n_bars: int,
    train_bars: int = 504,
    val_bars: int = 126,
    test_bars: int = 63,
    step_bars: int = 63,
    purge_bars: int = WARMUP_BARS,
) -> tuple[WalkForwardFold, ...]:
    """Rolling (not anchored) walk-forward folds (program doc §5.3): fold *k* starts at
    ``i = k * step_bars`` with ``train = [i, i + train)``, ``val`` right after, a ``purge_bars``
    gap, then ``test``. A fold is emitted only while its test replay fits: the last decision's move
    marks close ``test.hi``, so ``test.hi <= n_bars - 1``.

    ``step_bars < test_bars`` is refused (folds would share test bars and the fold-clustered SE
    would be too tight). Leakage is NOT asserted here — call
    :meth:`WalkForwardFold.assert_no_leakage` per fold (``build_rl_report`` and the job do).
    """
    for name, value in (
        ("n_bars", n_bars),
        ("train_bars", train_bars),
        ("val_bars", val_bars),
        ("test_bars", test_bars),
        ("step_bars", step_bars),
        ("purge_bars", purge_bars),
    ):
        _require_int(value, what=name)
    if train_bars < 1 or test_bars < 1 or step_bars < 1:
        raise RlEvaluationError("train_bars, test_bars and step_bars must be >= 1")
    if val_bars < 0 or purge_bars < 0:
        raise RlEvaluationError("val_bars and purge_bars must be >= 0")
    if step_bars < test_bars:
        raise RlEvaluationError(
            f"step_bars ({step_bars}) < test_bars ({test_bars}): folds would share test bars and "
            "the fold-clustered SE would be too tight"
        )
    folds: list[WalkForwardFold] = []
    start = 0
    while True:
        train = (start, start + train_bars)
        val = (train[1], train[1] + val_bars)
        test_lo = val[1] + purge_bars
        test = (test_lo, test_lo + test_bars)
        if test[1] > n_bars - 1:
            break
        folds.append(
            WalkForwardFold(
                index=len(folds), train=train, val=val, test=test, purge_bars=purge_bars
            )
        )
        start += step_bars
    return tuple(folds)


# --- results, verdicts, report --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyFoldResult:
    """One policy x fold x seed x cost replay over the fold's TEST window, as numbers.

    ``seed`` is the training seed for a PPO policy and the RNG seed for the random baseline
    (deterministic baselines use 0). ``train_sharpe`` is the SAME policy replayed over its own
    train window at the same cost — in-sample REPLAY evidence for GATE-OVERFIT, never a result;
    ``None`` for baselines (nothing was trained). ``bh_position_corr`` is the Pearson correlation of
    the held-position path with buy-and-hold's on the same window (``None`` when either path is
    constant) — corr(bh) is determined by exposure here; no timing information (see the module
    docstring), never independent evidence. ``risk_clipped_fraction`` is ``None`` when the
    producing path has no risk gate (the env path: n/a, not 0%).
    ``frictionless_growth`` (final equity with the costs divided out; ``None`` when ruined) and
    ``fill_sizes`` (``|delta position|`` per fill) feed the pooled break-even.
    """

    policy_id: str
    fold_index: int
    seed: int
    cost_bps: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe: float
    sortino: float
    volatility: float
    max_drawdown_pct: float
    win_rate: float | None
    annualized_turnover: float
    exposure_fraction: float
    n_trades: int
    n_decisions: int
    risk_clipped_fraction: float | None
    train_sharpe: float | None
    bh_position_corr: float | None
    action_entropy: float
    longest_holding_streak: int = 0
    final_equity: float = 1.0
    break_even_cost_bps: float | None = None
    frictionless_growth: float | None = 1.0
    fill_sizes: tuple[float, ...] = ()
    ruined: bool = False


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """One gate's verdict for one policy. ``passed``: ``True`` PASS, ``False`` FAIL (WARN for
    GATE-OVERFIT; "does not beat" for vs BUY-AND-HOLD), ``None`` VOID."""

    name: str
    policy_id: str
    passed: bool | None
    statistic: float | None = None
    standard_error: float | None = None
    t_critical: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PolicySummary:
    """One table row: a policy at one cost, aggregated over its ``(fold, seed)`` results.
    ``*_iqm`` are interquartile means; the Sharpe also carries mean +/- fold-clustered SE and the
    two-stage bootstrap 95% CI of its IQM — ``None`` (with ``sharpe_iqm_ci_note``) only when
    fewer than 2 folds exist, where no cluster variation can be resampled."""

    policy_id: str
    cost_bps: float
    n_folds: int
    n_seeds: int
    n_results: int
    annualized_return_pct_iqm: float
    sharpe_iqm: float
    sortino_iqm: float
    max_drawdown_pct_iqm: float
    annualized_turnover_iqm: float
    win_rate_iqm: float | None
    exposure_fraction_iqm: float
    n_trades_iqm: float
    action_entropy_iqm: float
    longest_holding_streak_iqm: float
    sharpe_mean: float
    sharpe_clustered_se: float | None
    sharpe_iqm_ci_low: float | None
    sharpe_iqm_ci_high: float | None
    train_sharpe_iqm: float | None
    sharpe_iqm_ci_note: str = ""


@dataclass(frozen=True, slots=True)
class RlEvaluationRequest:
    """Everything :func:`build_rl_report` needs: the folds, every replay result and the roles.

    ``agent_ids`` are the PPO policy groups to gate (e.g. ``ppo`` and ``ppo_untrained``);
    ``matched_random_ids[agent]`` is the random policy turnover-matched to that agent (the
    GATE-LEARN null); ``buy_and_hold_id`` is the env buy-and-hold baseline.
    """

    symbol: str
    adjustment: str
    dataset_sha256: str
    folds: tuple[WalkForwardFold, ...]
    results: tuple[PolicyFoldResult, ...]
    agent_ids: tuple[str, ...]
    matched_random_ids: Mapping[str, str]
    buy_and_hold_id: str
    model_ids: tuple[str, ...]
    artifact_sha256s: Mapping[str, str]
    cost_bps_grid: tuple[float, ...]
    n_configurations_evaluated: int
    parity_ties: int
    parity_observations: int
    evaluation_mode: str
    n_configurations_source: str = ""
    gate_cost_bps: float = GATE_COST_BPS
    obs_window: int = WARMUP_BARS
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RlEvaluationReport:
    """The rl-10 report (serialized by the job with ``dataclasses.asdict``)."""

    as_of: datetime
    symbol: str
    model_ids: tuple[str, ...]
    artifact_sha256s: dict[str, str]
    n_folds: int
    n_seeds: int
    low_n: bool
    cost_bps_grid: tuple[float, ...]
    break_even_cost_bps: dict[str, float | None]
    adjustment: str
    results: tuple[PolicyFoldResult, ...]
    gates: tuple[GateVerdict, ...]
    n_configurations_evaluated: int
    parity_ties: int
    parity_observations: int = 0
    dataset_sha256: str = ""
    evaluation_mode: str = ""
    gate_cost_bps: float = GATE_COST_BPS
    folds: tuple[WalkForwardFold, ...] = ()
    summaries: tuple[PolicySummary, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    n_configurations_source: str = ""
    gate_rules_version: str = GATE_RULES_VERSION
    # policy -> {cost_bps: mean net log-growth per (fold, seed) episode vs always-flat (0.0)}.
    edge_over_flat_log_growth: dict[str, dict[str, float | None]] | None = None


# --- statistics -----------------------------------------------------------------------------------


def t_critical(df: int) -> float:
    """t(G-1) at 0.975. clustered_standard_error is CR0 with NO G/(G-1) correction and is
    anti-conservative at G ~ 8-17, so we do NOT use 1.96.

    ``df`` between tabulated entries uses the next-LOWER tabulated df (conservative); ``df > 120``
    uses df = 120's value. ``df < 1`` raises (one cluster has no t reference)."""
    _require_int(df, what="df")
    if df < 1:
        raise RlEvaluationError(f"t_critical needs df >= 1, got {df}")
    usable = max(key for key in _T_CRIT_975 if key <= df)
    return _T_CRIT_975[usable]


def interquartile_mean(values: Sequence[float]) -> float:
    """Mean after trimming ``floor(0.25 * n)`` values from each end of the sorted sample
    (``scipy.stats.trim_mean(x, 0.25)``, as rliable reports it)."""
    if not values:
        raise RlEvaluationError("IQM of an empty sample")
    ordered = sorted(values)
    cut = int(_IQM_TRIM * len(ordered))
    kept = ordered[cut : len(ordered) - cut]
    return math.fsum(kept) / len(kept)


def two_stage_bootstrap_iqm_ci(
    values_by_fold: Mapping[str, Sequence[float]],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% percentile CI of the pooled IQM under a TWO-STAGE (cluster) bootstrap: each draw
    resamples the FOLDS with replacement, then each drawn fold's values with replacement within
    that fold, pools them and takes the IQM — so the interval carries fold-level (market-path)
    uncertainty, not just seed noise. With one value per fold the inner stage is a no-op and the
    draw degenerates to the fold-only bootstrap, which is the SAME estimator, so single-run
    policies (buy-and-hold, SMA, always-flat) get a real fold-level CI too. Returns ``None`` only
    with fewer than 2 folds (no cluster variation to resample). Deterministic given ``seed``."""
    groups = [list(v) for v in values_by_fold.values() if v]
    if not groups:
        raise RlEvaluationError("bootstrap of an empty sample")
    if n_draws < 2:
        raise RlEvaluationError(f"n_draws must be >= 2, got {n_draws}")
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_draws):
        pooled: list[float] = []
        for group in rng.choices(groups, k=len(groups)):
            pooled.extend(rng.choices(group, k=len(group)))
        stats.append(interquartile_mean(pooled))
    cuts = statistics.quantiles(stats, n=40, method="inclusive")
    return cuts[0], cuts[-1]


def paired_fold_diffs(
    results: Sequence[PolicyFoldResult],
    *,
    agent_id: str,
    baseline_id: str,
    metric: str = "sharpe",
    cost_bps: float = GATE_COST_BPS,
) -> dict[str, list[float]]:
    """CLUSTER KEY = fold index, one value per (fold, seed). Seeds within a fold share ONE price
    path and must NOT count as independent draws — feed straight into
    trading.application.diagnostic.clustered_standard_error.

    ``d[k, s] = agent[k, s].metric - mean_s'(baseline[k, s'].metric)`` over the results at
    ``cost_bps`` (the baseline's mean over its own seeds — 30 for random, 1 for deterministic
    baselines). Keys are ``str(fold_index)`` in ascending fold order. Raises when a fold has the
    agent but not the baseline.
    """
    agent: dict[int, list[float]] = defaultdict(list)
    baseline: dict[int, list[float]] = defaultdict(list)
    for row in results:
        if row.cost_bps != cost_bps:
            continue
        if row.policy_id == agent_id:
            agent[row.fold_index].append(_metric(row, metric))
        elif row.policy_id == baseline_id:
            baseline[row.fold_index].append(_metric(row, metric))
    out: dict[str, list[float]] = {}
    for fold in sorted(agent):
        if fold not in baseline:
            raise RlEvaluationError(
                f"fold {fold}: {agent_id!r} has results at {cost_bps} bps but {baseline_id!r} "
                "has none"
            )
        reference = math.fsum(baseline[fold]) / len(baseline[fold])
        out[str(fold)] = [value - reference for value in agent[fold]]
    return out


def _metric(row: PolicyFoldResult, metric: str) -> float:
    value = getattr(row, metric, None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RlEvaluationError(f"metric {metric!r} is not a number on {row.policy_id!r}")
    return float(value)


def fold_sign_flip_statistic(
    diffs_by_fold: Mapping[str, Sequence[float]], signs: Mapping[str, int]
) -> float:
    """The mean of all residuals after multiplying EVERY residual of fold ``g`` by ``signs[g]``
    (+1 / -1) — a whole fold flips together, never a single seed."""
    total = 0.0
    n = 0
    for key, values in diffs_by_fold.items():
        sign = signs[key]
        if sign not in (1, -1):
            raise RlEvaluationError(f"sign for fold {key} must be +1 or -1, got {sign}")
        total += sign * math.fsum(values)
        n += len(values)
    if n == 0:
        raise RlEvaluationError("no residuals to flip")
    return total / n


def sign_flip_permutation_p(
    diffs_by_fold: Mapping[str, Sequence[float]], *, n_draws: int = PERMUTATION_DRAWS, seed: int = 0
) -> float:
    """Flips the sign of a WHOLE fold's residuals together (respects clustering).

    Two-sided Monte-Carlo p-value of ``mean(d)`` under the null that each fold's total is
    symmetric about zero: ``(1 + #{|stat*| >= |stat|}) / (n_draws + 1)`` over ``n_draws`` random
    fold-sign vectors (``random.Random(seed)``). Two-sided to match the 0.975 ``t`` critical value.
    """
    keys = [key for key, values in diffs_by_fold.items() if values]
    if not keys:
        raise RlEvaluationError("permutation test of an empty sample")
    if n_draws < 1:
        raise RlEvaluationError(f"n_draws must be >= 1, got {n_draws}")
    data = {key: diffs_by_fold[key] for key in keys}
    observed = abs(fold_sign_flip_statistic(data, dict.fromkeys(keys, 1)))
    threshold = observed - _PERMUTATION_TIE_SLACK * max(1.0, observed)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_draws):
        signs = {key: (1 if rng.random() < 0.5 else -1) for key in keys}
        if abs(fold_sign_flip_statistic(data, signs)) >= threshold:
            extreme += 1
    return (1 + extreme) / (n_draws + 1)


def mean_net_log_growth(
    results: Sequence[PolicyFoldResult],
    *,
    policy_id: str,
    at_cost_bps: float,
    source_cost_bps: float = GATE_COST_BPS,
) -> float | None:
    """The policy's mean log final equity per ``(fold, seed)`` test episode if its (cost-invariant)
    position paths were charged ``at_cost_bps`` one-way — its edge over always-flat (0.0 at every
    cost). Read from the rows at ``source_cost_bps``. ``None`` if any episode was ruined."""
    rows = _rows(results, policy_id, source_cost_bps)
    total = 0.0
    for row in rows:
        if row.frictionless_growth is None:
            return None
        total += math.log(row.frictionless_growth)
        total += math.fsum(math.log1p(-at_cost_bps * _BPS * size) for size in row.fill_sizes)
    return total / len(rows)


def pooled_break_even_cost_bps(
    results: Sequence[PolicyFoldResult],
    *,
    policy_id: str,
    source_cost_bps: float = GATE_COST_BPS,
) -> float | None:
    """task-rl-07's break-even (vs ALWAYS-FLAT), pooled over every ``(fold, seed)`` test episode:
    the one-way cost ``c`` (bps) with ``sum_e [ln g_e + sum_k ln(1 - c * BPS * t_ek)] = 0`` — where
    the policy's mean net log-growth equals always-flat's. Unique (strictly decreasing in ``c``);
    bisection. ``0.0`` when there is no frictionless edge to lose; ``None`` if any episode was
    ruined (its path is not cost-invariant)."""
    rows = _rows(results, policy_id, source_cost_bps)
    if any(row.frictionless_growth is None for row in rows):
        return None
    target = math.fsum(math.log(row.frictionless_growth or 1.0) for row in rows)
    sizes = [size for row in rows for size in row.fill_sizes]
    if target <= 0:
        return 0.0
    largest = max(sizes, default=0.0)
    if largest <= 0:
        raise RlEvaluationError(f"{policy_id!r}: frictionless growth with nothing traded")

    def net(c: float) -> float:
        return target + math.fsum(math.log1p(-c * _BPS * size) for size in sizes)

    lo, hi = 0.0, 1.0 / (_BPS * largest)
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break
        if net(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def _rows(
    results: Sequence[PolicyFoldResult], policy_id: str, cost_bps: float
) -> list[PolicyFoldResult]:
    rows = [r for r in results if r.policy_id == policy_id and r.cost_bps == cost_bps]
    if not rows:
        raise RlEvaluationError(f"no results for {policy_id!r} at {cost_bps} bps")
    return rows


def _by_fold(rows: Sequence[PolicyFoldResult], value: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r.fold_index, r.seed)):
        out[str(row.fold_index)].append(_metric(row, value))
    return dict(out)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _counts(rows: Sequence[PolicyFoldResult]) -> tuple[int, int]:
    """``(n_folds, n_seeds)`` — seeds = the SMALLEST per-fold seed count (the honest floor)."""
    seeds: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        seeds[row.fold_index].add(row.seed)
    if not seeds:
        return 0, 0
    return len(seeds), min(len(s) for s in seeds.values())


# --- gates ----------------------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = "+.3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _not_degenerate(agent_id: str, rows: Sequence[PolicyFoldResult], *, low_n: bool) -> GateVerdict:
    exposure = _mean([r.exposure_fraction for r in rows])
    entropy = _mean([r.action_entropy for r in rows])
    # Mean |corr| over the episodes where it is defined (a never-traded episode has a constant
    # path and no correlation); the corr clause vetoes only when NO episode has it defined.
    defined = [abs(r.bh_position_corr) for r in rows if r.bh_position_corr is not None]
    n_undefined = len(rows) - len(defined)
    abs_corr = _mean(defined) if defined else None
    failures = []
    if not _MIN_EXPOSURE <= exposure <= _MAX_EXPOSURE:
        failures.append(f"exposure {exposure:.3f} outside [{_MIN_EXPOSURE}, {_MAX_EXPOSURE}]")
    if not entropy > _MIN_ACTION_ENTROPY:
        failures.append(f"action entropy {entropy:.3f} <= {_MIN_ACTION_ENTROPY} nats")
    if abs_corr is None:
        failures.append("every position path is constant (corr(bh) undefined everywhere)")
    elif not abs_corr < _MAX_BH_POSITION_CORR:
        failures.append(f"|corr(bh)| {abs_corr:.3f} >= {_MAX_BH_POSITION_CORR}")
    summary = (
        f"exposure={exposure:.2f}  entropy={entropy:.2f}  corr(bh)={_fmt(abs_corr, '.2f')} "
        f"({n_undefined}/{len(rows)} undefined; {CORR_NOTE})"
    )
    passed: bool | None
    if failures:
        passed, detail = False, f"{summary} — DEGENERATE: {'; '.join(failures)}"
    elif low_n:
        passed, detail = None, f"{summary} — VOID: low_n (no gate may pass)"
    else:
        passed, detail = True, summary
    return GateVerdict(
        name=GATE_NOTDEGENERATE,
        policy_id=agent_id,
        passed=passed,
        statistic=exposure,
        detail=detail,
    )


def _learn(
    agent_id: str,
    diffs: Mapping[str, Sequence[float]],
    *,
    n_folds: int,
    n_seeds: int,
    vetoed: bool,
    permutation_draws: int,
    permutation_seed: int,
) -> GateVerdict:
    values = [v for group in diffs.values() for v in group]
    mean_d = _mean(values)
    se = clustered_standard_error(diffs)
    t_crit = t_critical(n_folds - 1) if n_folds >= 2 else None
    t_stat = mean_d / se if se is not None and se > 0 else None
    ci = (
        (mean_d - t_crit * se, mean_d + t_crit * se)
        if t_crit is not None and se is not None
        else (None, None)
    )
    p = sign_flip_permutation_p(diffs, n_draws=permutation_draws, seed=permutation_seed)
    beat = sum(1 for group in diffs.values() if _mean(group) > 0) / len(diffs)
    checks = {
        f"K={n_folds} >= {_MIN_FOLDS}": n_folds >= _MIN_FOLDS,
        f"seeds={n_seeds} >= {_MIN_SEEDS}": n_seeds >= _MIN_SEEDS,
        "mean(d) > 0": mean_d > 0,
        "t >= t_crit": t_stat is not None and t_crit is not None and t_stat >= t_crit,
        f"perm p < {_PERMUTATION_ALPHA}": p < _PERMUTATION_ALPHA,
        f"beats fold random mean in >= {_MIN_BEAT_FRACTION:.0%} of folds": (
            beat >= _MIN_BEAT_FRACTION
        ),
    }
    stats = (
        f"mean(d)={mean_d:+.3f}  CR0 SE={_fmt(se, '.3f')}  t={_fmt(t_stat, '.2f')} "
        f"(t_crit({n_folds - 1})={_fmt(t_crit, '.3f')})  perm p={p:.3f}  "
        f"beats random in {beat:.0%} of folds  K={n_folds}"
    )
    failed = [name for name, ok in checks.items() if not ok]
    passed: bool | None
    if vetoed:
        passed, detail = None, f"{stats} — VOID: GATE-NOTDEGENERATE failed (veto)"
    elif failed:
        passed, detail = False, f"{stats} — failed: {', '.join(failed)}"
    else:
        passed, detail = True, stats
    return GateVerdict(
        name=GATE_LEARN,
        policy_id=agent_id,
        passed=passed,
        statistic=t_stat,
        standard_error=se,
        t_critical=t_crit,
        ci_low=ci[0],
        ci_high=ci[1],
        p_value=p,
        detail=detail,
    )


def _overfit(agent_id: str, rows: Sequence[PolicyFoldResult], *, low_n: bool) -> GateVerdict:
    if any(r.train_sharpe is None for r in rows):
        return GateVerdict(
            name=GATE_OVERFIT,
            policy_id=agent_id,
            passed=None,
            detail="VOID: no in-sample replay (train_sharpe) recorded",
        )
    gaps: dict[str, list[float]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r.fold_index, r.seed)):
        gaps[str(row.fold_index)].append((row.train_sharpe or 0.0) - row.sharpe)
    n_folds = len(gaps)
    gap = _mean([v for group in gaps.values() for v in group])
    se = clustered_standard_error(gaps)
    t_crit = t_critical(n_folds - 1) if n_folds >= 2 else None
    ci = (
        (gap - t_crit * se, gap + t_crit * se)
        if t_crit is not None and se is not None
        else (None, None)
    )
    mean_is = _mean([r.train_sharpe or 0.0 for r in rows])
    mean_oos = _mean([r.sharpe for r in rows])
    flagged = gap > _OVERFIT_MAX_GAP or (mean_is > _OVERFIT_IS_SHARPE and mean_oos <= 0)
    stats = (
        f"IS sharpe {mean_is:.2f} vs OOS {mean_oos:.2f} (gap {gap:.2f}, CR0 SE "
        f"{_fmt(se, '.2f')}); IS = the same policy REPLAYED over its own train window"
    )
    passed: bool | None
    if flagged:
        passed, detail = False, f"{stats} — WARN: overfit"
    elif low_n:
        passed, detail = None, f"{stats} — VOID: low_n"
    else:
        passed, detail = True, stats
    return GateVerdict(
        name=GATE_OVERFIT,
        policy_id=agent_id,
        passed=passed,
        statistic=gap,
        standard_error=se,
        t_critical=t_crit,
        ci_low=ci[0],
        ci_high=ci[1],
        detail=detail,
    )


def _vs_buy_and_hold(
    agent_id: str, diffs: Mapping[str, Sequence[float]], *, n_folds: int, low_n: bool
) -> GateVerdict:
    mean_d = _mean([v for group in diffs.values() for v in group])
    se = clustered_standard_error(diffs)
    t_crit = t_critical(n_folds - 1) if n_folds >= 2 else None
    ci_low = ci_high = None
    if t_crit is not None and se is not None:
        ci_low, ci_high = mean_d - t_crit * se, mean_d + t_crit * se
    line = BEATS_BH_NO_CI_INCLUDES_ZERO
    passed: bool | None = False
    if ci_low is not None and ci_high is not None:
        if ci_low > 0:
            line, passed = BEATS_BH_YES, (None if low_n else True)
        elif ci_high < 0:
            line = BEATS_BH_NO_CI_BELOW_ZERO
    detail = (
        f"mean diff {mean_d:+.2f} sharpe, 95% CI [{_fmt(ci_low, '+.2f')}, "
        f"{_fmt(ci_high, '+.2f')}] (t_crit({n_folds - 1})={_fmt(t_crit, '.3f')}) — measured, "
        f"NOT gated | {line}"
    )
    return GateVerdict(
        name=VS_BUY_AND_HOLD,
        policy_id=agent_id,
        passed=passed,
        statistic=mean_d,
        standard_error=se,
        t_critical=t_crit,
        ci_low=ci_low,
        ci_high=ci_high,
        detail=detail,
    )


def buy_and_hold_line(verdict: GateVerdict) -> str:
    """The literal beats-buy-and-hold line carried in a vs BUY-AND-HOLD verdict's detail."""
    if verdict.name != VS_BUY_AND_HOLD:
        raise RlEvaluationError(f"{verdict.name} is not the buy-and-hold comparison")
    return verdict.detail.rsplit(" | ", 1)[-1]


def evaluate_gates(
    results: Sequence[PolicyFoldResult],
    *,
    agent_id: str,
    random_id: str,
    buy_and_hold_id: str,
    cost_bps: float = GATE_COST_BPS,
    permutation_draws: int = PERMUTATION_DRAWS,
    permutation_seed: int = 0,
) -> tuple[GateVerdict, GateVerdict, GateVerdict, GateVerdict]:
    """The four §5.6 verdicts for one agent, in order: GATE-NOTDEGENERATE (evaluated FIRST; its
    failure makes GATE-LEARN VOID), GATE-LEARN, GATE-OVERFIT, vs BUY-AND-HOLD.

    Reads the rows at ``cost_bps`` (2 bps). ``random_id`` is the GATE-LEARN null
    (``S_rand[k]`` = its mean Sharpe over seeds on fold *k*). ``K < 8`` or ``seeds < 5`` is
    ``low_n``: GATE-LEARN fails and no other verdict may be PASS.
    """
    rows = [r for r in results if r.policy_id == agent_id and r.cost_bps == cost_bps]
    if not rows:
        raise RlEvaluationError(f"no results for agent {agent_id!r} at {cost_bps} bps")
    n_folds, n_seeds = _counts(rows)
    low_n = n_folds < _MIN_FOLDS or n_seeds < _MIN_SEEDS
    notdegenerate = _not_degenerate(agent_id, rows, low_n=low_n)
    learn = _learn(
        agent_id,
        paired_fold_diffs(results, agent_id=agent_id, baseline_id=random_id, cost_bps=cost_bps),
        n_folds=n_folds,
        n_seeds=n_seeds,
        vetoed=notdegenerate.passed is False,
        permutation_draws=permutation_draws,
        permutation_seed=permutation_seed,
    )
    overfit = _overfit(agent_id, rows, low_n=low_n)
    vs_bh = _vs_buy_and_hold(
        agent_id,
        paired_fold_diffs(
            results, agent_id=agent_id, baseline_id=buy_and_hold_id, cost_bps=cost_bps
        ),
        n_folds=n_folds,
        low_n=low_n,
    )
    return notdegenerate, learn, overfit, vs_bh


# --- the report -----------------------------------------------------------------------------------


def _summaries(
    results: Sequence[PolicyFoldResult], cost_grid: Sequence[float]
) -> tuple[PolicySummary, ...]:
    order: list[str] = []
    for row in results:
        if row.policy_id not in order:
            order.append(row.policy_id)
    out: list[PolicySummary] = []
    for policy_id in order:
        for cost in cost_grid:
            rows = [r for r in results if r.policy_id == policy_id and r.cost_bps == cost]
            if not rows:
                continue
            n_folds, n_seeds = _counts(rows)
            sharpe = _by_fold(rows, "sharpe")
            ci = two_stage_bootstrap_iqm_ci(sharpe)
            wins = [r.win_rate for r in rows if r.win_rate is not None]
            trains = [r.train_sharpe for r in rows if r.train_sharpe is not None]
            out.append(
                PolicySummary(
                    policy_id=policy_id,
                    cost_bps=cost,
                    n_folds=n_folds,
                    n_seeds=n_seeds,
                    n_results=len(rows),
                    annualized_return_pct_iqm=interquartile_mean(
                        [r.annualized_return_pct for r in rows]
                    ),
                    sharpe_iqm=interquartile_mean([r.sharpe for r in rows]),
                    sortino_iqm=interquartile_mean([r.sortino for r in rows]),
                    max_drawdown_pct_iqm=interquartile_mean([r.max_drawdown_pct for r in rows]),
                    annualized_turnover_iqm=interquartile_mean(
                        [r.annualized_turnover for r in rows]
                    ),
                    win_rate_iqm=interquartile_mean(wins) if wins else None,
                    exposure_fraction_iqm=interquartile_mean([r.exposure_fraction for r in rows]),
                    n_trades_iqm=interquartile_mean([float(r.n_trades) for r in rows]),
                    action_entropy_iqm=interquartile_mean([r.action_entropy for r in rows]),
                    longest_holding_streak_iqm=interquartile_mean(
                        [float(r.longest_holding_streak) for r in rows]
                    ),
                    sharpe_mean=_mean([r.sharpe for r in rows]),
                    sharpe_clustered_se=clustered_standard_error(sharpe),
                    sharpe_iqm_ci_low=None if ci is None else ci[0],
                    sharpe_iqm_ci_high=None if ci is None else ci[1],
                    sharpe_iqm_ci_note="" if ci is not None else CI_NOT_APPLICABLE,
                    train_sharpe_iqm=interquartile_mean(trains) if trains else None,
                )
            )
    return tuple(out)


def build_rl_report(request: RlEvaluationRequest, *, as_of: datetime) -> RlEvaluationReport:
    """Validate the request (leakage per fold, >= 5 agent seeds, roles present), compute every
    gate per agent, the per-policy summaries, the pooled break-even costs and the warnings.

    Raises :class:`LeakageError` on a leaky fold and :class:`RlEvaluationError` on fewer than
    five agent seeds on any fold ("a single-seed number is noise") or a missing role.
    """
    if not request.folds:
        raise RlEvaluationError("no walk-forward folds: the dataset is too short for one fold")
    for fold in request.folds:
        fold.assert_no_leakage(obs_window=request.obs_window)
    if not request.agent_ids:
        raise RlEvaluationError("no agent policy to evaluate")
    if request.gate_cost_bps not in request.cost_bps_grid:
        raise RlEvaluationError(
            f"the gate cost {request.gate_cost_bps} bps is not in the grid {request.cost_bps_grid}"
        )
    fold_ids = {fold.index for fold in request.folds}
    seed_counts: list[int] = []
    for agent_id in request.agent_ids:
        rows = [
            r
            for r in request.results
            if r.policy_id == agent_id and r.cost_bps == request.gate_cost_bps
        ]
        n_folds, n_seeds = _counts(rows)
        if n_seeds < _MIN_SEEDS:
            raise RlEvaluationError(
                f"error: {agent_id!r} has {n_seeds} seed(s) on some fold; at least {_MIN_SEEDS} "
                "training seeds are required — a single-seed (or few-seed) number is noise"
            )
        missing = fold_ids - {r.fold_index for r in rows}
        if missing:
            raise RlEvaluationError(f"{agent_id!r} has no results on fold(s) {sorted(missing)}")
        if agent_id not in request.matched_random_ids:
            raise RlEvaluationError(f"no matched random baseline for {agent_id!r}")
        seed_counts.append(n_seeds)
    n_folds = len(request.folds)
    n_seeds = min(seed_counts)
    low_n = n_folds < _MIN_FOLDS or n_seeds < _MIN_SEEDS

    gates: list[GateVerdict] = []
    for agent_id in request.agent_ids:
        gates.extend(
            evaluate_gates(
                request.results,
                agent_id=agent_id,
                random_id=request.matched_random_ids[agent_id],
                buy_and_hold_id=request.buy_and_hold_id,
                cost_bps=request.gate_cost_bps,
            )
        )

    policy_ids: list[str] = []
    for row in request.results:
        if row.policy_id not in policy_ids:
            policy_ids.append(row.policy_id)
    break_even = {
        policy_id: pooled_break_even_cost_bps(
            request.results, policy_id=policy_id, source_cost_bps=request.gate_cost_bps
        )
        for policy_id in policy_ids
        if any(
            r.policy_id == policy_id and r.cost_bps == request.gate_cost_bps
            for r in request.results
        )
    }
    edges = {
        policy_id: {
            f"{cost:g}": mean_net_log_growth(
                request.results,
                policy_id=policy_id,
                at_cost_bps=cost,
                source_cost_bps=request.gate_cost_bps,
            )
            for cost in request.cost_bps_grid
        }
        for policy_id in break_even
    }

    warnings: list[str] = []
    if low_n:
        warnings.append(
            f"LOW_N: K={n_folds} folds (need >= {_MIN_FOLDS}) x {n_seeds} seeds (need >= "
            f"{_MIN_SEEDS}) — no gate may pass; the numbers below are descriptive only"
        )
    clipped = sorted(
        {
            (r.policy_id, r.fold_index)
            for r in request.results
            if r.risk_clipped_fraction is not None and r.risk_clipped_fraction > _MAX_RISK_CLIPPED
        }
    )
    if clipped:
        warnings.append(
            f"WARNING: risk_clipped_fraction > {_MAX_RISK_CLIPPED:.0%} on {len(clipped)} "
            "(policy, fold) pair(s) — the comparison is INVALID: you are measuring the risk "
            "gate, not the agent"
        )
    if request.n_configurations_evaluated > _DEFLATED_SHARPE_LOOKS:
        warnings.append(
            f"CAVEAT: {request.n_configurations_evaluated} configurations evaluated against this "
            "test span — past 5 looks the best result is inflated by selection; read every "
            "Sharpe through the Deflated Sharpe Ratio (Bailey & Lopez de Prado, SSRN 2460551)"
        )
    if request.n_configurations_evaluated > _STOP_TUNING_LOOKS:
        warnings.append(
            "STOP: more than 10 looks at this test span — freeze the configuration and move to "
            "the look-once holdout (program doc §9 stop conditions)"
        )

    return RlEvaluationReport(
        as_of=as_of,
        symbol=request.symbol,
        model_ids=tuple(request.model_ids),
        artifact_sha256s=dict(request.artifact_sha256s),
        n_folds=n_folds,
        n_seeds=n_seeds,
        low_n=low_n,
        cost_bps_grid=tuple(request.cost_bps_grid),
        break_even_cost_bps=break_even,
        adjustment=request.adjustment,
        results=tuple(request.results),
        gates=tuple(gates),
        n_configurations_evaluated=request.n_configurations_evaluated,
        parity_ties=request.parity_ties,
        parity_observations=request.parity_observations,
        dataset_sha256=request.dataset_sha256,
        evaluation_mode=request.evaluation_mode,
        gate_cost_bps=request.gate_cost_bps,
        folds=tuple(request.folds),
        summaries=_summaries(request.results, request.cost_bps_grid),
        warnings=tuple(warnings),
        notes=tuple(request.notes),
        n_configurations_source=request.n_configurations_source,
        gate_rules_version=GATE_RULES_VERSION,
        edge_over_flat_log_growth=edges,
    )
