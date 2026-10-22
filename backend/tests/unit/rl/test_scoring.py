"""Scoring policies through the env (task-rl-07; program doc §5.1 Tier B, §5.2, §5.4).

gymnasium + numpy are in ``[dev]``: default ``pytest``, no ``rl`` marker. Every baseline is replayed
through the SAME env path the agent will be; these tests pin what that replay records and that the
metrics computed from it mean what ``rl/metrics.py`` says they mean.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from tests.unit.rl.conftest import bar_table, frozen_spy_table, synthetic_random_walk, with_closes
from trading.domain.rl_features import WARMUP_BARS, build_observation
from trading.rl.bars import BarTable
from trading.rl.baselines import (
    BASELINES,
    BUY,
    HOLD,
    SELL,
    Policy,
    PolicyFactory,
    always_flat,
    buy_and_hold,
    random_policy,
    sma_cross,
)
from trading.rl.costs import CostModel
from trading.rl.env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    DEFAULT_FEATURE_SPEC,
    EnvConfigError,
)
from trading.rl.metrics import (
    NULL_BAND_SEEDS,
    ScoringError,
    Segment,
    break_even_cost_bps,
    buy_and_hold_reference,
    exposure_fraction,
    frictionless_growth,
    matched_trade_probability,
    summarize,
)
from trading.rl.scoring import random_null_band, rollout, rollout_seeds

_WALK = synthetic_random_walk(700, seed=21)
_TABLE = bar_table(_WALK)


def _segment(start: int = 100, length: int = 252, *, lag: int = 0, bps: float = 0.0) -> Segment:
    return Segment(start, length, execution_lag=lag, cost=CostModel(bps))


def test_action_constants_mirror_the_env() -> None:
    assert (HOLD, BUY, SELL) == (ACTION_HOLD, ACTION_BUY, ACTION_SELL)


# --- spec acceptance: buy-and-hold, flat, entropy --------------------------------------------


@pytest.mark.parametrize("lag", [0, 1, 3])
@pytest.mark.parametrize("bps", [0.0, 2.0, 5.0])
def test_always_long_rollout_matches_buy_and_hold(lag: int, bps: float) -> None:
    segment = _segment(lag=lag, bps=bps)
    result = rollout(buy_and_hold, _TABLE, segment, seed=0)
    fill = segment.start + 1 + lag
    # Filled at C_{start+1+lag} (never C_start), charged once, then held to C_end.
    expected = (1.0 - bps * 1e-4) * _WALK[segment.end] / _WALK[fill]
    assert result.final_equity == pytest.approx(expected, rel=1e-12)
    reference = buy_and_hold_reference(_TABLE, segment)
    assert result.final_equity == pytest.approx(
        (1.0 - bps * 1e-4) * reference[-1] / reference[1 + lag], rel=1e-12
    )
    assert result.n_trades == 1 and result.turnover == 1.0
    assert result.fills[0].fill_index == fill
    assert result.cost_paid == pytest.approx(bps * 1e-4)  # charged on the 1.0 equity at the fill
    # Exposure is (n - 1 - lag) / n: the fill lag is honest exposure lost, not 1.0.
    assert exposure_fraction(result) == pytest.approx((segment.length - 1 - lag) / segment.length)


def test_always_flat_scores_zero_return_and_zero_turnover() -> None:
    for bps in (0.0, 5.0):
        result = rollout(always_flat, _TABLE, _segment(bps=bps), seed=0)
        score = summarize(result)
        assert set(result.equity) == {1.0}
        assert score.total_return_pct == 0.0 and score.sharpe == 0.0
        assert score.max_drawdown_pct == 0.0
        assert score.turnover == 0.0 and score.n_trades == 0 and score.cost_paid == 0.0
        assert score.exposure_fraction == 0.0
        assert score.break_even_cost_bps == 0.0


def test_action_entropy_is_zero_for_a_constant_policy_and_positive_for_random() -> None:
    assert summarize(rollout(always_flat, _TABLE, _segment(), seed=0)).action_entropy == 0.0
    random_entropy = summarize(rollout(random_policy(), _TABLE, _segment(), seed=0)).action_entropy
    assert 0.5 < random_entropy <= np.log(3)


# --- the replay record -----------------------------------------------------------------------


def test_rollout_records_the_env_accounting() -> None:
    segment = _segment(start=80, length=120, lag=1, bps=2.0)
    result = rollout(random_policy(0.2), _TABLE, segment, seed=5)
    assert result.cursors == tuple(range(80, 201))
    assert result.marked_at == tuple(_TABLE.bars[i].known_at for i in result.cursors)
    assert result.equity[0] == 1.0 and result.positions[0] == 0.0
    assert result.steps == 120 and not result.ruined
    assert len(result.fills) == result.n_trades > 0
    for fill in result.fills:
        assert fill.fill_index == fill.decision_index + 2  # t + 1 + lag
        assert result.positions[fill.fill_index - 80] == fill.position_after
        assert fill.cost_fraction == pytest.approx(2e-4)
    assert result.turnover == sum(f.traded for f in result.fills)
    assert result.requested_actions == result.actions  # a baseline never no-ops
    assert result.n_noops == 0


@pytest.mark.parametrize("name", sorted(BASELINES))
@pytest.mark.parametrize("lag", [0, 2])
def test_baselines_never_emit_an_illegal_action(name: str, lag: int) -> None:
    for seed in range(3):
        result = rollout(BASELINES[name], _TABLE, _segment(lag=lag), seed=seed)
        assert result.n_noops == 0
        assert result.requested_actions == result.actions


def test_rollout_is_deterministic_given_policy_seed_and_segment() -> None:
    segment = _segment(bps=1.0, lag=1)
    for factory in BASELINES.values():
        assert rollout(factory, _TABLE, segment, seed=4) == rollout(
            factory, _TABLE, segment, seed=4
        )
    random_a = rollout(random_policy(), _TABLE, segment, seed=4)
    random_b = rollout(random_policy(), _TABLE, segment, seed=5)
    assert random_a.actions != random_b.actions


def test_rollout_seeds_takes_a_seed_list() -> None:
    results = rollout_seeds(random_policy(), _TABLE, _segment(), [3, 1, 2])
    assert [r.seed for r in results] == [3, 1, 2]
    assert results[1] == rollout(random_policy(), _TABLE, _segment(), seed=1)
    with pytest.raises(ScoringError):
        rollout_seeds(random_policy(), _TABLE, _segment(), [])
    with pytest.raises(ScoringError):
        rollout_seeds(random_policy(), _TABLE, _segment(), [1, 1])
    with pytest.raises(ScoringError):
        rollout(random_policy(), _TABLE, _segment(), seed=True)


def test_a_segment_that_does_not_fit_the_bars_fails_closed() -> None:
    with pytest.raises(EnvConfigError):
        rollout(always_flat, _TABLE, Segment(len(_WALK) - 10, 20), seed=0)


def test_an_out_of_range_action_is_rejected() -> None:
    def bad(seed: int) -> Policy:
        return lambda observation, history: 3

    with pytest.raises(ValueError, match="action"):
        rollout(bad, _TABLE, _segment(), seed=0)


def test_ruin_stops_the_replay_and_voids_the_break_even() -> None:
    crash = [100.0] * 60 + [100.0 * 0.5**k for k in range(1, 8)] + [1.0] * 40
    table = bar_table(crash)
    result = rollout(buy_and_hold, table, Segment(55, 50), seed=0)
    assert result.ruined and result.steps < 50
    assert result.final_equity <= 0.2
    assert break_even_cost_bps(result) is None
    band = random_null_band(table, Segment(55, 50), seeds=range(5), trade_probability=0.0)
    assert band.final_equity == (1.0,) * 5  # trade_probability 0 never buys the crash


# --- point-in-time equivalence of the policy inputs ------------------------------------------


def _recording(log: list[tuple[tuple[float, ...], tuple[float, ...]]]) -> PolicyFactory:
    def factory(seed: int) -> Policy:
        inner = random_policy(0.1)(seed)

        def act(observation: Sequence[float], history: Sequence[float]) -> int:
            log.append((tuple(observation), tuple(history)))
            return inner(observation, history)

        return act

    return factory


@pytest.mark.parametrize("lag", [0, 2])
def test_history_is_exactly_the_window_the_observation_was_built_from(lag: int) -> None:
    log: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    segment = _segment(start=60, length=300, lag=lag)
    result = rollout(_recording(log), _TABLE, segment, seed=8)
    assert len(log) == segment.length
    for step, (observation, history) in enumerate(log):
        cursor = result.cursors[step]
        assert history == tuple(_WALK[cursor + 1 - WARMUP_BARS : cursor + 1])
        rebuilt = build_observation(
            history, held=False, entry_close=None, spec=DEFAULT_FEATURE_SPEC
        )
        # The 7 price features (the position features come from the env's state, not closes).
        assert observation[:7] == tuple(float(np.float32(v)) for v in rebuilt[:7])


@pytest.mark.parametrize("k", [150, 260, 420])
@pytest.mark.parametrize("factor", [3.0, 0.25])
def test_sma_cross_actions_are_invariant_to_every_bar_after_the_decision(
    k: int, factor: float
) -> None:
    poisoned = with_closes(_TABLE, [c * factor if i > k else c for i, c in enumerate(_WALK)])
    segment = _segment(start=100, length=500)
    clean = rollout(sma_cross(), _TABLE, segment, seed=0)
    dirty = rollout(sma_cross(), poisoned, segment, seed=0)
    decided_by_k = k - segment.start + 1  # decisions at cursors start .. k
    assert clean.actions[:decided_by_k] == dirty.actions[:decided_by_k]
    assert clean.cursors[:decided_by_k][-1] == k


def test_the_sma_poisoning_probe_is_live() -> None:
    # A steady decline keeps SMA_20 < SMA_50 (flat); x3 after k flips it long at k + 1, so the
    # invariance test above would see a future read.
    decline = [100.0 * 0.999**i for i in range(400)]
    k = 200
    table, poisoned = (
        bar_table(decline),
        bar_table([c * 3.0 if i > k else c for i, c in enumerate(decline)]),
    )
    segment = Segment(100, 250)
    clean = rollout(sma_cross(), table, segment, seed=0)
    dirty = rollout(sma_cross(), poisoned, segment, seed=0)
    assert set(clean.actions) == {HOLD}
    assert dirty.actions[k + 1 - segment.start] == BUY
    assert dirty.actions[: k + 1 - segment.start] == clean.actions[: k + 1 - segment.start]


# --- break-even cost and cost invariance -----------------------------------------------------


def test_the_position_path_does_not_depend_on_the_cost() -> None:
    for factory in BASELINES.values():
        paths = {
            rollout(factory, _TABLE, _segment(bps=bps, lag=1), seed=2).positions
            for bps in (0.0, 1.0, 5.0, 50.0)
        }
        assert len(paths) == 1


def test_frictionless_growth_equals_a_zero_cost_replay() -> None:
    for factory in (buy_and_hold, sma_cross(), random_policy()):
        costly = rollout(factory, _TABLE, _segment(bps=5.0), seed=6)
        free = rollout(factory, _TABLE, _segment(bps=0.0), seed=6)
        assert frictionless_growth(costly) == pytest.approx(free.final_equity, rel=1e-12)


def _profitable_seed(factory: PolicyFactory, table: BarTable, segment: Segment) -> int:
    for seed in range(50):
        result = rollout(factory, table, segment, seed=seed)
        if frictionless_growth(result) > 1.0 and result.n_trades >= 4:
            return seed
    raise AssertionError("no profitable seed")  # pragma: no cover


def test_replaying_at_the_break_even_cost_ends_at_exactly_one() -> None:
    segment = _segment(start=100, length=400, bps=1.0)
    factory = random_policy(0.05)
    seed = _profitable_seed(factory, _TABLE, segment)
    result = rollout(factory, _TABLE, segment, seed=seed)
    bps = break_even_cost_bps(result)
    assert bps is not None and 0.0 < bps < 1000.0
    at_break_even = rollout(factory, _TABLE, Segment(100, 400, cost=CostModel(bps)), seed=seed)
    assert at_break_even.positions == result.positions
    assert at_break_even.final_equity == pytest.approx(1.0, abs=1e-9)
    # Just below it the edge survives; just above it, it is gone.
    below = rollout(factory, _TABLE, Segment(100, 400, cost=CostModel(bps * 0.99)), seed=seed)
    above = rollout(factory, _TABLE, Segment(100, 400, cost=CostModel(bps * 1.01)), seed=seed)
    assert below.final_equity > 1.0 > above.final_equity


# --- the random null band --------------------------------------------------------------------


def test_random_null_band_defaults_to_thirty_seeds() -> None:
    segment = _segment(bps=2.0)
    band = random_null_band(_TABLE, segment)
    assert band.seeds == tuple(range(NULL_BAND_SEEDS)) and NULL_BAND_SEEDS == 30
    assert band.cursors == tuple(range(segment.start, segment.end + 1))
    assert len(band.p50) == len(band.cursors) == segment.length + 1
    assert all(lo <= mid <= hi for lo, mid, hi in zip(band.p25, band.p50, band.p75, strict=True))
    finals = [r.final_equity for r in rollout_seeds(random_policy(), _TABLE, segment, range(30))]
    assert band.final_equity == tuple(finals)
    assert min(finals) <= band.final_equity_iqm <= max(finals)
    assert band == random_null_band(_TABLE, segment)  # deterministic


def test_trade_probability_turnover_matches_a_policy() -> None:
    segment = _segment(start=60, length=600)
    target = rollout_seeds(sma_cross(), _TABLE, segment, [0])
    p = matched_trade_probability(target)
    band = random_null_band(_TABLE, segment, trade_probability=p)
    matched = sum(band.turnover) / len(band.turnover)
    # Same expected number of accepted orders -> mean turnover within sampling noise.
    assert matched == pytest.approx(target[0].turnover, rel=0.35)
    unmatched = random_null_band(_TABLE, segment)
    assert sum(unmatched.turnover) / len(unmatched.turnover) > 5 * matched


# --- the real frozen SPY data ----------------------------------------------------------------


def test_baselines_on_the_frozen_spy_series() -> None:
    table = frozen_spy_table()
    closes = table.closes()
    segment = Segment(len(table) - 1 - 252, 252, cost=CostModel(2.0))
    reference = buy_and_hold_reference(table, segment)
    bh = rollout(buy_and_hold, table, segment, seed=0)
    assert bh.final_equity == pytest.approx(
        (1 - 2e-4) * closes[segment.end] / closes[segment.start + 1], rel=1e-12
    )
    assert reference[-1] == pytest.approx(closes[segment.end] / closes[segment.start], rel=1e-15)
    assert rollout(always_flat, table, segment, seed=0).final_equity == 1.0
    sma = summarize(rollout(sma_cross(), table, segment, seed=0))
    assert sma.n_trades >= 1 and 0.0 < sma.exposure_fraction < 1.0
    band = random_null_band(table, segment)
    assert band.p05[-1] < band.p95[-1]
