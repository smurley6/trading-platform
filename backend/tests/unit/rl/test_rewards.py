"""Reward functions (task-rl-03; program doc §4.5). Stdlib only — no optional extras.

The reward is ``100 * ln`` of NET-OF-COST equity growth. The scale-invariance property test pins
the §4.2 BINDING rule: closes enter only as ratios, so multiplying the whole price series by a
constant changes nothing in growth, cost, reward or the differential Sharpe — and a negative control
proves the check can actually fail.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pytest

from trading.application.rl_dataset import rl_dataset_from_json
from trading.rl.bars import BarTable
from trading.rl.costs import BPS, COST_BPS_GRID, DEFAULT_COST, ZERO_COST, CostModel
from trading.rl.rewards import (
    GROWTH_FLOOR,
    REWARD_SCALE,
    DifferentialSharpe,
    RewardFn,
    log_return_reward,
    net_of_cost_growth,
)

_FROZEN = Path(__file__).parents[3] / "data" / "rl" / "spy_daily.json"

GrowthFn = Callable[..., float]


@dataclass(frozen=True)
class _Trajectory:
    growths: tuple[float, ...]
    costs: tuple[float, ...]
    rewards: tuple[float, ...]
    dsr: tuple[float, ...]
    final_equity: float


def _run(
    closes: Sequence[float],
    positions: Sequence[float],
    cost: CostModel,
    *,
    growth_fn: GrowthFn = net_of_cost_growth,
) -> _Trajectory:
    """TEST-ONLY bookkeeping: ``positions[t]`` is held over the period ``(closes[t],
    closes[t+1])``, changed from ``positions[t-1]`` at the period's start (flat before the start).

    This pairs each period with an index purely to exercise the arithmetic; it is NOT the env's
    timing contract (task-rl-05 decides the PIT-safe bar pairing / execution lag)."""
    equity, held = 1.0, 0.0
    sharpe = DifferentialSharpe()
    growths: list[float] = []
    costs: list[float] = []
    rewards: list[float] = []
    dsr: list[float] = []
    for t in range(len(closes) - 1):
        target = positions[t]
        growth = growth_fn(
            close_prev=closes[t],
            close_now=closes[t + 1],
            position_before=held,
            position_after=target,
            cost=cost,
        )
        new_equity = equity * growth
        growths.append(growth)
        costs.append(cost.cost_fraction(position_before=held, position_after=target))
        rewards.append(log_return_reward(equity_prev=equity, equity_now=new_equity))
        dsr.append(sharpe.update(growth - 1.0))
        equity, held = new_equity, target
    return _Trajectory(tuple(growths), tuple(costs), tuple(rewards), tuple(dsr), equity)


def _random_walk(rng: random.Random, n: int, start: float) -> list[float]:
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0003, 0.012)))
    return closes


def _close(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(a, b, strict=True)
    )


def _assert_scale_invariant(
    closes: Sequence[float],
    positions: Sequence[float],
    cost: CostModel,
    factor: float,
    *,
    growth_fn: GrowthFn = net_of_cost_growth,
) -> None:
    base = _run(closes, positions, cost, growth_fn=growth_fn)
    scaled = _run([c * factor for c in closes], positions, cost, growth_fn=growth_fn)
    assert _close(base.growths, scaled.growths), "growth depends on the price level"
    assert _close(base.costs, scaled.costs), "cost depends on the price level"
    assert _close(base.rewards, scaled.rewards), "reward depends on the price level"
    assert _close(base.dsr, scaled.dsr), "differential Sharpe depends on the price level"
    assert math.isclose(base.final_equity, scaled.final_equity, rel_tol=1e-12)


# --- the scale-invariance property (BINDING returns-only rule) --------------------------------


@pytest.mark.parametrize("factor", [10.0, 1e-3, 7.3])
@pytest.mark.parametrize("one_way_bps", [*COST_BPS_GRID, DEFAULT_COST.one_way_bps])
def test_scaling_the_whole_price_series_changes_nothing(factor: float, one_way_bps: float) -> None:
    cost = CostModel(one_way_bps=one_way_bps)
    rng = random.Random(20261021)
    for _ in range(40):
        closes = _random_walk(rng, n=120, start=rng.uniform(5.0, 900.0))
        positions = [float(rng.random() < 0.5) for _ in closes]
        _assert_scale_invariant(closes, positions, cost, factor)


def test_scaling_the_frozen_spy_series_by_ten_changes_nothing() -> None:
    closes = BarTable.from_dataset(rl_dataset_from_json(_FROZEN.read_bytes())).closes()
    # A deterministic, trade-heavy action path: long for 5 sessions, flat for 3, repeating.
    positions = [1.0 if t % 8 < 5 else 0.0 for t in range(len(closes))]
    _assert_scale_invariant(closes, positions, DEFAULT_COST, 10.0)


def test_scale_invariance_check_detects_a_price_level_cost() -> None:
    # Negative control — a check that cannot fail is decoration. A per-share fee ($0.01/share on
    # an all-in position of equity / close shares) needs the price LEVEL; the check must catch it.
    def per_share_fee_growth(
        *,
        close_prev: float,
        close_now: float,
        position_before: float,
        position_after: float,
        cost: CostModel,
    ) -> float:
        traded = abs(position_after - position_before)
        fee = traded * 0.01 / close_prev
        return 1.0 + position_after * (close_now / close_prev - 1.0) - fee

    rng = random.Random(7)
    closes = _random_walk(rng, n=60, start=100.0)
    positions = [float(t % 4 < 2) for t in range(len(closes))]
    with pytest.raises(AssertionError, match="price level"):
        _assert_scale_invariant(closes, positions, ZERO_COST, 10.0, growth_fn=per_share_fee_growth)


# --- log_return_reward ------------------------------------------------------------------------


def test_log_return_reward_is_additive() -> None:
    rng = random.Random(1998)
    equities = [1.0]
    for _ in range(500):
        equities.append(equities[-1] * rng.uniform(0.95, 1.05))

    rewards = [
        log_return_reward(equity_prev=prev, equity_now=now) for prev, now in pairwise(equities)
    ]

    assert abs(sum(rewards) / 100 - math.log(equities[-1] / equities[0])) <= 1e-12


def test_log_return_reward_scale_and_floor() -> None:
    assert log_return_reward(equity_prev=2.0, equity_now=2.0) == 0.0
    assert log_return_reward(equity_prev=1.0, equity_now=math.e, scale=1.0) == pytest.approx(1.0)
    floored = REWARD_SCALE * math.log(GROWTH_FLOOR)
    assert log_return_reward(equity_prev=1.0, equity_now=0.0) == floored
    assert log_return_reward(equity_prev=1.0, equity_now=-0.5) == floored


@pytest.mark.parametrize(
    ("prev", "now"), [(0.0, 1.0), (-1.0, 1.0), (math.nan, 1.0), (1.0, math.inf), (1.0, math.nan)]
)
def test_log_return_reward_rejects_invalid_equity(prev: float, now: float) -> None:
    with pytest.raises(ValueError, match="equity"):
        log_return_reward(equity_prev=prev, equity_now=now)


def test_log_return_reward_satisfies_the_reward_protocol() -> None:
    reward_fn: RewardFn = log_return_reward
    assert reward_fn(equity_prev=1.0, equity_now=1.01) == pytest.approx(100 * math.log(1.01))


# --- net_of_cost_growth: when and on what the cost is charged ---------------------------------


def test_net_of_cost_growth_charges_each_position_change_once_on_traded_notional() -> None:
    c = DEFAULT_COST.one_way_bps * BPS

    def up_two_percent(before: float, after: float) -> float:
        return net_of_cost_growth(
            close_prev=100.0,
            close_now=102.0,
            position_before=before,
            position_after=after,
            cost=DEFAULT_COST,
        )

    # Entering: the new (long) position earns the period's return and pays one one-way cost.
    assert up_two_percent(0.0, 1.0) == pytest.approx(1.02 - c, abs=1e-15)
    # Holding long: earns the return, pays nothing.
    assert up_two_percent(1.0, 1.0) == pytest.approx(1.02, abs=1e-15)
    # Exiting: the (now flat) position earns nothing and pays one one-way cost.
    assert up_two_percent(1.0, 0.0) == pytest.approx(1.0 - c, abs=1e-15)
    # Flat and staying flat: exactly 1.
    assert up_two_percent(0.0, 0.0) == 1.0


def test_a_round_trip_costs_two_one_way_charges_and_churn_strictly_loses() -> None:
    closes = [100.0] * 11
    churn = [float(t % 2 == 0) for t in range(len(closes))]  # buy, sell, buy, sell, ...

    result = _run(closes, churn, DEFAULT_COST)

    c = DEFAULT_COST.one_way_bps * BPS
    assert result.final_equity == pytest.approx((1.0 - c) ** 10, rel=1e-12)
    assert sum(result.rewards) < 0.0
    assert _run(closes, churn, ZERO_COST).final_equity == 1.0


def test_always_long_reproduces_buy_and_hold_at_zero_cost() -> None:
    closes = _random_walk(random.Random(42), n=253, start=431.7)
    always_long = [1.0] * len(closes)

    result = _run(closes, always_long, ZERO_COST)

    assert math.exp(sum(result.rewards) / 100) == pytest.approx(closes[-1] / closes[0], rel=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_net_of_cost_growth_rejects_a_non_positive_close(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        net_of_cost_growth(
            close_prev=bad, close_now=1.0, position_before=0, position_after=1, cost=ZERO_COST
        )
    with pytest.raises(ValueError, match="finite and positive"):
        net_of_cost_growth(
            close_prev=1.0, close_now=bad, position_before=0, position_after=1, cost=ZERO_COST
        )


# --- DifferentialSharpe (Moody & Saffell 1998, eqs. 4-5) --------------------------------------


def test_differential_sharpe_returns_zero_at_the_variance_singularity() -> None:
    fresh = DifferentialSharpe()
    assert fresh.update(0.01) == 0.0  # A = B = 0 -> B - A^2 = 0

    never_trades = DifferentialSharpe()
    assert [never_trades.update(0.0) for _ in range(1_000)] == [0.0] * 1_000

    exact = DifferentialSharpe(a=0.1, b=0.01)  # B == A^2 exactly
    assert exact.update(0.5) == 0.0
    near = DifferentialSharpe(a=0.1, b=0.01 + 1e-13)  # within the 1e-12 guard
    assert near.update(-0.5) == 0.0


def test_differential_sharpe_matches_the_hand_computed_update() -> None:
    dsr = DifferentialSharpe(eta=0.5, a=0.01, b=0.0002)
    # var = 2e-4 - 1e-4 = 1e-4; dA = 0.01; dB = 4e-4 - 2e-4 = 2e-4
    # D = (2e-4 * 0.01 - 0.5 * 0.01 * 2e-4) / (1e-4) ** 1.5 = (2e-6 - 1e-6) / 1e-6 = 1.0
    assert dsr.update(0.02) == pytest.approx(1.0, rel=1e-12)
    assert dsr.observation() == pytest.approx((0.015, 0.0003), rel=1e-12)


def test_differential_sharpe_observation_and_reset() -> None:
    dsr = DifferentialSharpe(eta=0.04)
    for r in (0.01, -0.02, 0.015):
        assert math.isfinite(dsr.update(r))
    a, b = dsr.observation()
    assert a != 0.0
    assert b > 0.0

    dsr.reset()

    assert dsr.observation() == (0.0, 0.0)
    assert dsr.eta == 0.04


@pytest.mark.parametrize("eta", [0.0, -0.1, 1.5, math.nan])
def test_differential_sharpe_rejects_an_invalid_eta(eta: float) -> None:
    with pytest.raises(ValueError, match="eta"):
        DifferentialSharpe(eta=eta)


def test_differential_sharpe_rejects_a_non_finite_return() -> None:
    with pytest.raises(ValueError, match="finite"):
        DifferentialSharpe().update(math.nan)
