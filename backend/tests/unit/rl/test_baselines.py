"""The four baseline policies (task-rl-07; program doc §5.2) — pure stdlib, no gymnasium.

Env-level behavior (actions accepted, fills, PIT-equivalence of the SMA history) is pinned in
``test_scoring.py``; this module pins each policy's decision rule in isolation.
"""

import random

import pytest

from trading.rl.baselines import (
    BASELINES,
    BUY,
    HOLD,
    SELL,
    UNIFORM_TRADE_PROBABILITY,
    Policy,
    always_flat,
    buy_and_hold,
    random_policy,
    sma_cross,
)

_OBS = (0.0,) * 9
_HISTORY = tuple(100.0 for _ in range(50))


def _drive(policy: Policy, steps: int, history: tuple[float, ...] = _HISTORY) -> list[int]:
    return [policy(_OBS, history) for _ in range(steps)]


def _assert_legal(actions: list[int]) -> None:
    """Starting flat, BUY/SELL must strictly alternate (BUY first): never an illegal no-op."""
    long = False
    for action in actions:
        if action == BUY:
            assert not long
            long = True
        elif action == SELL:
            assert long
            long = False
        else:
            assert action == HOLD


def test_always_flat_never_trades() -> None:
    assert _drive(always_flat(0), 100) == [HOLD] * 100


def test_buy_and_hold_buys_once_then_holds() -> None:
    assert _drive(buy_and_hold(7), 50) == [BUY] + [HOLD] * 49


def test_each_factory_call_is_a_fresh_episode() -> None:
    factory = buy_and_hold
    first = factory(0)
    _drive(first, 5)
    assert factory(0)(_OBS, _HISTORY) == BUY  # a new episode starts flat again


def test_random_policy_is_deterministic_for_a_seed() -> None:
    factory = random_policy()
    assert _drive(factory(11), 500) == _drive(factory(11), 500)
    assert _drive(factory(11), 500) != _drive(factory(12), 500)


def test_random_policy_uses_its_own_rng_not_the_global_one() -> None:
    random.seed(1)
    a = _drive(random_policy()(5), 200)
    random.seed(2)
    b = _drive(random_policy()(5), 200)
    assert a == b


def test_random_policy_draws_exactly_once_per_decision() -> None:
    p = 0.3
    rng = random.Random(42)
    expected_trades = [rng.random() < p for _ in range(1000)]
    actions = _drive(random_policy(p)(42), 1000)
    assert [a != HOLD for a in actions] == expected_trades


@pytest.mark.parametrize("p", [0.0, 0.05, UNIFORM_TRADE_PROBABILITY, 0.9, 1.0])
def test_random_policy_only_emits_legal_flips_at_its_propensity(p: float) -> None:
    actions = _drive(random_policy(p)(3), 6000)
    _assert_legal(actions)
    rate = sum(a != HOLD for a in actions) / len(actions)
    assert abs(rate - p) < 0.02


def test_random_policy_extremes() -> None:
    assert _drive(random_policy(0.0)(1), 50) == [HOLD] * 50
    assert _drive(random_policy(1.0)(1), 6) == [BUY, SELL, BUY, SELL, BUY, SELL]
    assert _drive(random_policy(1)(1), 2) == [BUY, SELL]  # an int propensity is accepted


@pytest.mark.parametrize("bad", [-0.1, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_random_policy_rejects_a_bad_propensity(bad: object) -> None:
    with pytest.raises(ValueError, match="trade_probability"):
        random_policy(bad)  # type: ignore[arg-type]


def test_uniform_default_is_one_third() -> None:
    # Uniform {BUY, HOLD, SELL} under the env's legality rule flips the committed position with
    # probability 1/3: exactly one of the three actions is the legal flip, whatever the position.
    assert pytest.approx(1 / 3) == UNIFORM_TRADE_PROBABILITY


def test_sma_cross_goes_long_on_a_golden_cross_and_flat_on_a_death_cross() -> None:
    rising = tuple(100.0 + i for i in range(50))  # SMA_20 > SMA_50
    falling = tuple(200.0 - i for i in range(50))  # SMA_20 < SMA_50
    policy = sma_cross()(0)
    assert policy(_OBS, falling) == HOLD  # already flat
    assert policy(_OBS, rising) == BUY
    assert policy(_OBS, rising) == HOLD  # already long
    assert policy(_OBS, falling) == SELL
    assert policy(_OBS, falling) == HOLD


def test_sma_cross_is_strict_and_reads_only_the_trailing_windows() -> None:
    policy = sma_cross()(0)
    assert policy(_OBS, _HISTORY) == HOLD  # SMA_20 == SMA_50 is not a cross
    # Prepended closes (older than 50) are ignored entirely.
    older = (1e9,) * 30 + tuple(100.0 + i for i in range(50))
    assert sma_cross()(0)(_OBS, older) == BUY
    # Fast window: only the last 20 closes; one bump just outside it changes nothing.
    flat_then_bump = (100.0,) * 29 + (1000.0,) + (100.0,) * 20
    assert sma_cross()(0)(_OBS, flat_then_bump) == HOLD  # bump in SMA_50 only -> fast < slow


def test_sma_cross_is_scale_free() -> None:
    walk = [100.0]
    rng = random.Random(9)
    for _ in range(400):
        walk.append(walk[-1] * (1 + rng.gauss(0, 0.01)))
    for scale in (2.0, 1e-3, 7.3):
        plain, scaled = sma_cross()(0), sma_cross()(0)
        for t in range(50, len(walk)):
            window = tuple(walk[t - 49 : t + 1])
            assert plain(_OBS, window) == scaled(_OBS, tuple(c * scale for c in window))


def test_sma_cross_emits_only_legal_flips_on_a_random_walk() -> None:
    walk = [100.0]
    rng = random.Random(4)
    for _ in range(2000):
        walk.append(walk[-1] * (1 + rng.gauss(0, 0.01)))
    policy = sma_cross()(0)
    actions = [policy(_OBS, tuple(walk[t - 49 : t + 1])) for t in range(49, len(walk))]
    _assert_legal(actions)
    assert sum(a != HOLD for a in actions) > 4


def test_sma_cross_parameters_are_validated() -> None:
    assert sma_cross(10, 50)(0)(_OBS, tuple(100.0 + i for i in range(50))) == BUY
    for fast, slow in [(20, 20), (0, 50), (30, 20), (20, 51), (20.0, 50), (True, 50)]:
        with pytest.raises(ValueError):
            sma_cross(fast, slow)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="history"):
        sma_cross()(0)(_OBS, (100.0,) * 49)


def test_registry_holds_the_four_section_5_2_baselines() -> None:
    assert set(BASELINES) == {"buy_and_hold", "always_flat", "random", "sma_cross_20_50"}
    for factory in BASELINES.values():
        _assert_legal(_drive(factory(0), 20, tuple(100.0 + i for i in range(50))))
