"""The gymnasium env and its timing contract (task-rl-05; program doc §4.3-§4.8).

gymnasium + numpy are in ``[dev]``, so everything here runs in the default ``pytest``; only the
stable-baselines3 checker needs the ``[rl]`` extra (``@pytest.mark.rl`` + ``importorskip``).

The timing contract under test (``rl/env.py`` module docstring; program doc §4.7 RESOLVED): decide
at ``bars[t].known_at``; an accepted order fills at the close of ``fill_index_for(t, L) = t+1+L``
(never a close known at decision time); the step at cursor ``t`` books the move ``C_t -> C_{t+1}``
for the position ALREADY held, then any fill at ``C_{t+1}``; a position earns only from its fill
close onward, and ``entry_close`` / ``held`` follow that actual fill.
"""

import math
import random
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pytest
from gymnasium.utils.env_checker import check_env

from trading.application.rl_dataset import rl_dataset_from_json
from trading.domain.prices import PriceSeries
from trading.domain.rl_features import MAX_LOOKBACK, OBS_DIM, WARMUP_BARS, FeatureScales
from trading.rl.bars import BarTable
from trading.rl.costs import BPS, DEFAULT_COST, STRESS_COST, ZERO_COST, CostModel
from trading.rl.env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    EnvConfig,
    EnvConfigError,
    SingleTickerTradingEnv,
    fill_index_for,
)

_FROZEN = Path(__file__).parents[3] / "data" / "rl" / "spy_daily.json"
_NY = ZoneInfo("America/New_York")
_UNREALIZED_SCALE = FeatureScales().unrealized

Obs = npt.NDArray[np.float32]
Policy = Callable[[int, dict[str, Any]], int]


def _table(closes: Sequence[float], symbol: str = "SPY") -> BarTable:
    """Weekday sessions stamped 00:00 New York (Alpaca's convention), through the ONE knowledge-time
    rule (``from_price_series`` -> ``session_known_at``). ``Decimal(repr(c))`` round-trips the
    float exactly, so ``closes()`` returns these closes bit-for-bit."""
    day = date(2020, 1, 2)
    points: list[tuple[datetime, Decimal]] = []
    for close in closes:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        stamp = datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC)
        points.append((stamp, Decimal(repr(close))))
        day += timedelta(days=1)
    return BarTable.from_price_series(PriceSeries(series={symbol: tuple(points)}), symbol)


def _walk(n: int, seed: int, *, vol: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0003, vol)))
    return closes


def _env(closes: Sequence[float], *, audit: bool = False, **config: Any) -> SingleTickerTradingEnv:
    return SingleTickerTradingEnv(_table(closes), EnvConfig(**config), audit=audit)


def _always(action: int) -> Policy:
    return lambda _cursor, _info: action


def _script(actions: dict[int, int]) -> Policy:
    """Act ``actions[cursor]`` at the listed cursors, HOLD everywhere else."""
    return lambda cursor, _info: actions.get(cursor, ACTION_HOLD)


def _rollout(
    env: SingleTickerTradingEnv, policy: Policy, *, start: int, max_steps: int | None = None
) -> tuple[list[Obs], list[float], list[dict[str, Any]], bool, bool]:
    """Reset at ``start``; returns ``(observations incl. the reset one, rewards, infos incl. the
    reset one, terminated, truncated)``."""
    obs, info = env.reset(options={"start_index": start})
    observations: list[Obs] = [obs]
    rewards: list[float] = []
    infos: list[dict[str, Any]] = [info]
    terminated = truncated = False
    while not (terminated or truncated):
        if max_steps is not None and len(rewards) >= max_steps:
            break
        obs, reward, terminated, truncated, info = env.step(np.int64(policy(env.cursor, infos[-1])))
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
    return observations, rewards, infos, terminated, truncated


# --- the bar pairing --------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 1, 2, 5])
def test_fill_index_is_the_first_close_not_known_at_decision_time(lag: int) -> None:
    table = _table(_walk(120, seed=1))
    for t in range(MAX_LOOKBACK, 100):
        fill = fill_index_for(t, execution_lag=lag)
        assert fill == t + 1 + lag
        # The fill price is NEVER public when the agent decides (strictly later knowledge time).
        assert table.bars[fill].known_at > table.bars[t].known_at
    with pytest.raises(EnvConfigError, match="execution_lag"):
        fill_index_for(60, execution_lag=-1)


def test_order_fills_at_the_next_close_and_earns_only_from_the_fill() -> None:
    # Flat warmup, then +10% INTO the fill close and +10% after it. The jump C_s -> C_{s+1} is
    # already-unknowable at decision s, but a lookahead fill (at C_s) would be credited with it.
    s = 60
    closes = [100.0] * (s + 1) + [110.0, 121.0] + [121.0] * 20
    env = _env(closes, cost=ZERO_COST, episode_length=10)
    env.reset(options={"start_index": s})

    obs, reward, _, _, info = env.step(np.int64(ACTION_BUY))
    assert reward == 0.0  # flat over s -> s+1: the +10% into the fill close is NOT earned
    assert info["position"] == 1.0
    assert info["entry_index"] == s + 1
    assert info["fill"] == {
        "decision_index": s,
        "fill_index": s + 1,
        "side": "buy",
        "position_before": 0.0,
        "position_after": 1.0,
        "cost_fraction": 0.0,
    }
    assert obs[7] == 1.0
    assert obs[8] == 0.0  # just filled at C_{s+1}: no unrealized P&L yet

    obs, reward, _, _, info = env.step(np.int64(ACTION_HOLD))
    assert reward == pytest.approx(100 * math.log(1.1), abs=1e-12)  # C_{s+1} -> C_{s+2}
    assert info["fill"] is None
    assert obs[8] == pytest.approx(0.1 / _UNREALIZED_SCALE, rel=1e-6)


def test_state_and_reward_agree_on_the_fill_price() -> None:
    closes = _walk(220, seed=7)
    s, buy_at, sell_at = 60, 64, 120
    env = _env(closes, cost=ZERO_COST, episode_length=100)
    observations, rewards, infos, _, _ = _rollout(
        env, _script({buy_at: ACTION_BUY, sell_at: ACTION_SELL}), start=s
    )
    fill_buy = buy_at + 1
    for k, info in enumerate(infos):
        cursor = info["cursor"]
        assert cursor == s + k
        obs = observations[k]
        if fill_buy <= cursor <= sell_at:
            assert info["entry_index"] == fill_buy
            assert obs[7] == 1.0
            ratio = closes[cursor] / closes[fill_buy]
            # Equity, the summed reward and the observation's unrealized P&L all use C_fill.
            assert info["equity"] == pytest.approx(ratio, rel=1e-12)
            assert math.exp(sum(rewards[:k]) / 100) == pytest.approx(ratio, rel=1e-9)
            expected = np.float32(np.clip((ratio - 1.0) / _UNREALIZED_SCALE, -10, 10))
            assert obs[8] == pytest.approx(expected, abs=1e-5)
        else:
            assert info["entry_index"] is None
            assert obs[7] == 0.0
            assert obs[8] == 0.0
    sold = infos[sell_at - s + 1]
    assert sold["fill"]["fill_index"] == sell_at + 1
    assert sold["fill"]["side"] == "sell"
    assert infos[-1]["equity"] == pytest.approx(closes[sell_at + 1] / closes[fill_buy], rel=1e-12)


def test_pending_order_is_not_held_under_execution_lag() -> None:
    s = 60
    env = _env(_walk(200, seed=3), cost=ZERO_COST, execution_lag=2, episode_length=30)
    obs, info = env.reset(options={"start_index": s})

    # Cursor s: BUY (fills at close s+3). The step at cursor c reveals bar c+1.
    obs, _, _, _, info = env.step(np.int64(ACTION_BUY))
    assert (info["cursor"], info["position"], info["pending_orders"], obs[7]) == (s + 1, 0.0, 1, 0)
    # Cursor s+1: a second BUY is judged against the COMMITTED (pending) long -> no-op.
    obs, _, _, _, info = env.step(np.int64(ACTION_BUY))
    assert info["action_was_noop"] is True
    assert (info["cursor"], info["position"], info["pending_orders"], obs[7]) == (s + 2, 0.0, 1, 0)
    # Cursor s+2: SELL is legal vs the committed long (fills at s+5); this step reveals s+3, where
    # the BUY decided at s fills — held only now, with the ACTUAL fill close as entry.
    obs, _, _, _, info = env.step(np.int64(ACTION_SELL))
    assert info["action_was_noop"] is False
    assert info["fill"]["decision_index"] == s
    assert (info["cursor"], info["position"], info["entry_index"], obs[7]) == (s + 3, 1.0, s + 3, 1)
    assert info["pending_orders"] == 1
    obs, _, _, _, info = env.step(np.int64(ACTION_HOLD))
    assert (info["position"], info["fill"]) == (1.0, None)
    obs, _, _, _, info = env.step(np.int64(ACTION_HOLD))  # reveals s+5: the SELL fills
    assert info["fill"]["decision_index"] == s + 2
    assert info["fill"]["fill_index"] == s + 5
    assert (info["position"], info["entry_index"], info["pending_orders"], obs[7]) == (
        0.0,
        None,
        0,
        0,
    )


def test_reward_reads_only_bars_t_and_t_plus_one_and_the_observation_never_t_plus_one() -> None:
    s, t = 60, 75
    base = _walk(200, seed=11)
    bumped_next = [c * 1.05 if i == t + 1 else c for i, c in enumerate(base)]
    poisoned_later = [c * 3.0 if i > t + 1 else c for i, c in enumerate(base)]
    poisoned_future = [c * 3.0 if i > t else c for i, c in enumerate(base)]

    def run(closes: Sequence[float], policy: Policy) -> tuple[list[Obs], list[float]]:
        env = _env(closes, audit=True, episode_length=40)
        observations, rewards, _, _, _ = _rollout(env, policy, start=s, max_steps=t - s + 1)
        return observations, rewards

    held = _always(ACTION_BUY)
    obs_base, rew_base = run(base, held)
    obs_next, rew_next = run(bumped_next, held)
    _, rew_later = run(poisoned_later, held)
    obs_future, _ = run(poisoned_future, held)
    k = t - s  # observation index at cursor t; reward index of the step taken at t
    # The observation at t is identical whatever happens after t (C_{t+1} included).
    for i in range(k + 1):
        assert np.array_equal(obs_base[i], obs_next[i])
        assert np.array_equal(obs_base[i], obs_future[i])
    # The reward of step t moves with C_{t+1} (while a position is held over t -> t+1) ...
    assert rew_base[:k] == rew_next[:k]
    assert rew_base[k] != rew_next[k]
    # ... and is invariant to every bar after t+1.
    assert rew_base == rew_later

    # While FLAT over t -> t+1 the step's reward cannot depend on C_{t+1}: a probe of the reward
    # direction must hold a position at t.
    _, flat_base = run(base, _always(ACTION_HOLD))
    _, flat_next = run(bumped_next, _always(ACTION_HOLD))
    assert flat_base == flat_next


def test_every_bar_read_for_an_observation_was_public_at_the_decision_time() -> None:
    env = _env(_walk(200, seed=5), audit=True, episode_length=40)
    env.reset(options={"start_index": 60})
    rng = random.Random(0)
    for _ in range(40):
        decision_at = env.decision_at
        assert env._audit_reads, "an observation that read no bars would pass vacuously"
        assert max(bar.known_at for bar in env._audit_reads) <= decision_at
        _, _, terminated, truncated, info = env.step(np.int64(rng.randrange(3)))
        if info["entry_index"] is not None:
            assert info["entry_index"] <= env.cursor
            assert env._bars.bars[info["entry_index"]] in env._audit_reads
        if terminated or truncated:
            break


def test_audit_records_exactly_the_window_and_an_entry_bar_older_than_it() -> None:
    # Hold one position for longer than the 50-bar window, so the entry (fill) bar falls OUTSIDE
    # the trailing window: the audit must then name it explicitly, or the check has no teeth.
    s, window = 60, WARMUP_BARS
    env = _env(_walk(300, seed=31), audit=True, episode_length=120)
    env.reset(options={"start_index": s})
    bars = env._bars.bars
    env.step(np.int64(ACTION_BUY))  # fills at s+1
    for _ in range(window + 10):
        _, _, _, _, info = env.step(np.int64(ACTION_HOLD))
    cursor, entry = env.cursor, info["entry_index"]
    assert entry == s + 1
    assert entry < cursor + 1 - window  # older than the observation's trailing window
    assert env._audit_reads == [*bars[cursor + 1 - window : cursor + 1], bars[entry]]
    assert max(bar.known_at for bar in env._audit_reads) <= env.decision_at
    env.step(np.int64(ACTION_SELL))  # flat again: the window only
    assert env._audit_reads == list(bars[env.cursor + 1 - window : env.cursor + 1])


# --- spec acceptance tests ---------------------------------------------------------------------


def test_reset_is_deterministic_for_a_seed() -> None:
    table = _table(_walk(600, seed=2))
    first = SingleTickerTradingEnv(table)
    second = SingleTickerTradingEnv(table)
    obs_a, info_a = first.reset(seed=42)
    obs_b, info_b = second.reset(seed=42)
    assert info_a["start_index"] == info_b["start_index"]
    assert np.array_equal(obs_a, obs_b)
    for _ in range(30):
        action = first.action_space.sample()
        step_a = first.step(action)
        step_b = second.step(action)
        assert np.array_equal(step_a[0], step_b[0])
        assert step_a[1:4] == step_b[1:4]
    # Random starts actually vary across seeds, and stay within the legal range.
    lo, hi = first.start_bounds
    starts = {first.reset(seed=seed)[1]["start_index"] for seed in range(20)}
    assert len(starts) > 1
    assert all(lo <= start <= hi for start in starts)


def test_forced_start_index_outside_range_raises() -> None:
    env = _env(_walk(400, seed=4), episode_length=100)
    lo, hi = env.start_bounds
    assert (lo, hi) == (MAX_LOOKBACK, 400 - 1 - 100)
    for start in (lo - 1, hi + 1, -1):
        with pytest.raises(EnvConfigError, match="start_index"):
            env.reset(options={"start_index": start})
    for start in (lo, hi):
        assert env.reset(options={"start_index": start})[1]["start_index"] == start
    with pytest.raises(EnvConfigError, match="start_index must be an int"):
        env.reset(options={"start_index": True})
    with pytest.raises(EnvConfigError, match="unknown reset options"):
        env.reset(options={"start": lo})
    fixed = _env(_walk(400, seed=4), episode_length=100, random_start=False)
    assert fixed.reset(seed=9)[1]["start_index"] == lo


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_always_buy_reproduces_buy_and_hold_at_zero_cost(lag: int) -> None:
    closes = _walk(300, seed=21, vol=0.012)
    s, length = 80, 150
    closes[s + 1] = closes[s] * 1.08  # make the decision-close ratio visibly different
    env = _env(closes, cost=ZERO_COST, execution_lag=lag, episode_length=length)
    _, rewards, infos, terminated, truncated = _rollout(env, _always(ACTION_BUY), start=s)
    assert (terminated, truncated, len(rewards)) == (False, True, length)
    end, first_fill = s + length, s + 1 + lag
    buy_and_hold = closes[end] / closes[first_fill]
    # §4.8 Test C, aligned with the timing contract: buy-and-hold from the FIRST FILL close.
    assert math.exp(sum(rewards) / 100) == pytest.approx(buy_and_hold, rel=1e-9)
    assert infos[-1]["equity"] == pytest.approx(buy_and_hold, rel=1e-12)
    # NOT from the decision close: that ratio is what a lookahead fill at C_start would earn.
    assert abs(math.exp(sum(rewards) / 100) - closes[end] / closes[s]) > 1e-3
    # With the default cost the only difference is the single entry fill.
    priced = _env(closes, execution_lag=lag, episode_length=length)
    _, _, priced_infos, _, _ = _rollout(priced, _always(ACTION_BUY), start=s)
    expected = buy_and_hold * (1 - DEFAULT_COST.one_way_bps * BPS)
    assert priced_infos[-1]["equity"] == pytest.approx(expected, rel=1e-12)


def test_terminated_only_on_ruin_and_truncated_on_budget() -> None:
    # Budget exhaustion is TRUNCATION (PPO bootstraps V(s')), never termination.
    env = _env(_walk(200, seed=8), episode_length=25)
    _, _, infos, terminated, truncated = _rollout(env, _always(ACTION_HOLD), start=60)
    assert (terminated, truncated, len(infos) - 1) == (False, True, 25)

    # Ruin is TERMINATION: a -15%/day crash while long drives equity <= ruin_fraction early.
    crash = [100.0] * 61 + [100.0 * 0.85**k for k in range(1, 40)] + [1.0] * 60
    env = _env(crash, episode_length=60, ruin_fraction=0.2)
    env.reset(options={"start_index": 60})
    flags: list[tuple[bool, bool]] = []
    info: dict[str, Any] = {}
    for _ in range(60):
        _, _, terminated, truncated, info = env.step(np.int64(ACTION_BUY))
        flags.append((terminated, truncated))
        if terminated or truncated:
            break
    assert flags[-1] == (True, False)
    assert all(flag == (False, False) for flag in flags[:-1])
    assert len(flags) < 60
    assert info["equity"] <= env.config.ruin_fraction
    # A finished episode cannot be stepped.
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.int64(ACTION_HOLD))


def test_ruin_boundary_terminates_when_equity_lands_exactly_on_ruin_fraction() -> None:
    # 100 -> 25 is an exact binary ratio: equity lands on 0.25 EXACTLY, which is ruin (<=).
    s = 60

    def crash_to(close: float, ruin_fraction: float) -> tuple[float, bool, bool]:
        closes = [100.0] * (s + 2) + [close] * 20
        env = _env(closes, cost=ZERO_COST, episode_length=10, ruin_fraction=ruin_fraction)
        env.reset(options={"start_index": s})
        env.step(np.int64(ACTION_BUY))  # fills at C_{s+1} = 100
        _, _, terminated, truncated, info = env.step(np.int64(ACTION_HOLD))  # 100 -> close
        return info["equity"], terminated, truncated

    equity, terminated, truncated = crash_to(25.0, ruin_fraction=0.25)
    assert equity == 0.25
    assert (terminated, truncated) == (True, False)
    # Just above the boundary (257/1024, also exact) is not ruin.
    equity, terminated, truncated = crash_to(100.0 * 257 / 1024, ruin_fraction=0.25)
    assert equity == 257 / 1024
    assert (terminated, truncated) == (False, False)


def test_step_before_reset_raises() -> None:
    env = _env(_walk(200, seed=8), episode_length=25)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.int64(ACTION_HOLD))


def test_illegal_action_is_a_noop_not_a_position_flip() -> None:
    s = 60
    closes = _walk(200, seed=9)
    env = _env(closes, episode_length=20)
    env.reset(options={"start_index": s})

    _, _, _, _, info = env.step(np.int64(ACTION_SELL))  # SELL while flat
    assert info["action_was_noop"] is True
    assert (info["action_requested"], info["action_effective"]) == (ACTION_SELL, ACTION_HOLD)
    assert (info["position"], info["pending_orders"], info["n_trades"]) == (0.0, 0, 0)

    _, _, _, _, info = env.step(np.int64(ACTION_BUY))
    assert (info["action_was_noop"], info["position"], info["n_trades"]) == (False, 1.0, 1)
    _, _, _, _, info = env.step(np.int64(ACTION_BUY))  # BUY while held: NOT a flip to flat
    assert info["action_was_noop"] is True
    assert (info["action_effective"], info["position"], info["n_trades"]) == (ACTION_HOLD, 1.0, 1)
    assert info["fill"] is None

    # The penalty is a learning lever: it lowers exactly the no-op step's reward, nothing else.
    script = {s: ACTION_SELL, s + 1: ACTION_BUY, s + 2: ACTION_BUY}
    _, plain, _, _, _ = _rollout(_env(closes, episode_length=20), _script(script), start=s)
    penalized_env = _env(closes, episode_length=20, invalid_action_penalty=0.5)
    _, penalized, _, _, _ = _rollout(penalized_env, _script(script), start=s)
    deltas = [a - b for a, b in zip(plain, penalized, strict=True)]
    assert deltas[0] == pytest.approx(0.5)
    assert deltas[2] == pytest.approx(0.5)
    assert all(d == 0.0 for i, d in enumerate(deltas) if i not in (0, 2))

    for bad in (3, -1):
        with pytest.raises(ValueError, match="action"):
            env.step(np.int64(bad))
    with pytest.raises(ValueError, match="action"):
        env.step(cast(Any, True))


def test_costs_reduce_equity_on_every_position_change() -> None:
    # A constant price isolates the cost: every fill multiplies equity by exactly (1 - 1 bp).
    s = 60
    env = _env([100.0] * 120, episode_length=40)
    actions = {s + 2 * i: (ACTION_BUY if i % 2 == 0 else ACTION_SELL) for i in range(10)}
    _, rewards, infos, _, _ = _rollout(env, _script(actions), start=s)
    one_way = DEFAULT_COST.one_way_bps * BPS
    for reward, before, after in zip(rewards, infos, infos[1:], strict=False):
        if after["fill"] is None:
            assert reward == 0.0
            assert after["equity"] == before["equity"]
        else:
            assert after["fill"]["cost_fraction"] == pytest.approx(one_way)
            assert after["equity"] == pytest.approx(before["equity"] * (1 - one_way), rel=1e-15)
            assert reward == pytest.approx(100 * math.log(1 - one_way), rel=1e-12)
            assert after["equity"] < before["equity"]
    final = infos[-1]
    assert (final["n_trades"], final["turnover"]) == (10, 10.0)
    assert final["equity"] == pytest.approx((1 - one_way) ** 10, rel=1e-12)
    assert final["cost_paid"] == pytest.approx(1 - (1 - one_way) ** 10, rel=1e-9)


def test_cost_paid_is_charged_on_the_equity_marked_at_the_fill_close() -> None:
    # A SELL fills at close t+1 AFTER the move C_t -> C_{t+1}: its cost is charged on the equity
    # marked at C_{t+1}, not on the pre-move equity. A +20% move into the sell fill separates them.
    s, buy_at, sell_at = 60, 62, 80
    closes = _walk(200, seed=37)
    closes[sell_at + 1] = closes[sell_at] * 1.2
    env = _env(closes, episode_length=40, cost=STRESS_COST)
    _, _, infos, _, _ = _rollout(env, _script({buy_at: ACTION_BUY, sell_at: ACTION_SELL}), start=s)
    c = STRESS_COST.one_way_bps * BPS
    buy_fill, sell_fill = buy_at + 1, sell_at + 1
    cost_buy = 1.0 * c  # flat before the buy: marked equity == 1.0
    marked_at_sell = (1 - c) * closes[sell_fill] / closes[buy_fill]
    expected = cost_buy + marked_at_sell * c
    final = infos[-1]
    assert final["n_trades"] == 2
    assert final["cost_paid"] == pytest.approx(expected, rel=1e-12)
    unmarked = cost_buy + (1 - c) * closes[sell_at] / closes[buy_fill] * c
    assert abs(final["cost_paid"] - unmarked) > 1e-6
    assert final["equity"] == pytest.approx(marked_at_sell * (1 - c), rel=1e-12)


def test_churning_policy_loses_money_under_default_costs() -> None:
    s = 60
    closes = _walk(300, seed=13)

    def churn(_cursor: int, info: dict[str, Any]) -> int:
        return ACTION_SELL if info["position"] == 1.0 else ACTION_BUY

    flat = _env([100.0] * 300, episode_length=200)
    _, _, flat_infos, _, _ = _rollout(flat, churn, start=s)
    assert flat_infos[-1]["n_trades"] == 200
    assert flat_infos[-1]["equity"] < 1.0  # churning a flat market strictly loses

    free = _env(closes, cost=ZERO_COST, episode_length=200)
    priced = _env(closes, episode_length=200)
    _, _, free_infos, _, _ = _rollout(free, churn, start=s)
    _, _, priced_infos, _, _ = _rollout(priced, churn, start=s)
    n_trades = priced_infos[-1]["n_trades"]
    assert n_trades == free_infos[-1]["n_trades"] == 200
    # Costs are the ONLY difference, compounding once per fill.
    one_way = DEFAULT_COST.one_way_bps * BPS
    assert priced_infos[-1]["equity"] == pytest.approx(
        free_infos[-1]["equity"] * (1 - one_way) ** n_trades, rel=1e-12
    )
    assert priced_infos[-1]["equity"] < free_infos[-1]["equity"]


def test_dsr_reward_variant_returns_finite_values_including_at_the_singularity() -> None:
    # Always flat: every period return is exactly 0 -> B - A^2 == 0, the do-nothing singularity.
    env = _env(_walk(200, seed=17), reward="dsr", episode_length=60)
    _, rewards, infos, _, _ = _rollout(env, _always(ACTION_HOLD), start=60)
    assert rewards == [0.0] * 60
    assert infos[-1]["dsr_state"] == (0.0, 0.0)

    env = _env(_walk(200, seed=17), reward="dsr", episode_length=60)
    rng = random.Random(1)
    _, rewards, infos, _, _ = _rollout(env, lambda _c, _i: rng.randrange(3), start=60)
    assert all(math.isfinite(r) for r in rewards)
    assert any(r != 0.0 for r in rewards)
    assert all(math.isfinite(v) for v in infos[-1]["dsr_state"])
    # A fresh episode restarts the EMA moments.
    assert env.reset(options={"start_index": 60})[1]["dsr_state"] == (0.0, 0.0)


# --- API surface -------------------------------------------------------------------------------

_BASE_INFO_KEYS = {
    "start_index",
    "cursor",
    "decision_at",
    "session_ts",
    "equity",
    "position",
    "entry_index",
    "pending_orders",
    "turnover",
    "cost_paid",
    "n_trades",
    "drawdown",
}
_STEP_INFO_KEYS = _BASE_INFO_KEYS | {
    "action_requested",
    "action_effective",
    "action_was_noop",
    "fill",
}


def test_info_schema_and_observation_box() -> None:
    closes = _walk(200, seed=19)
    env = _env(closes, episode_length=30)
    obs, info = env.reset(options={"start_index": 60})
    assert set(info) == _BASE_INFO_KEYS
    assert info["decision_at"] == env.decision_at == env._bars.bars[60].known_at
    assert info["session_ts"] == env._bars.bars[60].session_ts
    assert (info["equity"], info["position"], info["drawdown"]) == (1.0, 0.0, 0.0)
    assert obs.dtype == np.float32
    assert obs.shape == (OBS_DIM,)
    peak = 1.0
    for action in [ACTION_BUY] + [ACTION_HOLD] * 20 + [ACTION_SELL]:
        obs, _, _, _, info = env.step(np.int64(action))
        assert set(info) == _STEP_INFO_KEYS
        assert env.observation_space.contains(obs)
        assert info["decision_at"] == env.decision_at
        peak = max(peak, info["equity"])
        assert info["drawdown"] == pytest.approx(1 - info["equity"] / peak)
    dsr_env = _env(closes, episode_length=30, reward="dsr")
    assert set(dsr_env.reset(options={"start_index": 60})[1]) == _BASE_INFO_KEYS | {"dsr_state"}


def test_config_and_bars_are_validated() -> None:
    closes = _walk(200, seed=23)
    table = _table(closes)
    bad_configs: list[dict[str, Any]] = [
        {"execution_lag": -1},
        {"execution_lag": 252},
        {"episode_length": 0},
        {"episode_length": True},
        {"ruin_fraction": 1.0},
        {"ruin_fraction": -0.1},
        {"invalid_action_penalty": -0.1},
        {"invalid_action_penalty": float("nan")},
        {"reward": "sharpe"},
        {"symbol": ""},
    ]
    for kwargs in bad_configs:
        with pytest.raises(EnvConfigError):
            EnvConfig(**kwargs)
    with pytest.raises(EnvConfigError, match="config is for"):
        SingleTickerTradingEnv(_table(closes, symbol="QQQ"))
    with pytest.raises(EnvConfigError, match="cannot hold"):
        SingleTickerTradingEnv(table, EnvConfig(episode_length=200 - MAX_LOOKBACK))
    # The smallest table that holds one episode: lo == hi.
    tight = SingleTickerTradingEnv(table, EnvConfig(episode_length=200 - MAX_LOOKBACK - 1))
    assert tight.start_bounds == (MAX_LOOKBACK, MAX_LOOKBACK)
    assert EnvConfig().cost.one_way_bps > 0  # a zero-cost default is a correctness defect
    assert EnvConfig().cost == CostModel()


def test_gymnasium_env_checker_passes() -> None:
    env = _env(_walk(400, seed=29), episode_length=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env)
    # The only notice is that an unregistered env (no gymnasium.make spec) cannot be re-made in
    # alternative render modes — this env declares none.
    assert all("not having a spec" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


@pytest.mark.rl
def test_stable_baselines3_env_checker_passes() -> None:
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    env = _env(_walk(400, seed=29), episode_length=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sb3_checker.check_env(env, warn=True, skip_render_check=True)
    assert [str(w.message) for w in caught] == []


def test_runs_a_full_random_episode_on_the_frozen_spy_dataset() -> None:
    table = BarTable.from_dataset(rl_dataset_from_json(_FROZEN.read_bytes()))
    env = SingleTickerTradingEnv(table)
    closes = table.closes()
    obs, info = env.reset(seed=123)
    start = info["start_index"]
    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(np.int64(ACTION_BUY))
        rewards.append(reward)
    assert (terminated, truncated, len(rewards)) == (False, True, 252)
    assert all(math.isfinite(r) for r in rewards)
    buy_and_hold = closes[start + 252] / closes[start + 1]
    expected = buy_and_hold * (1 - DEFAULT_COST.one_way_bps * BPS)
    assert info["equity"] == pytest.approx(expected, rel=1e-12)
