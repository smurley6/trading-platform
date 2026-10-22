"""Replay a policy through the env and record what happened (task-rl-07; program doc §5.1 Tier B).

HEAVY half of ``trading.rl``: imports ``trading.rl.env`` (gymnasium + numpy, both in ``[dev]``),
never torch / stable-baselines3. Imported explicitly as ``trading.rl.scoring``.

**One replay path for everything.** The agent's exported policy (rl-09) and all four baselines
(``rl/baselines.py``) are scored by the SAME :func:`rollout`: a fresh
:class:`~trading.rl.env.SingleTickerTradingEnv` over the given bars with ``random_start=False``,
reset at ``segment.start``, stepped ``segment.length`` times with the policy's actions — so every
number inherits the env's verified timing contract (fill at ``C_{t+1+lag}``), cost accounting and
leakage guarantees (``tests/unit/rl/test_env_leakage.py``). Results are recorded straight from the
env's info dicts (equity, position, bar indices, fill events — never a price level) into the pure
:class:`~trading.rl.metrics.EpisodeResult`; ``rl/metrics.py`` turns them into numbers.

**Determinism.** A replay is a pure function of ``(bars, segment, policy factory, seed)``: the
env's start is fixed (its ``np_random`` is seeded with ``seed`` but never drawn from when
``start_index`` is given), and the policy gets ``factory(seed)`` — a fresh per-episode policy whose
only randomness (the random baseline's ``random.Random(seed)``) comes from that seed.

**The policy's inputs** at decision bar ``t = info["cursor"]``: the env observation (as Python
floats) and ``history = closes[t - 49 : t + 1]`` — the trailing ``WARMUP_BARS`` closes the env
built that observation from (its ``_observe`` reads ``closes[: t + 1]`` of the same
``BarTable.closes()`` tuple; ``build_observation`` reads only the last 50). Every close in it has
``known_at <= bars[t].known_at == info["decision_at"]``; the future is physically absent.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from trading.domain.rl_features import WARMUP_BARS
from trading.rl.bars import BarTable
from trading.rl.baselines import UNIFORM_TRADE_PROBABILITY, PolicyFactory, random_policy
from trading.rl.env import EnvConfig, SingleTickerTradingEnv
from trading.rl.metrics import (
    NULL_BAND_SEEDS,
    EpisodeResult,
    FillRecord,
    NullBand,
    ScoringError,
    Segment,
    null_band,
)


def _fill_record(fill: dict[str, Any]) -> FillRecord:
    return FillRecord(
        decision_index=int(fill["decision_index"]),
        fill_index=int(fill["fill_index"]),
        side=str(fill["side"]),
        position_before=float(fill["position_before"]),
        position_after=float(fill["position_after"]),
        cost_fraction=float(fill["cost_fraction"]),
    )


def rollout(policy: PolicyFactory, bars: BarTable, segment: Segment, *, seed: int) -> EpisodeResult:
    """Replay ``policy(seed)`` over ``segment`` of ``bars`` through the env; record every mark.

    Raises the env's ``EnvConfigError`` if the segment does not fit the bars (``start`` outside
    ``[MAX_LOOKBACK, len(bars) - 1 - length]``) and ``ValueError`` on an action outside
    ``Discrete(3)``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ScoringError(f"seed must be an int, got {seed!r}")
    env = SingleTickerTradingEnv(
        bars,
        EnvConfig(
            symbol=bars.symbol,
            episode_length=segment.length,
            cost=segment.cost,
            execution_lag=segment.execution_lag,
            random_start=False,
        ),
    )
    closes = bars.closes()
    act = policy(seed)
    obs, info = env.reset(seed=seed, options={"start_index": segment.start})

    cursors = [int(info["cursor"])]
    marked_at = [info["decision_at"]]
    equity = [float(info["equity"])]
    positions = [float(info["position"])]
    actions: list[int] = []
    requested: list[int] = []
    fills: list[FillRecord] = []
    n_noops = 0
    ruined = False
    done = False
    while not done:
        cursor = int(info["cursor"])
        history = closes[cursor + 1 - WARMUP_BARS : cursor + 1]
        action = act(tuple(float(x) for x in obs), history)
        obs, _reward, terminated, truncated, info = env.step(np.int64(action))
        requested.append(int(info["action_requested"]))
        actions.append(int(info["action_effective"]))
        n_noops += bool(info["action_was_noop"])
        if info["fill"] is not None:
            fills.append(_fill_record(info["fill"]))
        cursors.append(int(info["cursor"]))
        marked_at.append(info["decision_at"])
        equity.append(float(info["equity"]))
        positions.append(float(info["position"]))
        ruined = bool(terminated)
        done = bool(terminated or truncated)

    return EpisodeResult(
        segment=segment,
        seed=seed,
        cursors=tuple(cursors),
        marked_at=tuple(marked_at),
        equity=tuple(equity),
        positions=tuple(positions),
        actions=tuple(actions),
        requested_actions=tuple(requested),
        n_noops=n_noops,
        n_trades=int(info["n_trades"]),
        turnover=float(info["turnover"]),
        cost_paid=float(info["cost_paid"]),
        fills=tuple(fills),
        ruined=ruined,
    )


def rollout_seeds(
    policy: PolicyFactory, bars: BarTable, segment: Segment, seeds: Sequence[int]
) -> tuple[EpisodeResult, ...]:
    """:func:`rollout` once per seed, in the given order (a seed LIST, never one seed: a random
    or trained policy is a distribution, not a point)."""
    if not seeds:
        raise ScoringError("need at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ScoringError(f"seeds must be distinct, got {tuple(seeds)}")
    return tuple(rollout(policy, bars, segment, seed=seed) for seed in seeds)


def random_null_band(
    bars: BarTable,
    segment: Segment,
    *,
    seeds: Sequence[int] = tuple(range(NULL_BAND_SEEDS)),
    trade_probability: float = UNIFORM_TRADE_PROBABILITY,
) -> NullBand:
    """The §1 / §5.2 matched random null distribution over one segment: the random baseline at
    ``trade_probability`` (set it with ``metrics.matched_trade_probability`` to turnover-match a
    policy) replayed once per seed (default 30: ``0 .. 29``), collapsed by ``metrics.null_band``."""
    return null_band(rollout_seeds(random_policy(trade_probability), bars, segment, seeds))
