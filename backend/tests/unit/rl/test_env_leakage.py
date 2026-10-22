"""The leakage suite: executable proof that the env cannot see the future (task-rl-06; program doc
§1 item 1, §4.8 Tests A / A' / B / C-direction / E).

gymnasium + numpy are in ``[dev]``, so all of this runs in the default ``pytest`` (no ``rl``
marker, no torch).

**Two independent detectors**, each with a NEGATIVE CONTROL proving it can fail:

1. **Future poisoning (Test A)** — :func:`_run_invariance_check`. Build a clean env and a poisoned
   twin whose every close AFTER index ``k`` is multiplied by ``3.0`` / ``0.25`` / ``nan``; step both
   in lockstep with the same action script and require every observation, reward, termination flag
   and info dict to be BIT-IDENTICAL for every decision cursor ``<= k``. The step taken AT cursor
   ``k`` legitimately reads ``C_{k+1}`` for its reward (the timing contract, ``rl/env.py``), so the
   lockstep stops BEFORE stepping from ``k``. Finite factors poison the ``BarTable`` itself (every
   view of the closes is poisoned); ``BarTable`` rejects NaN, so the NaN probe is written into the
   env's single float view of the closes, ``env._closes`` (§4.8 note) — after which ANY arithmetic
   touch of a future bar turns into NaN (or a ``ValueError`` from the fail-closed builders).
   Detection is black-box: it does not care HOW a future bar was reached.
2. **Read audit (Test B)** — :func:`_run_audit_check`. With ``audit=True`` the env records
   exactly the bars each observation read (``env._audit_reads``: the trailing 50-bar window + the
   entry bar); every one must have ``known_at <= env.decision_at``, checked BEFORE stepping. It is
   white-box and self-reported: it sees reads made THROUGH the audited path (a mis-sliced window)
   and is blind to a read that bypasses it — which is exactly what the poison detector is for
   (pinned below as a coverage matrix, not left as a claim).

**Why the negative controls make the suite non-vacuous.** ``_LeakyEnv`` leaks ``C_{t+1}`` — the
very next close — so under poison-after-``k`` it diverges at cursor ``k`` and NOWHERE earlier.
Hence: if ``_poison`` started one bar late, or the lockstep stopped one cursor early, the control
would pass and its test would fail; if ``_poison`` started one bar early, the honest env's held
reward of the step from ``k - 1`` would diverge and Test A would fail. The pair (honest env passes,
leaky env fails) pins both the poison boundary and the comparison horizon exactly.

The reward identity ``exp(sum(r)/100) == C[start+len] / C[start+1+lag]`` (Test C) is pinned in
``test_env.py`` and is not repeated here; this module covers the poison / audit / negative-control
/ invariance family.
"""

import contextlib
import functools
import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tests.unit.rl.conftest import (
    bar_table,
    frozen_spy_table,
    synthetic_random_walk,
    with_closes,
)
from trading.domain.rl_features import MAX_LOOKBACK, WARMUP_BARS
from trading.rl.bars import BarTable
from trading.rl.costs import STRESS_COST
from trading.rl.env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    EnvConfig,
    SingleTickerTradingEnv,
)

_BACKEND = Path(__file__).parents[3]
_ENV_SOURCE = _BACKEND / "src" / "trading" / "rl" / "env.py"
_FEATURES_SOURCE = _BACKEND / "src" / "trading" / "domain" / "rl_features.py"

Obs = npt.NDArray[np.float32]
Script = Callable[[int, dict[str, Any]], int]
EnvClass = type[SingleTickerTradingEnv]

_FACTORS = [3.0, 0.25, float("nan")]
_FACTOR_IDS = ["x3", "x0.25", "nan"]
_LAGS = (0, 1, 3)
_RUNWAY = 40  # default number of decisions compared before the horizon k
_SYNTHETIC_SEEDS = (606, 707)
_SYNTHETIC_BARS = 400


# --- the poison ---------------------------------------------------------------------------------


def _poison(closes: Sequence[float], k: int, *, factor: float) -> list[float]:
    """Replace every close AFTER index k. Anything the env legitimately sees is untouched."""
    return [c if i <= k else c * factor for i, c in enumerate(closes)]


def _poisoned_env(
    env_cls: EnvClass, table: BarTable, config: EnvConfig, *, k: int, factor: float
) -> SingleTickerTradingEnv:
    """``env_cls`` over ``table`` with every close after ``k`` poisoned. Finite factors go through
    the ``BarTable`` (same knowledge times, every view poisoned); NaN — which ``BarTable`` rejects —
    goes into the env's one float view of the closes. Guards that the poison is live and starts at
    exactly ``k + 1``: a mis-written poison would make the whole suite vacuously green."""
    clean = table.closes()
    if math.isfinite(factor):
        env = env_cls(with_closes(table, _poison(clean, k, factor=factor)), config, audit=True)
    else:
        env = env_cls(table, config, audit=True)
        env._closes = tuple(_poison(env._closes, k, factor=factor))
    assert env._closes[: k + 1] == clean[: k + 1], "the poison touched a bar at or before k"
    unpoisoned = [i for i in range(k + 1, len(clean)) if env._closes[i] == clean[i]]
    assert not unpoisoned, f"bars after k escaped the poison: {unpoisoned}"
    return env


# --- action scripts (identical for both envs; indexed by cursor) ---------------------------------


def _always_buy(_cursor: int, _info: dict[str, Any]) -> int:
    """Holds a position from the first fill on — the subtle path: the observation's unrealized P&L
    reads the entry (fill) bar, which drifts out of the 50-bar window after 50 decisions."""
    return ACTION_BUY


def _round_trips(cursor: int, _info: dict[str, Any]) -> int:
    """Repeated buy/sell cycles, so entries, exits and fills land at many different bars."""
    if cursor % 6 == 0:
        return ACTION_BUY
    if cursor % 17 == 0:
        return ACTION_SELL
    return ACTION_HOLD


def _random_script(seed: int) -> Script:
    rng = random.Random(seed)
    actions = [rng.randrange(3) for _ in range(4096)]
    return lambda cursor, _info: actions[cursor]


_SCRIPTS: tuple[Script, ...] = (_always_buy, _round_trips, _random_script(11))


# --- the tables --------------------------------------------------------------------------------


@functools.cache
def _synthetic(seed: int) -> BarTable:
    return bar_table(synthetic_random_walk(_SYNTHETIC_BARS, seed))


@functools.cache
def _spy() -> BarTable:
    return frozen_spy_table()


# --- detector 1: future poisoning (Test A) ------------------------------------------------------


@dataclass(frozen=True)
class _Lockstep:
    clean: SingleTickerTradingEnv
    dirty: SingleTickerTradingEnv
    observations_compared: int
    rewards_compared: int
    horizon_info: dict[str, Any]


def _run_invariance_check(
    env_cls: EnvClass = SingleTickerTradingEnv,
    *,
    k: int,
    factor: float,
    table: BarTable | None = None,
    start: int | None = None,
    lag: int = 0,
    script: Script = _always_buy,
    **config: Any,
) -> _Lockstep:
    """THE detector. Raises ``AssertionError`` iff anything the agent or the learner sees at a
    decision cursor ``<= k`` — observation, reward, flags, info — differs between the clean env
    and its twin poisoned after ``k``. The episode starts at ``start`` (default ``k - 40``) and its
    budget ends one step after ``k``, so the lockstep can never be cut short by truncation."""
    table = table if table is not None else _synthetic(_SYNTHETIC_SEEDS[0])
    start = start if start is not None else max(MAX_LOOKBACK, k - _RUNWAY)
    assert MAX_LOOKBACK <= start <= k < len(table) - 1
    env_config = EnvConfig(
        execution_lag=lag,
        episode_length=max(k - start + 1, lag + 1),
        random_start=False,
        **config,
    )
    clean = env_cls(table, env_config, audit=True)
    dirty = _poisoned_env(env_cls, table, env_config, k=k, factor=factor)

    obs_clean, info_clean = clean.reset(options={"start_index": start})
    with _crash_is_a_detection(start, k):
        obs_dirty, info_dirty = dirty.reset(options={"start_index": start})
    observations = rewards = 0
    while True:
        cursor = clean.cursor
        assert dirty.cursor == cursor
        assert np.array_equal(obs_clean, obs_dirty), (
            f"observation at cursor {cursor} changed when only bars after {k} were poisoned:\n"
            f"clean={obs_clean}\ndirty={obs_dirty}"
        )
        info_changed = f"info at cursor {cursor} changed when only bars after {k} were poisoned"
        assert info_clean == info_dirty, info_changed
        assert not np.isnan(obs_clean).any()
        assert not np.isnan(obs_dirty).any()
        observations += 1
        if cursor == k:
            break  # the step FROM k reads C_{k+1} for its reward: legitimately poisoned
        action = np.int64(script(cursor, info_clean))
        obs_clean, r_clean, term_clean, trunc_clean, info_clean = clean.step(action)
        with _crash_is_a_detection(cursor, k):
            obs_dirty, r_dirty, term_dirty, trunc_dirty, info_dirty = dirty.step(action)
        assert r_clean == r_dirty, (
            f"reward of the step from cursor {cursor} changed when only bars after {k} were "
            f"poisoned: clean={r_clean!r} dirty={r_dirty!r}"
        )
        assert (term_clean, trunc_clean) == (term_dirty, trunc_dirty)
        assert not (term_clean or trunc_clean), f"episode ended at cursor {clean.cursor} < k={k}"
        rewards += 1
    # Non-vacuous by construction: one comparison per cursor in [start, k], one reward per step.
    assert (observations, rewards) == (k - start + 1, k - start)
    return _Lockstep(clean, dirty, observations, rewards, info_clean)


@contextlib.contextmanager
def _crash_is_a_detection(cursor: int, k: int) -> Iterator[None]:
    """The NaN probe makes a touch of a future bar FATAL: the fail-closed builders
    (``build_observation`` / ``net_of_cost_growth``) raise ``ValueError`` on a NaN close. In the
    poisoned twin that crash IS a detection, reported as one."""
    try:
        yield
    except ValueError as exc:
        raise AssertionError(
            f"the env crashed at cursor {cursor} when only bars after {k} were poisoned "
            f"(it touched a future bar): {exc}"
        ) from exc


def _starts_for(k: int, *, runway: int) -> list[int]:
    """The earliest legal start (a long held position whose entry leaves the window), a start
    ``runway`` decisions before ``k``, and ``k`` itself (the reset observation alone)."""
    return sorted({MAX_LOOKBACK, max(MAX_LOOKBACK, k - runway), k})


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
@pytest.mark.parametrize("k", [60, 120, 300])
def test_observation_is_invariant_to_every_future_bar(k: int, factor: float) -> None:
    held_at_horizon = entry_older_than_window = 0
    configs: tuple[dict[str, Any], ...] = (
        {},
        {"reward": "dsr", "cost": STRESS_COST, "invalid_action_penalty": 0.1},
    )
    for seed in _SYNTHETIC_SEEDS:
        for lag in _LAGS:
            for start in _starts_for(k, runway=_RUNWAY):
                for script in _SCRIPTS:
                    for config in configs:
                        run = _run_invariance_check(
                            k=k,
                            factor=factor,
                            table=_synthetic(seed),
                            start=start,
                            lag=lag,
                            script=script,
                            **config,
                        )
                        entry = run.horizon_info["entry_index"]
                        if entry is not None:
                            held_at_horizon += 1
                            entry_older_than_window += entry <= k - WARMUP_BARS
    # The subtle path is actually exercised: a position HELD at the horizon, so the unrealized
    # P&L reads the entry bar — for k >= 120 an entry bar older than the 50-bar window.
    assert held_at_horizon > 0
    if k >= 120:
        assert entry_older_than_window > 0


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
@pytest.mark.parametrize("k", [301, 1157, 2008])
def test_frozen_spy_observations_and_rewards_are_invariant_to_every_future_bar(
    k: int, factor: float
) -> None:
    # Real SPY closes (k = 2008 poisons only the last bar of the dataset).
    for lag in (0, 1):
        for start in _starts_for(k, runway=251):  # a full 252-decision year before the horizon
            if start + lag + 1 > len(_spy()) - 1:
                continue  # a lagged episode's budget must exceed its lag: no room at the data's end
            for script in _SCRIPTS:
                run = _run_invariance_check(
                    k=k, factor=factor, table=_spy(), start=start, lag=lag, script=script
                )
                assert run.observations_compared == k - start + 1


def test_poison_changes_every_bar_after_k_and_no_bar_up_to_k() -> None:
    closes = synthetic_random_walk(200, seed=1)
    for k in (60, 120, 198):
        for factor in _FACTORS:
            poisoned = _poison(closes, k, factor=factor)
            assert poisoned[: k + 1] == closes[: k + 1]
            assert all(p != c for p, c in zip(poisoned[k + 1 :], closes[k + 1 :], strict=True))
            assert len(poisoned) == len(closes)


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
def test_the_very_next_close_after_the_horizon_is_already_poisoned(factor: float) -> None:
    # Stepping FROM k while holding reads C_{k+1}: the first poisoned bar. Its effect must show
    # up immediately (finite factors) or fail closed (NaN) — the boundary is tight, not slack.
    run = _run_invariance_check(k=120, factor=factor, script=_always_buy)
    assert run.horizon_info["position"] == 1.0
    _, reward_clean, *_ = run.clean.step(np.int64(ACTION_HOLD))
    if math.isfinite(factor):
        _, reward_dirty, *_ = run.dirty.step(np.int64(ACTION_HOLD))
        assert reward_dirty != reward_clean
        assert reward_dirty - reward_clean == pytest.approx(100 * math.log(factor), rel=1e-9)
    else:
        with pytest.raises(ValueError, match="finite"):
            run.dirty.step(np.int64(ACTION_HOLD))


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
def test_a_seeded_random_episode_start_never_depends_on_prices(factor: float) -> None:
    # The RNG path (reset(seed) -> np_random.integers) must read the bar COUNT, never a close:
    # poisoned and clean twins pick the same start and agree on the reset observation. The poison
    # starts after the LAST legal start, so every start's observation is comparable (with NaN,
    # an observation reading a poisoned bar would crash, as it should).
    table = _synthetic(_SYNTHETIC_SEEDS[1])
    config = EnvConfig(episode_length=100)
    clean = SingleTickerTradingEnv(table, config)
    k = clean.start_bounds[1]
    dirty = _poisoned_env(SingleTickerTradingEnv, table, config, k=k, factor=factor)
    starts = set()
    for seed in range(40):
        obs_clean, info_clean = clean.reset(seed=seed)
        obs_dirty, info_dirty = dirty.reset(seed=seed)
        assert info_clean["start_index"] == info_dirty["start_index"]
        assert np.array_equal(obs_clean, obs_dirty)
        assert info_clean == info_dirty
        starts.add(info_clean["start_index"])
    assert len(starts) > 1


# --- detector 2: the read audit (Test B) ---------------------------------------------------------


def _run_audit_check(
    env: SingleTickerTradingEnv, *, seed: int, script: Script, start: int | None = None
) -> tuple[int, int]:
    """THE audit detector, over one full seeded episode (random start unless ``start`` is given).
    Before every step — and on the final observation — every bar the observation read must have
    been public at the decision time. Returns ``(decisions checked, decisions whose entry bar lay
    outside the trailing window)``."""
    options = None if start is None else {"start_index": start}
    _, info = env.reset(seed=seed, options=options)
    checked = entry_outside_window = 0
    terminated = truncated = False
    while True:
        decision_at = env.decision_at  # captured BEFORE stepping
        reads = env._audit_reads
        assert reads, "an observation that read no bars would pass vacuously"
        latest = max(bar.known_at for bar in reads)
        assert latest <= decision_at, (
            f"the observation at cursor {env.cursor} read a bar public at {latest.isoformat()}, "
            f"after the decision time {decision_at.isoformat()}"
        )
        checked += 1
        entry = info["entry_index"]
        if entry is not None and entry < env.cursor + 1 - WARMUP_BARS:
            assert env._bars.bars[entry] in reads
            entry_outside_window += 1
        if terminated or truncated:
            break
        _, _, terminated, truncated, info = env.step(np.int64(script(env.cursor, info)))
    return checked, entry_outside_window


def test_every_bar_read_was_public_at_the_decision_time() -> None:
    checked = entry_outside_window = 0
    tables = [(_spy(), 252), *((_synthetic(seed), 252) for seed in _SYNTHETIC_SEEDS)]
    for table, length in tables:
        for lag in (0, 1, 3):
            config = EnvConfig(execution_lag=lag, episode_length=length)
            env = SingleTickerTradingEnv(table, config, audit=True)
            for seed in range(6):
                for script in _SCRIPTS:
                    n, outside = _run_audit_check(env, seed=seed, script=script)
                    assert n == length + 1  # every decision of a full episode + the final obs
                    checked += n
                    entry_outside_window += outside
    assert checked > 10_000
    # The entry bar is audited even when it has left the trailing window (long holds).
    assert entry_outside_window > 0


def test_the_latest_bar_an_observation_reads_is_exactly_the_decision_bar() -> None:
    # The window ends AT the cursor: not later (a leak) and not earlier (a stale observation).
    env = SingleTickerTradingEnv(_spy(), EnvConfig(execution_lag=1), audit=True)
    for seed in range(5):
        env.reset(seed=seed)
        rng = random.Random(seed)
        terminated = truncated = False
        while True:
            reads = env._audit_reads
            assert reads[WARMUP_BARS - 1] is env._bars.bars[env.cursor]
            assert max(bar.known_at for bar in reads) == env.decision_at
            if terminated or truncated:
                break
            _, _, terminated, truncated, _ = env.step(np.int64(rng.randrange(3)))


# --- Tests 4/5: the reward direction (reward forward, observation backward) ----------------------


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("source", ["synthetic", "spy"])
def test_reward_uses_the_next_bar_not_the_current_one(source: str, lag: int) -> None:
    # Two series identical except closes[t+1]: the SAME observation at t (and before), a DIFFERENT
    # reward at t — moved by exactly the bump, because the step from t books C_t -> C_{t+1} for the
    # position held over it (always-BUY, already filled: a flat step cannot depend on C_{t+1}).
    table = _synthetic(_SYNTHETIC_SEEDS[0]) if source == "synthetic" else _spy()
    start = 100 if source == "synthetic" else 1500
    bump = 1.01
    for t in (start + 2 + lag, start + 37, start + 120):
        closes = list(table.closes())
        closes[t + 1] *= bump
        bumped = with_closes(table, closes)
        config = EnvConfig(execution_lag=lag, episode_length=t - start + 5, random_start=False)
        base_env = SingleTickerTradingEnv(table, config)
        bumped_env = SingleTickerTradingEnv(bumped, config)
        obs_a, _ = base_env.reset(options={"start_index": start})
        obs_b, _ = bumped_env.reset(options={"start_index": start})
        while base_env.cursor < t:
            assert np.array_equal(obs_a, obs_b)
            obs_a, reward_a, *_ = base_env.step(np.int64(ACTION_BUY))
            obs_b, reward_b, *_ = bumped_env.step(np.int64(ACTION_BUY))
            assert reward_a == reward_b  # every step before t reads only bars <= t
        assert np.array_equal(obs_a, obs_b)  # the observation AT t never saw C_{t+1}
        obs_a, reward_a, _, _, info_a = base_env.step(np.int64(ACTION_BUY))
        obs_b, reward_b, _, _, _ = bumped_env.step(np.int64(ACTION_BUY))
        assert info_a["position"] == 1.0
        assert info_a["fill"] is None  # no cost in this step: the difference is the move alone
        assert reward_a != reward_b
        assert reward_b - reward_a == pytest.approx(100 * math.log(bump), abs=1e-9)
        # ... and C_{t+1} reaches the observation only once it is revealed (cursor t + 1).
        assert not np.array_equal(obs_a, obs_b)


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
@pytest.mark.parametrize("lag", _LAGS)
def test_reward_is_invariant_to_bars_after_t_plus_one(lag: int, factor: float) -> None:
    # Poison every bar after t+1: every reward up to AND INCLUDING the step from t (which reads
    # C_{t+1}, untouched) is bit-identical — the reward reads no bar beyond the one it reveals.
    for table in (_synthetic(_SYNTHETIC_SEEDS[1]), _spy()):
        for t in (MAX_LOOKBACK + 30, 260):
            for script in _SCRIPTS:
                run = _run_invariance_check(
                    k=t + 1, factor=factor, table=table, start=MAX_LOOKBACK, lag=lag, script=script
                )
                assert run.rewards_compared == t + 1 - MAX_LOOKBACK  # steps from cursors <= t


# --- the negative controls: leaky envs the detectors MUST catch -----------------------------------


class _LeakyEnv(SingleTickerTradingEnv):
    """§4.8 Test A': writes the NEXT close's return ``C_{t+1} / C_t`` into ``obs[0]`` — the
    textbook one-bar lookahead. It reads ``_closes`` directly, so it bypasses the audit."""

    def _observe(self) -> Obs:
        obs = super()._observe()
        obs[0] = np.float32(self._closes[self._cursor + 1] / self._closes[self._cursor])
        return obs


class _OffByOneWindowEnv(SingleTickerTradingEnv):
    """Builds every observation one bar ahead (``closes[: t + 2]``) — the off-by-one slice — by
    running the env's OWN audited observation path with the cursor advanced, so the audit records
    the future bar it reads."""

    def _observe(self) -> Obs:
        self._cursor += 1
        try:
            return super()._observe()
        finally:
            self._cursor -= 1


class _CentredMeanEnv(SingleTickerTradingEnv):
    """A centred rolling mean (``C_{t-5} .. C_{t+5}``), the classic smoothing leak."""

    def _observe(self) -> Obs:
        obs = super()._observe()
        t = self._cursor
        centred = self._closes[t - 5 : t + 6]
        obs[3] = np.float32(self._closes[t] / (sum(centred) / len(centred)) - 1.0)
        return obs


class _DatasetMeanEnv(SingleTickerTradingEnv):
    """A dataset-wide scaler: the close divided by the mean of EVERY close, future included."""

    def _observe(self) -> Obs:
        obs = super()._observe()
        obs[5] = np.float32(self._closes[self._cursor] / (sum(self._closes) / len(self._closes)))
        return obs


class _RewardPeekEnv(SingleTickerTradingEnv):
    """A reward that also reads ``C_{t+2}`` — one bar beyond the single bar a step may reveal."""

    def step(self, action: np.int64) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        t = self._cursor
        peek = math.log(self._closes[t + 2] / self._closes[t + 1])
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward + 1e-3 * peek, terminated, truncated, info


class _FarRatioEnv(SingleTickerTradingEnv):
    """Leaks a SCALE-FREE function of two future bars, ``C_{t+3} / C_{t+2}`` — the shape of every
    feature this env computes, and invisible to multiplicative poison wherever both bars lie past
    the horizon. Only the NaN probe (or a comparison earlier than ``k - 1``) can see it."""

    def _observe(self) -> Obs:
        obs = super()._observe()
        t = self._cursor
        obs[1] = np.float32(self._closes[t + 3] / self._closes[t + 2])
        return obs


class _InfoPeekEnv(SingleTickerTradingEnv):
    """Leaks ONLY through the info dict: ``drawdown`` nudged by the next close's return. The
    observation and reward stay clean, so only the detector's info comparison can see it (the
    learner never reads info, but SB3 ``Monitor`` logs and the demo's diagnostics do)."""

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        t = self._cursor
        info["drawdown"] += 1e-3 * math.log(self._closes[t + 1] / self._closes[t])
        return info


class _TinyLeakEnv(SingleTickerTradingEnv):
    """A leak far below any tolerance: ``obs[0] += 1e-6 * C_{t+1} / C_t``. A tolerance-based
    comparison (``np.allclose(rtol=1e-3)``) swallows it; only bit-exact comparison catches it."""

    def _observe(self) -> Obs:
        obs = super()._observe()
        t = self._cursor
        obs[0] += np.float32(1e-6 * self._closes[t + 1] / self._closes[t])
        return obs


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
def test_leakage_detector_actually_detects_leakage(factor: float) -> None:
    # A leakage test that cannot fail is decoration (program doc §1 item 1).
    with pytest.raises(AssertionError, match="observation at cursor 120 changed"):
        _run_invariance_check(_LeakyEnv, k=120, factor=factor)
    # ... and the honest env passes the identical check (same k, start, script, factor).
    _run_invariance_check(SingleTickerTradingEnv, k=120, factor=factor)


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
@pytest.mark.parametrize(
    "leaky_env",
    [
        _OffByOneWindowEnv,
        _CentredMeanEnv,
        _DatasetMeanEnv,
        _RewardPeekEnv,
        _InfoPeekEnv,
        _TinyLeakEnv,
        _FarRatioEnv,
    ],
    ids=lambda cls: cls.__name__,
)
def test_future_poisoning_catches_every_planted_leak(leaky_env: EnvClass, factor: float) -> None:
    with pytest.raises(AssertionError, match="when only bars after 120 were poisoned"):
        _run_invariance_check(leaky_env, k=120, factor=factor, script=_round_trips)


def test_the_audit_catches_an_observation_window_that_reaches_one_bar_into_the_future() -> None:
    for table in (_spy(), _synthetic(_SYNTHETIC_SEEDS[0])):
        env = _OffByOneWindowEnv(table, EnvConfig(episode_length=100), audit=True)
        with pytest.raises(AssertionError, match="after the decision time"):
            _run_audit_check(env, seed=3, script=_always_buy, start=100)
        honest = SingleTickerTradingEnv(table, EnvConfig(episode_length=100), audit=True)
        assert _run_audit_check(honest, seed=3, script=_always_buy, start=100)[0] == 101


@pytest.mark.parametrize(
    "leaky_env",
    [
        _LeakyEnv,
        _CentredMeanEnv,
        _DatasetMeanEnv,
        _RewardPeekEnv,
        _InfoPeekEnv,
        _TinyLeakEnv,
        _FarRatioEnv,
    ],
    ids=lambda cls: cls.__name__,
)
def test_the_audit_is_blind_to_reads_that_bypass_it_so_future_poisoning_is_the_backstop(
    leaky_env: EnvClass,
) -> None:
    # The coverage matrix, pinned rather than claimed: the self-reported audit only sees reads
    # made through the audited path, so these leaks pass it — and every one of them is caught by
    # the black-box poison detector above. Neither detector alone is the proof; the pair is.
    env = leaky_env(_synthetic(_SYNTHETIC_SEEDS[0]), EnvConfig(episode_length=100), audit=True)
    assert _run_audit_check(env, seed=3, script=_round_trips, start=100)[0] == 101
    with pytest.raises(AssertionError, match="when only bars after 120 were poisoned"):
        _run_invariance_check(leaky_env, k=120, factor=float("nan"), script=_round_trips)


def test_the_nan_probe_catches_a_scale_free_future_read_that_multiplicative_poison_cannot() -> None:
    # Compare ONLY the decision at k: _FarRatioEnv reads C_{k+3} / C_{k+2}, both poisoned by the
    # same factor, so the ratio is unchanged under x3 / x0.25 — but any touch of a NaN is fatal.
    for factor in (3.0, 0.25):
        _run_invariance_check(_FarRatioEnv, k=120, factor=factor, start=120)
    with pytest.raises(AssertionError, match="observation at cursor 120 changed"):
        _run_invariance_check(_FarRatioEnv, k=120, factor=float("nan"), start=120)
    # A runway of decisions before k closes the finite-factor gap too: at cursor k - 2 the ratio
    # straddles the boundary (C_{k+1} poisoned, C_k clean). Hence the default runway of 40.
    with pytest.raises(AssertionError, match="observation at cursor 118 changed"):
        _run_invariance_check(_FarRatioEnv, k=120, factor=3.0)


@pytest.mark.parametrize("factor", _FACTORS, ids=_FACTOR_IDS)
def test_a_leak_that_reaches_only_the_info_dict_is_caught(factor: float) -> None:
    # Observation and reward are clean (bit-identical up to k); the info dict alone carries the
    # future — so this control fails if the detector's info comparison is ever dropped.
    with pytest.raises(AssertionError, match="info at cursor 120 changed"):
        _run_invariance_check(_InfoPeekEnv, k=120, factor=factor)


@pytest.mark.parametrize("factor", [3.0, 0.25], ids=["x3", "x0.25"])
def test_a_leak_too_small_for_any_tolerance_is_caught_because_comparison_is_bit_exact(
    factor: float,
) -> None:
    # Pin that the planted leak really is tolerance-sized: at the decision k the poisoned twin's
    # observation is np.allclose(rtol=1e-3) to the clean one, yet not equal ...
    table, k = _synthetic(_SYNTHETIC_SEEDS[0]), 120
    config = EnvConfig(episode_length=10, random_start=False)
    obs_clean, _ = _TinyLeakEnv(table, config).reset(options={"start_index": k})
    dirty = _poisoned_env(_TinyLeakEnv, table, config, k=k, factor=factor)
    obs_dirty, _ = dirty.reset(options={"start_index": k})
    assert np.allclose(obs_clean, obs_dirty, rtol=1e-3)
    assert not np.array_equal(obs_clean, obs_dirty)
    # ... so only a BIT-EXACT detector catches it (a tolerance would make this control pass).
    with pytest.raises(AssertionError, match="observation at cursor 120 changed"):
        _run_invariance_check(_TinyLeakEnv, k=k, factor=factor, table=table)


# --- env-level scale invariance (the composition of the verified level-free pieces) --------------


def _rollout(
    table: BarTable, *, lag: int, script: Script, start: int, length: int
) -> tuple[list[Obs], list[float], list[dict[str, Any]]]:
    env = SingleTickerTradingEnv(
        table, EnvConfig(execution_lag=lag, episode_length=length, random_start=False)
    )
    obs, info = env.reset(options={"start_index": start})
    observations, rewards, infos = [obs], [], [info]
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.int64(script(env.cursor, info)))
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
    assert len(rewards) == length
    return observations, rewards, infos


_SCALE_SCRIPTS: tuple[Script, ...] = (
    lambda _cursor, _info: ACTION_HOLD,  # flat throughout
    _always_buy,  # held throughout (unrealized P&L vs the entry bar)
    _round_trips,
)


@pytest.mark.parametrize("multiplier", [2.0, 0.5, 1024.0, 2.0**-10])
def test_multiplying_every_close_by_a_power_of_two_changes_no_observation_or_reward_by_one_bit(
    multiplier: float,
) -> None:
    # A power of two scales every close exactly, so a level-free env must be BIT-identical: any
    # read of a price LEVEL (§4.2 BINDING: adjustment=all levels encode later dividends) shows up.
    for table, start in ((_spy(), 1200), (_synthetic(_SYNTHETIC_SEEDS[0]), MAX_LOOKBACK)):
        scaled = with_closes(table, [c * multiplier for c in table.closes()])
        for lag in (0, 1):
            for script in _SCALE_SCRIPTS:
                base = _rollout(table, lag=lag, script=script, start=start, length=252)
                moved = _rollout(scaled, lag=lag, script=script, start=start, length=252)
                for obs_a, obs_b in zip(base[0], moved[0], strict=True):
                    assert np.array_equal(obs_a, obs_b)
                assert base[1] == moved[1]
                assert base[2] == moved[2]


@pytest.mark.parametrize("multiplier", [3.7, 1 / 7, 1000.0, 0.013])
def test_multiplying_every_close_by_any_constant_leaves_observations_and_rewards_unchanged(
    multiplier: float,
) -> None:
    # An arbitrary constant perturbs each close by <= 1 ulp of rounding, so equality is up to
    # float noise — orders of magnitude below any feature or reward resolution.
    for table, start in ((_spy(), 300), (_synthetic(_SYNTHETIC_SEEDS[1]), 100)):
        scaled = with_closes(table, [c * multiplier for c in table.closes()])
        for lag in (0, 1):
            for script in _SCALE_SCRIPTS:
                base = _rollout(table, lag=lag, script=script, start=start, length=252)
                moved = _rollout(scaled, lag=lag, script=script, start=start, length=252)
                np.testing.assert_allclose(np.stack(moved[0]), np.stack(base[0]), atol=1e-5)
                np.testing.assert_allclose(moved[1], base[1], rtol=1e-9, atol=1e-9)
                for info_a, info_b in zip(base[2], moved[2], strict=True):
                    assert info_b["equity"] == pytest.approx(info_a["equity"], rel=1e-12)
                    assert info_b["entry_index"] == info_a["entry_index"]
                    assert info_b["n_trades"] == info_a["n_trades"]


# --- Test E and the normalization guard: grep-style architectural assertions --------------------


def test_env_module_never_calls_price_at() -> None:
    # domain/prices.py:23-24 — PriceSeries.price_at returns the EARLIEST point when asked for a
    # time before the series: a silent left-edge forward-look. The env must address bars by
    # integer index only; this stops a future "simplification" back into that bug.
    for source in (_ENV_SOURCE, _FEATURES_SOURCE):
        assert "price_at" not in source.read_text(), f"{source.name} must never call price_at"


_FITTED_NORMALIZATION = (
    "StandardScaler",
    "MinMaxScaler",
    "VecNormalize",
    "RunningMeanStd",
    "fit_transform",
    ".mean(",
    ".std(",
    ".var(",
    "np.mean",
    "np.std",
    "nanmean",
    "fmean",
    "import statistics",
    "from statistics",
)


def test_no_fitted_normalization_statistics() -> None:
    # The observation is normalized by FIXED a-priori constants (FeatureScales). A statistic fitted
    # over a full array (a scaler, a running-mean wrapper, .mean()/.std() of the dataset) would leak
    # the future's mean and variance into every observation — and the SAME wrapper would have to be
    # shipped bit-exactly to inference, the classic train/serve skew.
    for source in (_ENV_SOURCE, _FEATURES_SOURCE):
        text = source.read_text()
        found = [token for token in _FITTED_NORMALIZATION if token in text]
        assert not found, (
            f"{source.name} mentions {found}: normalization must use fixed FeatureScales "
            "constants, never a statistic fitted over a full array"
        )
