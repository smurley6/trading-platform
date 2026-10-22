"""The single-ticker trading environment (task-rl-05; docs/RL-SLEEVE-PROGRAM.md §4.3-§4.8).

HEAVY half of ``trading.rl``: the only module of this task importing ``gymnasium`` / ``numpy``. It
is imported explicitly as ``trading.rl.env`` (``trading.rl.__init__`` stays inert) and never reaches
the app import graph (import-linter "Heavy ML deps never enter the app import graph"). It composes
the verified PURE pieces — ``rl/bars.py`` (PIT ``BarTable``), ``rl/costs.py`` (bps ``CostModel``),
``rl/rewards.py`` (``net_of_cost_growth`` / ``log_return_reward`` / ``DifferentialSharpe``) and
``domain/rl_features.py`` (``build_observation``) — and adds only the timing contract below.

**Behavior contract** (program doc §4.7, verbatim):

    At step *t* the agent stands just after the close of session *t*. Its observation is built ONLY
    from ``closes[: t+1]``, every one of which has ``known_at <= bars[t].known_at``. The action is
    filled at the close of session *t+1*, and the reward is the portfolio log return over
    *t -> t+1*. ``bars[t+1]`` is read **exactly once**, in ``step``, to score the action; it never
    reaches ``_observe``.

**The timing contract, stated precisely** (resolves the §4.7 flag raised by the task-rl-03 PIT
review; the ONE place in code that owns the bar pairing is :func:`fill_index_for`). With
``L = EnvConfig.execution_lag`` (default 0) and the decision bar *t* (``cursor``):

1. **Decision time** is ``bars[t].known_at`` (``decision_at``): bar *t*'s close is public. The
   observation reads ``closes[: t+1]`` plus the position state, nothing else.
2. **Fill time.** An accepted BUY/SELL decided at *t* fills at the close of bar
   ``f = fill_index_for(t, L) = t + 1 + L`` — the first close NOT yet known at decision time (lag
   0), or ``L`` sessions later. ``C_f`` is never known at decision time: ``known_at`` is strictly
   ascending, so ``bars[f].known_at > bars[t].known_at``. A fill at ``C_t`` (decide-at-close /
   fill-at-the-same-close) would be a lookahead fill and is impossible by construction.
3. **Accounting period of a step.** ``step`` at cursor *t* reveals bar *t+1* (the single read of
   ``bars[t+1]``) and books, in this order:
   (a) the MARKET MOVE ``C_t -> C_{t+1}`` earned by the position held over that period — the
   position after every fill at closes ``<= t`` (``info["position"]`` before the step, decided at
   ``t - 1 - L`` or earlier, NEVER the action just chosen) — via
   ``net_of_cost_growth(C_t, C_{t+1}, position, position)``; then
   (b) any FILL at close *t+1* (the order decided at ``t - L``), charged
   ``CostModel.cost_fraction(position_before, position_after)`` of the equity marked at
   ``C_{t+1}``. ``growth = move * (1 - cost_fraction)``; ``reward = 100 * ln(max(growth, 1e-9))``.
   So "the portfolio log return over t -> t+1" is the return of the position ALREADY HELD; §4.5's
   ``position_t`` is that held position, not the position requested at *t*.
4. **First earning period.** A position filled at ``C_f`` earns ``C_f -> C_{f+1}`` (booked by the
   step at cursor *f*) and every period after it while held — never any move before its fill.
5. **Position state** (the ``rl_features`` binding contract): ``held`` / ``info["position"]`` at
   cursor *t* reflect ONLY fills at closes ``<= t``; an order still pending (``L > 0``) is not held.
   ``entry_close`` is ``C_f`` of the ACTUAL fill (``f <= t``, already public at decision time),
   never the decision bar's close; ``held`` is True iff an entry exists, so the env never passes
   ``held=True`` with ``entry_close=None``. State and reward therefore agree on the fill price: at
   zero cost ``C_t / entry_close == exp(sum of rewards since the fill / 100)``.

Consequences pinned in ``tests/unit/rl/test_env.py``: an action's MARKET return is first credited
``1 + L`` steps after it is chosen (its cost lands in step ``t + L``); the reward of step *t* reads
only ``C_t`` and ``C_{t+1}`` (invariant to every later bar), and depends on ``C_{t+1}`` only when a
position is held over ``t -> t+1``; always-BUY at zero cost reproduces buy-and-hold from the FIRST
FILL close, ``exp(sum(rewards) / 100) == C_end / C_{start + 1 + L}`` with
``end = start + episode_length`` (never ``C_end / C_start`` — that ratio is what a lookahead fill
would earn). At ``L = 0`` the reward is a function of ``(s_t, a_t, s_{t+1})`` (Markov); at
``L > 0`` a pending order is hidden state by design (``execution_lag`` is a robustness probe:
report lag 0 AND lag 1 — if the edge vanishes at lag 1, there was no edge).

**Episode structure** (§4.7). ``reset`` starts flat, with no pending order and ``equity = 1.0``
at decision index ``start`` in ``[lo, hi]``: ``lo = MAX_LOOKBACK`` (``WARMUP_BARS - 1 = 49``, the
first index with a full 50-close window; the lag needs nothing earlier, because the episode starts
flat) and ``hi = len(bars) - 1 - episode_length`` (every one of the episode's reveals exists). An
episode's bar footprint is ``[start - MAX_LOOKBACK, start + episode_length]`` — the walk-forward
purge must cover it. ``terminated`` is RUIN only (``equity <= ruin_fraction``, the one absorbing
state); ``truncated`` is the step budget / end of data, and is False whenever ``terminated`` is True
(PPO bootstraps ``V(s')`` on truncation, not termination — swapping them biases the value
function). Orders still pending at the end of an episode are discarded unfilled.

**Ruin is detected from EQUITY, never from the reward.** ``equity *= growth`` every step; a growth
``<= 0`` drives equity ``<= 0 <= ruin_fraction`` and terminates. ``log_return_reward``'s ``1e-9``
growth floor is a numeric guard that keeps the reward finite, NOT a ruin signal — a floored reward
must never be read as (or instead of) the equity test.

**Scale-free account** (§4.6). All-in / flat: BUY moves 100% of env equity into the symbol,
positions are in ``{0.0, 1.0}`` (long-only, §4.4), equity starts at 1.0 and is a pure ratio. Costs
are flat bps of traded notional (default ``DEFAULT_COST``, 1.0 bp one-way — never zero by default),
charged inside the equity, once per fill. Real sizing happens in the live sleeve
(``position_size``), so **live realized P&L will not equal env reward**. Closes are consumed ONLY
as ratios (§4.2 BINDING rule: ``adjustment=all`` levels are back-adjusted); ``info`` exposes bar
INDICES, never a price level.

**Actions** (§4.4): ``Discrete(3)``, ``0 = HOLD`` (inert default), ``1 = BUY``, ``2 = SELL``.
Legality is judged against the COMMITTED position (held, or the target of the pending order
when ``L > 0``): BUY while committed long or SELL while committed flat resolves to HOLD with
``info["action_was_noop"] = True`` and an optional ``invalid_action_penalty`` (default 0.0; a
learning lever). The principled upgrade is action masking via ``sb3-contrib``'s ``MaskablePPO``;
that dependency is deliberately not added.

**Observation**: ``Box(-10, 10, (9,), float32)`` from the shared ``build_observation`` (fixed
a-priori ``FeatureScales``; no fitted statistic and no running-statistics normalizing wrapper —
normalization lives inside the same function training and production compute). ``reward="dsr"``
(experiment #2) uses Moody & Saffell's differential Sharpe on the per-step net return
``growth - 1``; its hidden ``(A, B)`` EMA state is exposed as ``info["dsr_state"]`` but is NOT
appended to the observation (fixed at the 9 shared features), so the DSR variant is not Markov — a
known limitation of the experiment, not of the default reward.

**Info dict** (one builder, ``_info``; every value is env state, nothing recomputed elsewhere):

- ``start_index``, ``cursor``; ``decision_at`` = ``bars[cursor].known_at`` (also the instant
  ``equity`` is marked at); ``session_ts`` (a label, never knowledge time).
- ``equity`` (starts at 1.0), ``drawdown`` (``1 - equity / peak``), ``position`` (after fills at
  closes ``<= cursor``), ``entry_index`` (bar index of the open position's fill, or None),
  ``pending_orders``.
- Episode-cumulative: ``turnover`` (sum of ``|delta position|``), ``cost_paid`` (in units of the
  starting equity), ``n_trades`` (fills).
- ``step`` adds ``action_requested``, ``action_effective``, ``action_was_noop`` and ``fill``: None,
  or ``{"decision_index", "fill_index", "side", "position_before", "position_after",
  "cost_fraction"}`` for the fill at the bar just revealed.
- ``dsr_state`` = ``(A, B)`` when ``reward="dsr"``.

Never looks a close up by TIME through ``PriceSeries`` (its point lookup in ``domain/prices.py``
returns the EARLIEST point for a time before the series — a silent left-edge forward-look; rl-06
greps this file for that method's name) and never treats ``session_ts`` as knowledge time: bars are
addressed by integer index only.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from trading.domain.rl_features import (
    FEATURE_NAMES,
    MAX_LOOKBACK,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    WARMUP_BARS,
    FeatureScales,
    FeatureSpec,
    build_observation,
)
from trading.rl.bars import Bar, BarTable
from trading.rl.costs import DEFAULT_COST, CostModel
from trading.rl.rewards import DifferentialSharpe, log_return_reward, net_of_cost_growth

ACTION_HOLD: Final = 0
ACTION_BUY: Final = 1
ACTION_SELL: Final = 2
N_ACTIONS: Final = 3

_FLAT: Final = 0.0
_LONG: Final = 1.0
_REWARDS: Final = frozenset({"log_return", "dsr"})
_RESET_OPTIONS: Final = frozenset({"start_index"})

DEFAULT_FEATURE_SPEC: Final = FeatureSpec(
    window=WARMUP_BARS, names=FEATURE_NAMES, scales=FeatureScales()
)


class EnvConfigError(ValueError):
    """The env cannot be built or reset as configured (fails closed, never guesses)."""


def fill_index_for(decision_index: int, *, execution_lag: int) -> int:
    """THE bar pairing (program doc §4.7): the bar at whose close an order decided at
    ``decision_index`` fills — ``decision_index + 1 + execution_lag``.

    Lag 0 fills at the first close NOT known at decision time (``C_{t+1}``); each unit of lag
    delays the fill one more session. Never ``decision_index`` itself: a fill at the decision bar's
    close is a lookahead fill (that close is already public when the agent decides). Every fill in
    the env goes through this function.
    """
    if execution_lag < 0:
        raise EnvConfigError(f"execution_lag must be >= 0, got {execution_lag}")
    return decision_index + 1 + execution_lag


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Everything that parameterizes the env. Defaults are the training defaults (§4.7/§4.9)."""

    symbol: str = "SPY"
    episode_length: int = 252
    cost: CostModel = DEFAULT_COST
    reward: Literal["log_return", "dsr"] = "log_return"
    # Sessions of delay ON TOP OF the mandatory one-session lag (0 = fill at C_{t+1}).
    execution_lag: int = 0
    invalid_action_penalty: float = 0.0
    ruin_fraction: float = 0.2
    random_start: bool = True
    spec: FeatureSpec = field(default=DEFAULT_FEATURE_SPEC)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise EnvConfigError("symbol must be non-empty")
        for name in ("episode_length", "execution_lag"):
            value: object = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise EnvConfigError(f"{name} must be an int, got {value!r}")
        if self.episode_length < 1:
            raise EnvConfigError(f"episode_length must be >= 1, got {self.episode_length}")
        if not 0 <= self.execution_lag < self.episode_length:
            raise EnvConfigError(
                f"execution_lag must be in [0, episode_length) so an order can fill inside an "
                f"episode, got {self.execution_lag}"
            )
        if not isinstance(self.cost, CostModel):
            raise EnvConfigError(f"cost must be a CostModel, got {self.cost!r}")
        if self.reward not in _REWARDS:
            raise EnvConfigError(f"reward must be one of {sorted(_REWARDS)}, got {self.reward!r}")
        penalty = self.invalid_action_penalty
        if not math.isfinite(penalty) or penalty < 0:
            raise EnvConfigError(f"invalid_action_penalty must be finite and >= 0, got {penalty}")
        if not math.isfinite(self.ruin_fraction) or not 0.0 <= self.ruin_fraction < 1.0:
            raise EnvConfigError(f"ruin_fraction must be in [0, 1), got {self.ruin_fraction}")
        if not isinstance(self.spec, FeatureSpec):
            raise EnvConfigError(f"spec must be a FeatureSpec, got {self.spec!r}")


@dataclass(frozen=True, slots=True)
class _Order:
    """An accepted order awaiting its fill close (``fill_index``)."""

    decision_index: int
    fill_index: int
    target: float


class SingleTickerTradingEnv(gym.Env[npt.NDArray[np.float32], np.int64]):
    """All-in / flat single-ticker env over a PIT ``BarTable``. See the module docstring for the
    timing contract, the episode structure and the info schema."""

    def __init__(
        self, bars: BarTable, config: EnvConfig | None = None, *, audit: bool = False
    ) -> None:
        # Per-instance (gymnasium declares ``metadata`` as an instance variable, so a ClassVar
        # override fails mypy; the base class's default is one dict shared by every Env).
        self.metadata = {"render_modes": []}
        self._config = config if config is not None else EnvConfig()
        if bars.symbol != self._config.symbol:
            raise EnvConfigError(
                f"bars are for {bars.symbol!r} but the config is for {self._config.symbol!r}"
            )
        self._bars = bars
        # The ONE float view of the closes; both _observe and step read closes only from here.
        self._closes: tuple[float, ...] = bars.closes()
        self._lo = MAX_LOOKBACK
        self._hi = len(bars) - 1 - self._config.episode_length
        if self._hi < self._lo:
            raise EnvConfigError(
                f"{len(bars)} bars cannot hold a {self._config.episode_length}-step episode after "
                f"the {WARMUP_BARS}-bar warmup (need >= "
                f"{MAX_LOOKBACK + self._config.episode_length + 1})"
            )
        self._audit = audit
        self._audit_reads: list[Bar] = []

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, shape=(OBS_DIM,), dtype=np.float32
        )

        self._start = self._lo
        self._cursor = self._lo
        self._steps = 0
        self._done = True  # step() before the first reset() is an error
        self._equity = 1.0
        self._peak_equity = 1.0
        self._position = _FLAT
        self._entry_index: int | None = None
        self._pending: list[_Order] = []
        self._turnover = 0.0
        self._cost_paid = 0.0
        self._n_trades = 0
        self._dsr = DifferentialSharpe()

    @property
    def config(self) -> EnvConfig:
        return self._config

    @property
    def start_bounds(self) -> tuple[int, int]:
        """Inclusive ``(lo, hi)`` range of legal episode start (decision) indices."""
        return self._lo, self._hi

    @property
    def decision_at(self) -> datetime:
        """When the agent decides at the current cursor: the moment bar ``cursor`` became public."""
        return self._bars.bars[self._cursor].known_at

    @property
    def cursor(self) -> int:
        return self._cursor

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)  # FIRST: seeds self.np_random (never the `random` module)
        self._done = True  # a reset that fails below leaves an env that cannot be stepped
        start = self._choose_start(options)
        self._start = start
        self._cursor = start
        self._steps = 0
        self._done = False
        self._equity = 1.0
        self._peak_equity = 1.0
        self._position = _FLAT
        self._entry_index = None
        self._pending = []
        self._turnover = 0.0
        self._cost_paid = 0.0
        self._n_trades = 0
        self._dsr.reset()
        return self._observe(), self._info()

    def step(
        self, action: np.int64
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() on a finished (or never reset) episode; call reset() first")
        requested = _as_action(action)
        effective, was_noop = self._resolve(requested)
        decided_at = self._cursor
        if effective != ACTION_HOLD:
            self._pending.append(
                _Order(
                    decision_index=decided_at,
                    fill_index=fill_index_for(decided_at, execution_lag=self._config.execution_lag),
                    target=_LONG if effective == ACTION_BUY else _FLAT,
                )
            )

        revealed = decided_at + 1
        close_prev = self._closes[decided_at]
        close_now = self._closes[revealed]  # the ONE read of bar t+1; never reaches _observe

        # (a) market move t -> t+1, earned by the position already held (fills at closes <= t).
        move = net_of_cost_growth(
            close_prev=close_prev,
            close_now=close_now,
            position_before=self._position,
            position_after=self._position,
            cost=self._config.cost,
        )
        # (b) the fill (if any) at close t+1, charged on the equity marked at C_{t+1}.
        fill = self._fill_at(revealed, marked_equity=self._equity * move)
        growth = move * (1.0 - (0.0 if fill is None else float(fill["cost_fraction"])))

        if self._config.reward == "dsr":
            reward = self._dsr.update(growth - 1.0)
        else:
            reward = log_return_reward(equity_prev=1.0, equity_now=growth)
        if was_noop:
            reward -= self._config.invalid_action_penalty

        self._equity *= growth
        self._peak_equity = max(self._peak_equity, self._equity)
        self._cursor = revealed
        self._steps += 1

        # Ruin from EQUITY (growth <= 0 => equity <= 0 <= ruin_fraction); never from the reward.
        terminated = self._equity <= self._config.ruin_fraction
        truncated = not terminated and (
            self._steps >= self._config.episode_length or self._cursor >= len(self._closes) - 1
        )
        self._done = terminated or truncated

        info = self._info()
        info.update(
            action_requested=requested,
            action_effective=effective,
            action_was_noop=was_noop,
            fill=fill,
        )
        return self._observe(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    # --- internals ----------------------------------------------------------------------------

    def _choose_start(self, options: dict[str, Any] | None) -> int:
        if options:
            unknown = set(options) - _RESET_OPTIONS
            if unknown:
                raise EnvConfigError(f"unknown reset options {sorted(unknown)}")
        if options and "start_index" in options:
            raw: object = options["start_index"]
            if isinstance(raw, bool) or not isinstance(raw, int | np.integer):
                raise EnvConfigError(f"start_index must be an int, got {raw!r}")
            start = int(raw)
            if not self._lo <= start <= self._hi:
                raise EnvConfigError(
                    f"start_index {start} outside the legal range [{self._lo}, {self._hi}]"
                )
            return start
        if self._config.random_start:
            return int(self.np_random.integers(self._lo, self._hi + 1))
        return self._lo

    def _committed_position(self) -> float:
        """The position once every accepted order has filled (== held when nothing is pending)."""
        return self._pending[-1].target if self._pending else self._position

    def _resolve(self, requested: int) -> tuple[int, bool]:
        """``(effective action, was it an illegal no-op)``; illegal BUY/SELL resolve to HOLD."""
        if requested == ACTION_HOLD:
            return ACTION_HOLD, False
        committed = self._committed_position()
        if (requested == ACTION_BUY and committed == _FLAT) or (
            requested == ACTION_SELL and committed == _LONG
        ):
            return requested, False
        return ACTION_HOLD, True

    def _fill_at(self, index: int, *, marked_equity: float) -> dict[str, Any] | None:
        """Execute the order whose fill close is bar ``index`` (at most one), updating position,
        entry and the cumulative counters. Returns the fill event, or None."""
        if not self._pending or self._pending[0].fill_index != index:
            return None
        order = self._pending.pop(0)
        before = self._position
        if order.target == before:  # unreachable: legality alternates the committed position
            raise RuntimeError(f"order {order} does not change the position {before}")
        cost_fraction = self._config.cost.cost_fraction(
            position_before=before, position_after=order.target
        )
        self._position = order.target
        self._entry_index = index if order.target == _LONG else None
        self._turnover += abs(order.target - before)
        self._cost_paid += marked_equity * cost_fraction
        self._n_trades += 1
        return {
            "decision_index": order.decision_index,
            "fill_index": index,
            "side": "buy" if order.target == _LONG else "sell",
            "position_before": before,
            "position_after": order.target,
            "cost_fraction": cost_fraction,
        }

    def _observe(self) -> npt.NDArray[np.float32]:
        t = self._cursor
        held = self._position == _LONG
        entry = self._entry_index
        if held != (entry is not None) or (entry is not None and entry > t):
            # The rl_features contract: never held=True with entry_close=None, and the entry is a
            # fill close already public at decision time.
            raise RuntimeError(f"inconsistent position state: held={held}, entry_index={entry}")
        window = self._closes[: t + 1]  # a SLICE ending at the decision bar
        entry_close = None if entry is None else self._closes[entry]
        if self._audit:
            # Exactly what the observation reads: build_observation's trailing spec.window closes
            # (it ignores anything earlier) plus the entry bar, which may lie before that window.
            reads = list(self._bars.bars[t + 1 - self._config.spec.window : t + 1])
            if entry is not None:
                reads.append(self._bars.bars[entry])
            self._audit_reads = reads
        values = build_observation(
            window, held=held, entry_close=entry_close, spec=self._config.spec
        )
        clipped = np.clip(np.asarray(values, dtype=np.float64), OBS_LOW, OBS_HIGH)
        return clipped.astype(np.float32, copy=False)

    def _info(self) -> dict[str, Any]:
        bar = self._bars.bars[self._cursor]
        info: dict[str, Any] = {
            "start_index": self._start,
            "cursor": self._cursor,
            "decision_at": bar.known_at,
            "session_ts": bar.session_ts,
            "equity": self._equity,
            "position": self._position,
            "entry_index": self._entry_index,
            "pending_orders": len(self._pending),
            "turnover": self._turnover,
            "cost_paid": self._cost_paid,
            "n_trades": self._n_trades,
            "drawdown": 1.0 - self._equity / self._peak_equity,
        }
        if self._config.reward == "dsr":
            info["dsr_state"] = self._dsr.observation()
        return info


def _as_action(action: object) -> int:
    """Accept an int-like scalar (Python int, ``np.int64``, a size-1 integer array); reject
    anything else (bools, floats, out-of-range) loudly rather than coercing it."""
    if isinstance(action, bool):
        raise ValueError(f"action must be an integer in [0, {N_ACTIONS}), got {action!r}")
    array = np.asarray(action)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"action must be an integer in [0, {N_ACTIONS}), got {action!r}")
    value = int(array.reshape(-1)[0])
    if not 0 <= value < N_ACTIONS:
        raise ValueError(f"action must be an integer in [0, {N_ACTIONS}), got {value}")
    return value
