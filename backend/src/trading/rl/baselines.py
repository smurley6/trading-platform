"""The four baseline policies the agent is measured against (task-rl-07; program doc §5.2).

PURE half of ``trading.rl``: stdlib + ``trading.domain`` only — no numpy, no gymnasium, no torch —
so the baselines import (and are unit-tested) with zero optional deps.

**Retarget override (2026-10-21).** Program doc §5.2 and the task-rl-07 spec describe the baselines
as platform ``Strategy`` implementations (``strategies/rl_baselines.py``, ``@register("rl_*")``) for
the ``Backtest`` replay path. The 2026-10-21 retarget DELETED the platform sleeve wiring
(task-rl-11, M8-M11), so that path has no consumer. The baselines are instead plain POLICY
CALLABLES for :class:`~trading.rl.env.SingleTickerTradingEnv`'s ``Discrete(3)`` action space, run by
``rl/scoring.py`` through the IDENTICAL env replay (same bars, same timing contract, same fills,
same costs) the agent's exported policy will be. Nothing here imports ``trading.strategies``.

**The policy contract.** A :data:`PolicyFactory` maps a seed to a FRESH per-episode
:data:`Policy`; a policy is called once per decision with

- ``observation`` — the env's 9-float observation at the decision bar (``domain/rl_features``), and
- ``history`` — the trailing ``WARMUP_BARS`` (50) closes ENDING AT the decision bar
  (``closes[cursor - 49 : cursor + 1]``): exactly the window the env built that observation from
  (``SingleTickerTradingEnv._observe`` reads ``closes[: cursor + 1]``; ``build_observation`` reads
  only its last 50). The future is physically absent from both arguments.

and returns an action: :data:`HOLD` (0), :data:`BUY` (1) or :data:`SELL` (2) — the env's
``ACTION_*`` values, mirrored here so this module never loads gymnasium (a test pins the equality).
A trained agent ignores ``history`` (its policy is a function of the observation alone).

Every baseline emits ONLY legal actions: it tracks its own COMMITTED position (the env starts each
episode flat; every BUY/SELL it emits flips the committed position and is accepted, because it is
legal against exactly that position — env ``_resolve``), so a baseline never produces an
``action_was_noop`` and its requested and effective actions coincide.

- :func:`buy_and_hold` — BUY at the first decision (always legal: episodes start flat), then HOLD.
- :func:`always_flat` — HOLD forever: the zero-cost / zero-turnover floor.
- :func:`random_policy` — each decision, with probability ``trade_probability`` emit the legal flip
  (BUY when flat, SELL when long), else HOLD; ``random.Random(seed)``.
- :func:`sma_cross` — target long iff ``SMA_fast > SMA_slow`` over ``history`` (default 20/50),
  else flat.

**Random = §5.2's uniform policy, with its trade propensity exposed.** Uniform over {BUY, HOLD,
SELL} against the env's legality rule flips the committed position with probability exactly 1/3 per
decision (one of the three actions is the legal flip; the illegal one resolves to HOLD), so the
default ``trade_probability = 1/3`` (:data:`UNIFORM_TRADE_PROBABILITY`) has the same distribution of
EFFECTIVE actions and position paths as the uniform policy — minus its no-ops. rl-10 sets
``trade_probability`` from the agent (:func:`trading.rl.metrics.matched_trade_probability`) to
TURNOVER-MATCH the null for the T2 gate.

**SMA cross is point-in-time by construction.** The SMA(20, 50) rule cannot be read off the
observation (it carries ``SMA_10 / SMA_50``, the ``trend`` feature, not ``SMA_20``), so the policy
reads ``history`` — the same 50 closes, ending at the same decision bar, the agent's observation
was built from (the scoring tests rebuild every observation from ``history`` and match the env's;
a future-poisoning test pins its actions invariant to every later bar). ``SMA_fast > SMA_slow`` is
homogeneous of degree zero in the closes (dividing both sums by ``C_t`` leaves only ratios), so the
§4.2 returns-only rule holds: back-adjusted LEVELS never matter, only ratios. Its ``SMA_n`` is the
left-to-right float sum divided by ``n`` (the ``rl_features`` arithmetic).
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from trading.domain.rl_features import WARMUP_BARS

HOLD: Final = 0
BUY: Final = 1
SELL: Final = 2

# Uniform {BUY, HOLD, SELL} flips the committed position with probability 1/3 per decision.
UNIFORM_TRADE_PROBABILITY: Final = 1.0 / 3.0
SMA_FAST: Final = 20
SMA_SLOW: Final = 50

# (observation, history) -> action. ``history`` = the trailing WARMUP_BARS closes ending at the
# decision bar (see the module docstring); the agent's policy ignores it.
Policy = Callable[[Sequence[float], Sequence[float]], int]
# seed -> a fresh per-episode policy (deterministic policies ignore the seed).
PolicyFactory = Callable[[int], Policy]


class _Committed:
    """The baseline's own view of its committed position. Starts flat (every env episode does)
    and flips on every trade the baseline emits — each of which is legal, hence accepted."""

    __slots__ = ("long",)

    def __init__(self) -> None:
        self.long = False

    def to(self, want_long: bool) -> int:
        """The legal action that moves the committed position to ``want_long`` (or HOLD)."""
        if want_long == self.long:
            return HOLD
        self.long = want_long
        return BUY if want_long else SELL


def always_flat(seed: int) -> Policy:
    """Never trades: equity stays exactly 1.0 at every cost. Anchors the cost / turnover floor
    and is the reference ``break_even_cost_bps`` measures an edge against."""
    del seed

    def act(observation: Sequence[float], history: Sequence[float]) -> int:
        del observation, history
        return HOLD

    return act


def buy_and_hold(seed: int) -> Policy:
    """BUY at the first decision of the episode, then HOLD to the end (``n_trades == 1``). Under
    the env's timing contract the fill is at ``C_{start + 1 + lag}``, so it earns
    ``C_end / C_{start + 1 + lag}`` before costs — compare :func:`trading.rl.metrics.
    buy_and_hold_reference` (the frictionless, lag-free window return) for the execution drag."""
    del seed
    committed = _Committed()

    def act(observation: Sequence[float], history: Sequence[float]) -> int:
        del observation, history
        return committed.to(True)

    return act


def random_policy(
    trade_probability: float = UNIFORM_TRADE_PROBABILITY,
) -> PolicyFactory:
    """The seeded random baseline (the NULL distribution — run it over many seeds, never one).

    Each decision draws ``u = rng.random()`` from a per-episode ``random.Random(seed)`` (never the
    global RNG); ``u < trade_probability`` emits the legal flip, otherwise HOLD. Exactly one draw
    per decision, so a seed fixes the whole action path. ``trade_probability`` in ``[0, 1]``:
    ``0`` is always-flat, ``1`` flips every decision, the default ``1/3`` is §5.2's uniform policy.
    """
    if (
        isinstance(trade_probability, bool)
        or not isinstance(trade_probability, int | float)
        or not math.isfinite(trade_probability)
        or not 0.0 <= trade_probability <= 1.0
    ):
        raise ValueError(f"trade_probability must be in [0, 1], got {trade_probability!r}")
    p = float(trade_probability)

    def factory(seed: int) -> Policy:
        rng = random.Random(seed)
        committed = _Committed()

        def act(observation: Sequence[float], history: Sequence[float]) -> int:
            del observation, history
            if rng.random() < p:
                return committed.to(not committed.long)
            return HOLD

        return act

    return factory


def _sma(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def sma_cross(fast: int = SMA_FAST, slow: int = SMA_SLOW) -> PolicyFactory:
    """Long when ``SMA_fast > SMA_slow`` (strict) over ``history``, else flat; trades only on a
    change of that target. ``0 < fast < slow <= WARMUP_BARS`` (``history`` holds 50 closes)."""
    for name, value in (("fast", fast), ("slow", slow)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {value!r}")
    if not 0 < fast < slow <= WARMUP_BARS:
        raise ValueError(f"need 0 < fast < slow <= {WARMUP_BARS}, got fast={fast}, slow={slow}")

    def factory(seed: int) -> Policy:
        del seed
        committed = _Committed()

        def act(observation: Sequence[float], history: Sequence[float]) -> int:
            del observation
            if len(history) < slow:
                raise ValueError(f"history must hold at least {slow} closes, got {len(history)}")
            return committed.to(_sma(history[-fast:]) > _sma(history[-slow:]))

        return act

    return factory


# The four §5.2 baselines at their defaults, by report name.
BASELINES: Final[Mapping[str, PolicyFactory]] = {
    "buy_and_hold": buy_and_hold,
    "always_flat": always_flat,
    "random": random_policy(),
    "sma_cross_20_50": sma_cross(),
}
