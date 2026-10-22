"""Reward functions for the RL environment (task-rl-03; docs/RL-SLEEVE-PROGRAM.md §4.5).

PURE half of ``trading.rl``: stdlib only — no numpy, no gymnasium, no torch.

**The reward is the log return of NET-OF-COST equity, x100.** One accounting period (the env picks
which bars bound it; its timing contract and ``execution_lag`` live in ``rl/env.py``)::

    growth = 1 + position_after * (C_now / C_prev - 1) - cost.cost_fraction(position_before,
                                                                            position_after)
    reward = 100 * ln(max(growth, 1e-9))

Arithmetically, ``position_after`` is the position held over the period bounded by
``(C_prev, C_now)`` and earns ``C_now / C_prev - 1``; the cost of the change
``position_before -> position_after`` is charged ONCE, on the traded notional
``|position_after - position_before|`` x equity, at the same one-way rate for an entry (BUY) and an
exit (SELL). This module makes NO timing claim: which bars bound a period, at which close a fill
happens, and where its cost is booked are decided SOLELY by the env — see ``rl/env.py`` and the
RESOLVED timing contract in program doc §4.7 (task-rl-05: a fill happens at a close never known at
decision time, and the env books it at the END of the step that reveals that close, applying
``cost_fraction`` to the equity marked there). Do not read either endpoint of the period as the
fill time from this module.
Costs live INSIDE the equity accounting — there is no ``-lambda * turnover`` shaping weight anyone
can dial to zero; the only knob is ``CostModel``, a physical claim about the market.

**Returns only** (§4.2 BINDING rule): closes enter solely as the ratio ``C_now / C_prev``, and the
cost model never sees a price, so multiplying every close by a constant changes nothing (pinned by
``tests/unit/rl/test_rewards.py``'s scale-invariance property test).

Why this reward: ``sum(reward) / 100 == ln(E_T / E_0)`` (auditable against buy-and-hold); the only
ways to raise it are to be long when the market rises and flat when it falls; churn strictly loses;
concave in wealth. ``DifferentialSharpe`` (Moody & Saffell 1998, eqs. 4-5) is experiment #2 behind
the env's ``reward="dsr"`` flag. Its reward depends on the hidden ``(A, B)`` EMA state, so the
process is not Markov unless that state is observed; the env deliberately keeps its observation at
the 9 shared features and exposes ``(A, B)`` via ``info["dsr_state"]`` only — a documented
limitation of the DSR experiment (``rl/env.py``), not of the default reward.
"""

import math
from dataclasses import dataclass
from typing import Final, Protocol

from trading.rl.costs import CostModel

# x100 makes daily rewards ~+/-2: an affine rescale that leaves the optimal policy unchanged.
REWARD_SCALE: Final = 100.0
# ln is undefined at or below zero growth (a total loss); clamp so the reward stays finite.
GROWTH_FLOOR: Final = 1e-9
# DSR denominator guard: B - A^2 -> 0 for a do-nothing policy, which would make "never trade" an
# exploding attractor (§4.5's mandatory guard).
DSR_VARIANCE_FLOOR: Final = 1e-12


class RewardFn(Protocol):
    """A per-step reward from the equity before and after the step."""

    def __call__(self, *, equity_prev: float, equity_now: float) -> float: ...


def log_return_reward(
    *, equity_prev: float, equity_now: float, scale: float = REWARD_SCALE
) -> float:
    """``scale * ln(max(equity_now / equity_prev, GROWTH_FLOOR))``.

    Additive: over a path with no floored step, ``sum(rewards) / scale == ln(E_T / E_0)``.
    """
    if not math.isfinite(equity_prev) or equity_prev <= 0:
        raise ValueError(f"equity_prev must be finite and positive, got {equity_prev}")
    if not math.isfinite(equity_now):
        raise ValueError(f"equity_now must be finite, got {equity_now}")
    return scale * math.log(max(equity_now / equity_prev, GROWTH_FLOOR))


def net_of_cost_growth(
    *,
    close_prev: float,
    close_now: float,
    position_before: float,
    position_after: float,
    cost: CostModel,
) -> float:
    """One period's equity growth factor ``E_now / E_prev``, net of the trade's cost (§4.5).

    ``1 + position_after * (close_now / close_prev - 1) - cost_fraction``: the closes are consumed
    ONLY as their ratio, so the result is invariant to the price level (the canonical
    ``adjustment=all`` levels are back-adjusted with later dividends — §4.2 BINDING rule). May be
    <= 0 for an impossible-in-practice loss; ``log_return_reward`` floors it.
    """
    for value, what in ((close_prev, "close_prev"), (close_now, "close_now")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{what} must be finite and positive, got {value}")
    cost_paid = cost.cost_fraction(position_before=position_before, position_after=position_after)
    return 1.0 + position_after * (close_now / close_prev - 1.0) - cost_paid


@dataclass(slots=True)
class DifferentialSharpe:
    """Moody & Saffell (NeurIPS 1998) differential Sharpe ratio, eqs. 4-5.

    With ``A``/``B`` exponential moving estimates of the first/second moments of the period
    return ``R_t`` (adaptation rate ``eta``)::

        dA_t = R_t - A_{t-1};   dB_t = R_t^2 - B_{t-1}
        D_t  = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2) ** 1.5
        A_t  = A_{t-1} + eta * dA_t;   B_t = B_{t-1} + eta * dB_t

    ``D_t`` is 0.0 whenever ``B_{t-1} - A_{t-1}^2 <= 1e-12`` (the do-nothing singularity). Just
    above that guard ``D_t`` can be large (the denominator is a 3/2 power of a tiny variance); that
    is acceptable at daily-return scale, where the EMA variance quickly leaves the guard region.
    """

    eta: float = 0.02
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta) or not 0.0 < self.eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {self.eta}")

    def update(self, period_return: float) -> float:
        """Score ``period_return`` against the PRIOR ``(A, B)``, then advance the EMAs."""
        if not math.isfinite(period_return):
            raise ValueError(f"period_return must be finite, got {period_return}")
        delta_a = period_return - self.a
        delta_b = period_return * period_return - self.b
        variance = self.b - self.a * self.a
        if variance <= DSR_VARIANCE_FLOOR:
            reward = 0.0
        else:
            reward = (self.b * delta_a - 0.5 * self.a * delta_b) / variance**1.5
        self.a += self.eta * delta_a
        self.b += self.eta * delta_b
        return reward

    def observation(self) -> tuple[float, float]:
        """The hidden EMA state ``(A, B)`` the reward depends on (the env exposes it via
        ``info["dsr_state"]``; it is not part of the 9-feature observation)."""
        return self.a, self.b

    def reset(self) -> None:
        """Start a new episode from zero moments."""
        self.a = 0.0
        self.b = 0.0
