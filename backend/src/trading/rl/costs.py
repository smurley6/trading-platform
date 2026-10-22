"""Transaction-cost model for the RL environment (task-rl-03; docs/RL-SLEEVE-PROGRAM.md §4.6).

PURE half of ``trading.rl``: stdlib only — no numpy, no gymnasium, no torch.

**Scale-free by construction (2026-10-21 ruling).** The cost of a trade is a flat ``one_way_bps``
of the TRADED NOTIONAL, charged on every fill — entering and exiting alike. The model never sees a
price, a share count or an account size: its only input is the change in position expressed as a
FRACTION OF EQUITY. That is what the BINDING returns-only rule (§4.2 "Bar adjustment — DECIDED")
requires — the canonical dataset is ``adjustment=all``, whose price LEVELS are back-adjusted with
later dividends, so any per-share fee (which needs a level) would leak current-vintage information
— and it is what lets the static demo page recompute equity client-side from bps alone at
0 / 2 / 5 bps. The program doc's per-share parameterization (``taf_per_share_on_sells``,
``fees(shares, price)``, ``effective_price(price)``) was REMOVED for exactly this reason.

**Why the flat 1.0 bp one-way default is conservative** (documented evidence, deliberately NOT
parameters — the measured components below are folded into one flat rate):

- Half-spread: SPY's quoted spread is typically $0.01 on a ~$650 share ≈ 0.15 bp (SSGA, as of
  2026-03-31), so crossing half of it costs ≈ 0.08 bp; ≈ 0.15 bp is used as the bound.
- SEC Section 31 fee, sells only: $20.60 per $1M of sale proceeds = 0.206 bp, effective
  2026-04-04 (<https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2>); it was $0.00 per
  $1M through 2026-04-03. THIS RATE CHANGES — re-check each fiscal year.
- FINRA Trading Activity Fee, sells only: $0.000119 per share (as cited in the program doc,
  2026-07-31) ≈ 0.002 bp of notional at a ~$650 share (the level appears in this comment only, to
  size the evidence; it is not and must not become an input).
- Commission: $0 — Alpaca US equities are commission-free (2026-07-31).

Measured friction is therefore ≈ 0.15 bp on a buy and ≈ 0.36 bp on a sell (≈ 0.4-0.5 bp round
trip). The default 1.0 bp one-way (2 bp round trip) is deliberately ~3-5x that, because neither
simulator charges anything realistic: Alpaca paper fills at the NBBO with no impact, slippage or
regulatory fees; the in-repo ``PaperBroker`` stamps ``fees=0``; the replay path is bid == ask ==
close (§4.6). At 2 bp round trip a policy that flips daily on a fully-invested notional pays ≈ 5%/yr
— churn is visibly punished but not instantly fatal.

``ZERO_COST`` exists ONLY as a labeled ablation: a zero-cost reward lets the optimal policy churn
on noise (program doc R8). Headline metrics are reported on ``COST_BPS_GRID`` (one-way bps).
"""

import math
from dataclasses import dataclass
from typing import Final

# One basis point as a fraction.
BPS: Final = 1e-4
# Sanity bound: 1000 bp = 10% per one-way trade is not a cost model, it is a typo (e.g. a fraction
# or a percent passed as bps). Rejected rather than silently trained on.
MAX_ONE_WAY_BPS: Final = 1000.0

# The evaluation grid every headline metric is reported on (program doc §5.4 and the demo page's
# cost toggle): ONE-WAY cost in bps of traded notional, i.e. what ``CostModel.one_way_bps`` means.
COST_BPS_GRID: Final[tuple[float, ...]] = (0.0, 2.0, 5.0)


def _require_position(value: float, *, what: str) -> None:
    # Long-only, one fixed size (§4.4): a position is the held fraction of equity in [0, 1].
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{what} must be a fraction of equity in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class CostModel:
    """Flat one-way cost in basis points of traded notional (see the module docstring for why the
    1.0 bp default is ~3-5x measured SPY friction, as of 2026-10-21)."""

    one_way_bps: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.one_way_bps) or not 0 <= self.one_way_bps < MAX_ONE_WAY_BPS:
            raise ValueError(
                f"one_way_bps must be finite and in [0, {MAX_ONE_WAY_BPS}), got {self.one_way_bps}"
            )

    @property
    def round_trip_bps(self) -> float:
        """Cost of entering AND exiting once, in bps of the position's notional."""
        return 2.0 * self.one_way_bps

    def cost_fraction(self, *, position_before: float, position_after: float) -> float:
        """Cost of moving from ``position_before`` to ``position_after`` (each the held fraction
        of equity, in [0, 1]) as a FRACTION OF CURRENT EQUITY.

        Charged once per position change on the traded notional
        ``|position_after - position_before|`` x equity, at the same rate whether the trade enters
        (BUY, flat -> long) or exits (SELL, long -> flat); no trade, no cost. Never takes a price:
        scale-free by construction.
        """
        _require_position(position_before, what="position_before")
        _require_position(position_after, what="position_after")
        return self.one_way_bps * BPS * abs(position_after - position_before)


# Training/evaluation default: ~1.0 bp one-way (2 bp round trip), as of 2026-10-21.
DEFAULT_COST: Final = CostModel()
# ABLATION ONLY — never a training default (program doc R8).
ZERO_COST: Final = CostModel(one_way_bps=0.0)
# ~5 bp one-way robustness sweep (§4.6), the top of COST_BPS_GRID.
STRESS_COST: Final = CostModel(one_way_bps=5.0)
