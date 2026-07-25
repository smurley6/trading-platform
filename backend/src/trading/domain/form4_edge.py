"""Form 4 buy edge calibration (task-13, docs/RESEARCH-PLAN.md §2-§3).

Two pure calibration primitives anchored on the EPISODE-level Phase-7 buy-edge analysis (label run
``a38d3848-f1eb-4a55-8a6d-15e8f1464f7a``), where an *episode* is one ``(ticker, owner)`` filer-on-
name relationship rather than a raw filing row:

1. **Expected edge.** The honest per-trade edge the buy signal carries at h=1 is the EX-LUV
   per-episode h=1 mean, ``+0.41%`` — NOT the naive label-level pooled ``+1.0%`` (which is inflated
   by same-day / same-filer stacking, most egregiously LUV's 194 same-day rows). Sizing/gate code
   that needs an expected edge must anchor on :data:`DEFAULT_FORM4_EXPECTED_EDGE_H1`; the old value
   stays recoverable as :data:`OLD_FORM4_EXPECTED_EDGE_H1` (and via the ``FORM4_EXPECTED_EDGE_H1``
   config key).

2. **Post-acceptance decay.** The edge lives in days 0-5 after acceptance: episode h=1 hit 61.4% /
   +0.41%; h=5 is noise; by h=20 the hit rate is 46% with a NEGATIVE median. A holding/weight curve
   must therefore concentrate weight in days 0-5 and fall away fast. :func:`form4_buy_decay_weight`
   is an exponential half-life curve doing exactly that (monotone non-increasing; weight at t+20d is
   well under 25% of t+1d for the default half-life). ``half_life_days <= 0`` recovers the OLD flat
   (no-decay) holding — every day weighted ``1.0`` — so the pre-task-13 behaviour is recoverable.

This module is pure (zero I/O). The two calibration constants are surfaced through ``config.py``
(``form4_expected_edge_h1`` / ``form4_buy_decay_half_life_days``) so an operator can recover the old
values without a code change; the diagnostic evaluator reports the anchored numbers it is using.
"""

# Old label-level pooled anchor: the h=1 mean abnormal return over ALL Form 4 buy *labels* was
# ~+1.0% (ex-LUV Phase-6 read +1.01%, n=724) — inflated by same-day / same-filer row stacking.
# Recoverable via config (``FORM4_EXPECTED_EDGE_H1=0.01``) for an apples-to-apples comparison.
OLD_FORM4_EXPECTED_EDGE_H1 = 0.01

# New anchor: the EX-LUV per-EPISODE h=1 mean abnormal return, +0.41% (127 episodes / 61.4% hit;
# pinned run a38d3848). This is the honest per-trade edge — one ``(ticker, owner)`` relationship is
# one draw, not one-per-filing.
DEFAULT_FORM4_EXPECTED_EDGE_H1 = 0.004

# Post-acceptance decay half-life (calendar days). Chosen so weight concentrates in days 0-5 and
# t+20d weight is < 25% of t+1d weight (episode edge is at h=1; h=20 hit 46% / median negative).
# ``<= 0`` disables decay (flat 1.0 every day — recovers the pre-task-13 holding).
DEFAULT_FORM4_BUY_DECAY_HALF_LIFE_DAYS = 3.0


def form4_buy_decay_weight(days_since_acceptance: float, *, half_life_days: float) -> float:
    """Exponential decay weight in ``(0, 1]`` for a Form 4 buy ``days_since_acceptance`` days old.

    ``weight = 0.5 ** (days_since_acceptance / half_life_days)`` — monotone non-increasing in the
    day, ``1.0`` at day 0. ``half_life_days <= 0`` disables decay and returns a flat ``1.0`` (the
    OLD, pre-task-13 no-decay holding, recoverable via ``FORM4_BUY_DECAY_HALF_LIFE_DAYS=0``).

    Raises ``ValueError`` for a negative day (there is no weight before acceptance).
    """
    if days_since_acceptance < 0:
        raise ValueError("days_since_acceptance must be non-negative")
    if half_life_days <= 0:
        return 1.0
    return float(0.5 ** (days_since_acceptance / half_life_days))
