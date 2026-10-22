"""Cost model (task-rl-03; program doc §4.6, amended 2026-10-21 to basis-points-only).

The model is scale-free: a flat one-way bps of TRADED NOTIONAL, expressed as a fraction of equity.
It never receives a price, a share count or an account size (§4.2 BINDING returns-only rule).
"""

import inspect
import math

import pytest

from trading.rl.costs import (
    BPS,
    COST_BPS_GRID,
    DEFAULT_COST,
    MAX_ONE_WAY_BPS,
    STRESS_COST,
    ZERO_COST,
    CostModel,
)


@pytest.mark.parametrize("one_way_bps", [0.0, 1.0, 2.0, 5.0])
def test_cost_is_charged_on_traded_notional_entering_and_exiting(one_way_bps: float) -> None:
    model = CostModel(one_way_bps=one_way_bps)
    rate = one_way_bps * BPS

    entry = model.cost_fraction(position_before=0.0, position_after=1.0)  # BUY: flat -> long
    exit_ = model.cost_fraction(position_before=1.0, position_after=0.0)  # SELL: long -> flat

    assert entry == pytest.approx(rate, abs=1e-18)
    assert exit_ == pytest.approx(rate, abs=1e-18)
    assert model.cost_fraction(position_before=0.0, position_after=0.5) == pytest.approx(rate / 2)
    # No trade, no cost — holding (long or flat) is free.
    assert model.cost_fraction(position_before=1.0, position_after=1.0) == 0.0
    assert model.cost_fraction(position_before=0.0, position_after=0.0) == 0.0


def test_every_non_zero_model_charges_both_directions() -> None:
    # Replaces the price-level `effective_price` / sells-only `fees` checks the 2026-10-21 ruling
    # removed: both an entry and an exit strictly reduce equity under any non-zero model.
    for model in (
        DEFAULT_COST,
        STRESS_COST,
        *(CostModel(one_way_bps=b) for b in COST_BPS_GRID[1:]),
    ):
        assert model.cost_fraction(position_before=0.0, position_after=1.0) > 0.0
        assert model.cost_fraction(position_before=1.0, position_after=0.0) > 0.0


def test_rate_just_under_the_sanity_bound_is_accepted() -> None:
    assert CostModel(one_way_bps=999.0).one_way_bps == 999.0


def test_named_models_and_the_evaluation_grid() -> None:
    assert DEFAULT_COST == CostModel() == CostModel(one_way_bps=1.0)
    assert DEFAULT_COST.round_trip_bps == 2.0
    assert ZERO_COST.one_way_bps == 0.0
    assert STRESS_COST.one_way_bps == 5.0
    assert COST_BPS_GRID == (0.0, 2.0, 5.0)
    assert DEFAULT_COST.one_way_bps > 0.0  # a zero-cost training default is a correctness defect


def test_flat_default_is_conservative_against_measured_friction() -> None:
    # Evidence cited in the module docstring (documented, deliberately NOT parameters):
    half_spread_bps = 0.15  # $0.01 SPY spread on a ~$650 share, rounded UP to the full spread
    sec_31_sell_bps = 0.206  # $20.60 per $1M of sales, effective 2026-04-04
    finra_taf_sell_bps = 0.002  # $0.000119/share at ~$650
    measured_round_trip_bps = 2 * half_spread_bps + sec_31_sell_bps + finra_taf_sell_bps

    ratio = DEFAULT_COST.round_trip_bps / measured_round_trip_bps
    assert 3.0 <= ratio <= 5.0


def test_cost_model_api_never_takes_a_price_or_share_count() -> None:
    # Structural guard for the BINDING returns-only rule: a per-share fee needs a price LEVEL.
    params = set(inspect.signature(CostModel.cost_fraction).parameters) - {"self"}
    assert params == {"position_before", "position_after"}
    public = {name for name in dir(CostModel) if not name.startswith("_")}
    assert public == {"one_way_bps", "round_trip_bps", "cost_fraction"}


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, MAX_ONE_WAY_BPS, 5_000.0])
def test_rejects_a_negative_non_finite_or_absurd_rate(bad: float) -> None:
    with pytest.raises(ValueError, match="one_way_bps"):
        CostModel(one_way_bps=bad)


@pytest.mark.parametrize(
    ("before", "after"), [(-0.1, 0.0), (0.0, 1.5), (math.nan, 1.0), (0.0, math.inf)]
)
def test_rejects_a_position_outside_long_only_unit_range(before: float, after: float) -> None:
    with pytest.raises(ValueError, match=r"fraction of equity in \[0, 1\]"):
        DEFAULT_COST.cost_fraction(position_before=before, position_after=after)
