"""Form 4 buy edge calibration (task-13): decay-curve shape + anchored constants.

Pure, hand-checked. The decay curve must concentrate weight in days 0-5, be monotone
non-increasing, and put < 25% of the t+1d weight at t+20d; ``half_life <= 0`` must recover the OLD
flat (no-decay) holding. The expected-edge constants must expose the new episode-level anchor
(+0.4%) with the OLD label-level anchor (+1.0%) still available.
"""

import math
from itertools import pairwise

import pytest

from trading.domain.form4_edge import (
    DEFAULT_FORM4_BUY_DECAY_HALF_LIFE_DAYS,
    DEFAULT_FORM4_EXPECTED_EDGE_H1,
    OLD_FORM4_EXPECTED_EDGE_H1,
    form4_buy_decay_weight,
)

_H = DEFAULT_FORM4_BUY_DECAY_HALF_LIFE_DAYS


def test_expected_edge_constants_new_and_old_recoverable() -> None:
    # New anchor = ex-LUV per-episode h=1 mean (+0.41% -> 0.004); OLD label-level pooled (+1.0%).
    assert DEFAULT_FORM4_EXPECTED_EDGE_H1 == 0.004
    assert OLD_FORM4_EXPECTED_EDGE_H1 == 0.01
    assert DEFAULT_FORM4_EXPECTED_EDGE_H1 < OLD_FORM4_EXPECTED_EDGE_H1


def test_decay_weight_is_one_at_day_zero() -> None:
    assert form4_buy_decay_weight(0.0, half_life_days=_H) == 1.0


def test_decay_weight_halves_every_half_life() -> None:
    assert math.isclose(form4_buy_decay_weight(_H, half_life_days=_H), 0.5, abs_tol=1e-12)
    assert math.isclose(form4_buy_decay_weight(2 * _H, half_life_days=_H), 0.25, abs_tol=1e-12)


def test_decay_weight_t20_under_quarter_of_t1() -> None:
    # Acceptance test 3: weight at t+20d < 25% of weight at t+1d (edge lives in days 0-5).
    w1 = form4_buy_decay_weight(1.0, half_life_days=_H)
    w20 = form4_buy_decay_weight(20.0, half_life_days=_H)
    assert w20 < 0.25 * w1


def test_decay_weight_monotone_non_increasing() -> None:
    # Acceptance test 3: monotone non-increasing across the whole holding window.
    weights = [form4_buy_decay_weight(float(day), half_life_days=_H) for day in range(0, 21)]
    assert all(later <= earlier for earlier, later in pairwise(weights))


def test_decay_weight_concentrated_in_days_0_to_5() -> None:
    # The majority of the total holding weight sits in days 0-5 (the edge window).
    days_0_5 = sum(form4_buy_decay_weight(float(d), half_life_days=_H) for d in range(0, 6))
    days_6_20 = sum(form4_buy_decay_weight(float(d), half_life_days=_H) for d in range(6, 21))
    assert days_0_5 > days_6_20


def test_half_life_zero_recovers_old_flat_holding() -> None:
    # OLD (pre-task-13) behaviour recoverable: no decay -> flat 1.0 at every day.
    for day in (0.0, 1.0, 5.0, 20.0, 100.0):
        assert form4_buy_decay_weight(day, half_life_days=0.0) == 1.0
    assert form4_buy_decay_weight(20.0, half_life_days=-1.0) == 1.0


def test_negative_day_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        form4_buy_decay_weight(-1.0, half_life_days=_H)
