"""Conviction engine tuning (FR-POL-4): time decay, corroboration shape, audit breakdown.

These tests pin the *new* tunable behaviour of :func:`compute_conviction` and the
back-compat guarantee that default :class:`ConvictionParams` reproduce the original M1
formula. Signals are constructed directly (not via the ``make_signal`` fixture) where
precise ``created_at`` control is needed for decay.
"""

from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import SignalFactory
from trading.domain.conviction import (
    ConvictionAction,
    ConvictionParams,
    ConvictionThresholds,
    compute_conviction,
)
from trading.domain.signals import Signal, SignalSource

THRESHOLDS = ConvictionThresholds(trade=70.0, watch=40.0)
_REF = datetime(2026, 6, 12, 14, 30, tzinfo=UTC)


def _signal(
    *,
    sid: str,
    created_at: datetime,
    sentiment: float = 0.5,
    politician_weight: float = 1.0,
    source_reliability: float = 1.0,
    source_type: SignalSource = SignalSource.STATEMENT,
    ticker: str = "AAPL",
) -> Signal:
    return Signal(
        id=sid,
        created_at=created_at,
        source_type=source_type,
        ticker=ticker,
        politician_id="pol-1",
        sentiment=sentiment,
        politician_weight=politician_weight,
        source_reliability=source_reliability,
        raw_payload={},
    )


# --------------------------------------------------------------------------- params


def test_params_defaults_match_m1_constants() -> None:
    p = ConvictionParams()
    assert p.scale == 25.0
    assert p.corroboration_bonus_per_source == 0.25
    assert p.decay_half_life_days == 0.0
    assert p.min_sources_for_full_corroboration == 1
    assert p.max_corroboration_factor is None
    assert p.decay_enabled is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scale": 0.0}, "scale must be positive"),
        ({"scale": -1.0}, "scale must be positive"),
        ({"corroboration_bonus_per_source": -0.1}, "non-negative"),
        ({"decay_half_life_days": float("inf")}, "finite"),
        ({"decay_half_life_days": float("nan")}, "finite"),
        ({"min_sources_for_full_corroboration": 0}, ">= 1"),
        ({"max_corroboration_factor": 0.5}, ">= 1 when set"),
    ],
)
def test_params_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ConvictionParams(**kwargs)  # type: ignore[arg-type]


def test_decay_enabled_flag() -> None:
    assert ConvictionParams(decay_half_life_days=5.0).decay_enabled is True
    assert ConvictionParams(decay_half_life_days=0.0).decay_enabled is False
    assert ConvictionParams(decay_half_life_days=-3.0).decay_enabled is False


# --------------------------------------------------------------------------- back-compat


def test_default_params_reproduce_m1_score(make_signal: SignalFactory) -> None:
    signals = [
        make_signal(sentiment=0.9, politician_weight=1.5, source_type=SignalSource.STATEMENT),
        make_signal(sentiment=0.8, politician_weight=1.3, source_type=SignalSource.CONTRACT),
    ]
    result = compute_conviction("AAPL", signals, window_days=21, thresholds=THRESHOLDS)
    # (0.9*1.5 + 0.8*1.3) * 1.25 * 25 = 2.39 * 1.25 * 25
    assert result.score == pytest.approx(74.6875)
    assert result.action is ConvictionAction.TRADE
    assert result.corroboration_factor == pytest.approx(1.25)
    assert result.raw_weighted_sum == pytest.approx(2.39)
    # decay disabled -> decayed == raw
    assert result.decayed_weighted_sum == pytest.approx(result.raw_weighted_sum)
    assert result.reference_time is None


def test_explicit_params_none_equals_default() -> None:
    s = [_signal(sid="a", created_at=_REF, sentiment=0.4)]
    a = compute_conviction("AAPL", s, window_days=21, thresholds=THRESHOLDS, params=None)
    b = compute_conviction(
        "AAPL", s, window_days=21, thresholds=THRESHOLDS, params=ConvictionParams()
    )
    assert a.score == b.score


# --------------------------------------------------------------------------- time decay


def test_decay_downweights_older_signals() -> None:
    params = ConvictionParams(decay_half_life_days=7.0)
    fresh = _signal(sid="fresh", created_at=_REF, sentiment=0.5)
    old = _signal(sid="old", created_at=_REF - timedelta(days=7), sentiment=0.5)
    res = compute_conviction(
        "AAPL", [fresh, old], window_days=21, thresholds=THRESHOLDS, params=params
    )
    # fresh has age 0 -> weight 1.0; old has age == half_life -> weight 0.5
    assert res.contributions["fresh"] == pytest.approx(0.5)  # 1.0 * 0.5
    assert res.contributions["old"] == pytest.approx(0.25)  # 0.5 * 0.5
    assert res.decayed_weighted_sum == pytest.approx(0.75)
    assert res.raw_weighted_sum == pytest.approx(1.0)
    assert res.reference_time == _REF


def test_decay_disabled_is_equal_weight() -> None:
    params = ConvictionParams(decay_half_life_days=0.0)
    fresh = _signal(sid="fresh", created_at=_REF, sentiment=0.5)
    old = _signal(sid="old", created_at=_REF - timedelta(days=30), sentiment=0.5)
    res = compute_conviction(
        "AAPL", [fresh, old], window_days=21, thresholds=THRESHOLDS, params=params
    )
    assert res.contributions["fresh"] == pytest.approx(res.contributions["old"])
    assert res.decayed_weighted_sum == pytest.approx(res.raw_weighted_sum)


def test_reference_time_defaults_to_max_created_at() -> None:
    params = ConvictionParams(decay_half_life_days=10.0)
    a = _signal(sid="a", created_at=_REF - timedelta(days=2), sentiment=0.5)
    newest = _signal(sid="b", created_at=_REF, sentiment=0.5)
    res = compute_conviction(
        "AAPL", [a, newest], window_days=21, thresholds=THRESHOLDS, params=params
    )
    assert res.reference_time == _REF
    # newest signal has age 0 -> full weight
    assert res.contributions["b"] == pytest.approx(0.5)


def test_explicit_now_overrides_reference() -> None:
    params = ConvictionParams(decay_half_life_days=7.0)
    s = _signal(sid="a", created_at=_REF, sentiment=0.5)
    later = _REF + timedelta(days=7)
    res = compute_conviction(
        "AAPL", [s], window_days=21, thresholds=THRESHOLDS, params=params, now=later
    )
    # age == half_life relative to the explicit now -> weight 0.5
    assert res.contributions["a"] == pytest.approx(0.25)
    assert res.reference_time == later


def test_future_dated_signal_not_amplified() -> None:
    params = ConvictionParams(decay_half_life_days=7.0)
    # Signal dated after the explicit reference -> negative age -> clamped to weight 1.0.
    s = _signal(sid="a", created_at=_REF + timedelta(days=5), sentiment=0.5)
    res = compute_conviction(
        "AAPL", [s], window_days=21, thresholds=THRESHOLDS, params=params, now=_REF
    )
    assert res.contributions["a"] == pytest.approx(0.5)  # 1.0 * 0.5, not amplified


def test_decay_can_drop_action_below_threshold() -> None:
    # Two equally strong signals would TRADE without decay; ageing one drops it to WATCH.
    strong = ConvictionParams(decay_half_life_days=0.0)
    decayed = ConvictionParams(decay_half_life_days=3.0)
    fresh = _signal(sid="f", created_at=_REF, sentiment=0.9, politician_weight=1.5)
    old = _signal(
        sid="o",
        created_at=_REF - timedelta(days=12),
        sentiment=0.9,
        politician_weight=1.5,
        source_type=SignalSource.CONTRACT,
    )
    no_decay = compute_conviction(
        "AAPL", [fresh, old], window_days=21, thresholds=THRESHOLDS, params=strong
    )
    with_decay = compute_conviction(
        "AAPL", [fresh, old], window_days=21, thresholds=THRESHOLDS, params=decayed
    )
    assert no_decay.score > with_decay.score


# --------------------------------------------------------------------------- corroboration


def test_corroboration_monotonic_in_sources() -> None:
    params = ConvictionParams()
    scores: list[float] = []
    for sources in (
        [SignalSource.STATEMENT],
        [SignalSource.STATEMENT, SignalSource.CONTRACT],
        [SignalSource.STATEMENT, SignalSource.CONTRACT, SignalSource.DISCLOSED_TRADE],
    ):
        sigs = [
            _signal(sid=f"{i}", created_at=_REF, sentiment=0.3, source_type=src)
            for i, src in enumerate(sources)
        ]
        scores.append(
            compute_conviction(
                "AAPL", sigs, window_days=21, thresholds=THRESHOLDS, params=params
            ).corroboration_factor
        )
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.25)
    assert scores[2] == pytest.approx(1.5)


def test_min_sources_ramps_corroboration() -> None:
    # Require 3 distinct sources for full corroboration: the marginal bonus ramps up.
    params = ConvictionParams(min_sources_for_full_corroboration=3)

    def factor(n: int) -> float:
        srcs = [SignalSource.STATEMENT, SignalSource.CONTRACT, SignalSource.DISCLOSED_TRADE][:n]
        sigs = [
            _signal(sid=f"{i}", created_at=_REF, sentiment=0.3, source_type=s)
            for i, s in enumerate(srcs)
        ]
        return compute_conviction(
            "AAPL", sigs, window_days=21, thresholds=THRESHOLDS, params=params
        ).corroboration_factor

    f1, f2, f3 = factor(1), factor(2), factor(3)
    # ramp_span = 2: step1 fraction=0.5 -> +0.125, step2 fraction=1.0 -> +0.25
    assert f1 == pytest.approx(1.0)
    assert f2 == pytest.approx(1.125)
    assert f3 == pytest.approx(1.375)
    # Monotonic and the ramp makes 2-source weaker than the no-ramp default (1.25).
    assert f1 < f2 < f3
    assert f2 < 1.25


def test_max_corroboration_factor_caps_diminishing_returns() -> None:
    params = ConvictionParams(max_corroboration_factor=1.25)
    sigs = [
        _signal(sid="0", created_at=_REF, sentiment=0.3, source_type=SignalSource.STATEMENT),
        _signal(sid="1", created_at=_REF, sentiment=0.3, source_type=SignalSource.CONTRACT),
        _signal(sid="2", created_at=_REF, sentiment=0.3, source_type=SignalSource.DISCLOSED_TRADE),
    ]
    res = compute_conviction("AAPL", sigs, window_days=21, thresholds=THRESHOLDS, params=params)
    # 3 sources would give 1.5 uncapped; capped to 1.25.
    assert res.corroboration_factor == pytest.approx(1.25)


def test_zero_bonus_disables_corroboration() -> None:
    params = ConvictionParams(corroboration_bonus_per_source=0.0)
    sigs = [
        _signal(sid="0", created_at=_REF, sentiment=0.3, source_type=SignalSource.STATEMENT),
        _signal(sid="1", created_at=_REF, sentiment=0.3, source_type=SignalSource.CONTRACT),
    ]
    res = compute_conviction("AAPL", sigs, window_days=21, thresholds=THRESHOLDS, params=params)
    assert res.corroboration_factor == pytest.approx(1.0)


# --------------------------------------------------------------------------- audit breakdown


def test_result_exposes_full_breakdown() -> None:
    params = ConvictionParams(decay_half_life_days=7.0)
    a = _signal(sid="a", created_at=_REF, sentiment=0.5, source_type=SignalSource.STATEMENT)
    b = _signal(
        sid="b",
        created_at=_REF - timedelta(days=7),
        sentiment=0.5,
        source_type=SignalSource.CONTRACT,
    )
    res = compute_conviction("AAPL", [a, b], window_days=14, thresholds=THRESHOLDS, params=params)

    assert res.window_days == 14
    assert res.contributing_signal_ids == ("a", "b")
    assert set(res.contributions) == {"a", "b"}
    # contributions sum to decayed_weighted_sum
    assert sum(res.contributions.values()) == pytest.approx(res.decayed_weighted_sum)
    assert res.distinct_source_types == tuple(
        sorted((SignalSource.STATEMENT, SignalSource.CONTRACT))
    )
    assert res.corroboration_factor == pytest.approx(1.25)
    assert res.reference_time == _REF


def test_empty_and_mismatched_ticker_breakdown() -> None:
    res = compute_conviction(
        "MSFT",
        [_signal(sid="a", created_at=_REF, sentiment=1.0, ticker="AAPL")],
        window_days=21,
        thresholds=THRESHOLDS,
    )
    assert res.score == 0.0
    assert res.action is ConvictionAction.IGNORE
    assert res.contributing_signal_ids == ()
    assert res.contributions == {}
    assert res.distinct_source_types == ()
    assert res.raw_weighted_sum == 0.0
    assert res.decayed_weighted_sum == 0.0
    assert res.corroboration_factor == pytest.approx(1.0)
    assert res.reference_time is None


def test_negative_sentiment_yields_ignore() -> None:
    params = ConvictionParams(decay_half_life_days=5.0)
    res = compute_conviction(
        "AAPL",
        [_signal(sid="a", created_at=_REF, sentiment=-0.9, politician_weight=1.5)],
        window_days=21,
        thresholds=THRESHOLDS,
        params=params,
    )
    assert res.score == 0.0
    assert res.action is ConvictionAction.IGNORE
    # Breakdown still records the (negative) contribution for the audit trail.
    assert res.decayed_weighted_sum < 0.0
