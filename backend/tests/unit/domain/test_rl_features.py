"""The shared RL feature builder (task-rl-04; program doc §4.3). Stdlib only — no optional extras.

The two properties the rest of the RL program leans on are pinned here:

- **truncation invariance** — the observation at ``t`` is a function of ``closes[t-49 .. t]`` only,
  bit-identically, however the caller produced that slice (the env's leakage suite relies on it);
- **no raw price level** (program doc §4.2 BINDING returns-only rule) — scaling every close (and
  the entry close) by a constant changes nothing, and a negative control proves the detector can
  actually fail.
"""

import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from trading.application.rl_dataset import rl_dataset_from_json
from trading.domain.rl_features import (
    FEATURE_LOOKBACKS,
    FEATURE_NAMES,
    MAX_LOOKBACK,
    OBS_DIM,
    OBS_HIGH,
    OBS_LOW,
    WARMUP_BARS,
    FeatureScales,
    FeatureSpec,
    build_observation,
    clip,
)

_FROZEN = Path(__file__).parents[3] / "data" / "rl" / "spy_daily.json"
_SPEC = FeatureSpec(window=WARMUP_BARS, names=FEATURE_NAMES, scales=FeatureScales())
_HELD = FEATURE_NAMES.index("held")
_UNREALIZED = FEATURE_NAMES.index("unrealized")
_PRICE_FEATURES = range(_HELD)  # indices 0-6: every feature derived from the closes alone

Builder = Callable[..., tuple[float, ...]]


def _random_walk(seed: int, n: int, start: float = 100.0) -> list[float]:
    rng = random.Random(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0003, 0.012)))
    return closes


def _observe(
    closes: Sequence[float], *, held: bool = False, entry_close: float | None = None
) -> tuple[float, ...]:
    return build_observation(closes, held=held, entry_close=entry_close, spec=_SPEC)


# --- acceptance 1-2: warmup and shape --------------------------------------------------------


def test_raises_below_warmup() -> None:
    closes = _random_walk(1, WARMUP_BARS)
    with pytest.raises(ValueError, match="at least 50"):
        _observe(closes[:-1])
    with pytest.raises(ValueError, match="at least 50"):
        _observe([])
    assert len(_observe(closes)) == OBS_DIM


def test_output_length_is_obs_dim_and_names_match() -> None:
    obs = _observe(_random_walk(2, 80), held=True, entry_close=95.0)
    assert OBS_DIM == 9
    assert len(obs) == OBS_DIM == len(FEATURE_NAMES) == len(FEATURE_LOOKBACKS)
    assert FEATURE_NAMES == (
        "r1",
        "r5",
        "r21",
        "sma10_gap",
        "sma50_gap",
        "vol21",
        "trend",
        "held",
        "unrealized",
    )
    assert all(isinstance(value, float) for value in obs)


def test_lookback_constants_are_consistent() -> None:
    # rl-05 starts episodes at MAX_LOOKBACK and rl-10 purges by the window: both key off these.
    assert WARMUP_BARS == 50
    assert MAX_LOOKBACK == WARMUP_BARS - 1 == max(FEATURE_LOOKBACKS) == 49
    assert _SPEC.window == WARMUP_BARS


# --- acceptance 3: clipping -------------------------------------------------------------------


@pytest.mark.parametrize("gap", [10.0, 0.1])
def test_all_values_are_clipped_into_bounds(gap: float) -> None:
    closes = _random_walk(3, 120)
    closes[-1] *= gap  # a pathological 10x gap up (or down) on the decision bar
    for held, entry in [(False, None), (True, 1.0), (True, 1e6), (True, closes[-2])]:
        obs = _observe(closes, held=held, entry_close=entry)
        assert all(OBS_LOW <= value <= OBS_HIGH for value in obs), obs
    obs = _observe(closes)
    saturated = OBS_HIGH if gap > 1 else OBS_LOW
    assert obs[0] == obs[1] == obs[2] == obs[3] == saturated
    assert obs[5] == OBS_HIGH  # a 10x move in a 21-session window is extreme vol either way


def test_clip_saturates_infinities_and_rejects_nan() -> None:
    assert clip(3.25) == 3.25
    assert clip(10.0000001) == OBS_HIGH
    assert clip(-1e9) == OBS_LOW
    assert clip(math.inf) == OBS_HIGH
    assert clip(-math.inf) == OBS_LOW
    with pytest.raises(ValueError, match="NaN"):
        clip(math.nan)


# --- acceptance 4: truncation invariance (the property the env's leakage suite relies on) ----


def test_truncation_invariance() -> None:
    full = _random_walk(4, 500)
    as_tuple = tuple(full)
    for t in range(WARMUP_BARS, len(full)):
        expected = _observe(full[: t + 1])
        truncated = list(full)  # a physically truncated copy: the future does not exist in it
        del truncated[t + 1 :]
        future_poisoned = full[: t + 1] + [math.nan] * (len(full) - t - 1)
        # Anything older than the 50-bar window is never read, not even validated.
        past_poisoned = [math.nan] * (t - MAX_LOOKBACK) + full[t - MAX_LOOKBACK : t + 1]
        variants = (
            truncated,
            full[t - MAX_LOOKBACK : t + 1],  # the minimal window
            as_tuple[: t + 1],
            past_poisoned,
        )
        for variant in variants:
            assert _observe(variant) == expected, t
        assert _observe(future_poisoned[: t + 1]) == expected, t
        assert not any(math.isnan(value) for value in expected)


def test_first_legal_decision_index_is_max_lookback() -> None:
    full = _random_walk(5, 60)
    assert len(_observe(full[: MAX_LOOKBACK + 1])) == OBS_DIM
    with pytest.raises(ValueError, match="at least"):
        _observe(full[:MAX_LOOKBACK])


@pytest.mark.parametrize("index", range(OBS_DIM))
def test_declared_lookbacks_match_what_each_feature_reads(index: int) -> None:
    # FEATURE_LOOKBACKS is the contract rl-05/rl-10 target: feature i reads C_{t-k} but nothing
    # older. Bumping the close at offset k must move it (except `held`, which reads no close);
    # bumping offset k+1 must leave it bit-identical.
    closes = _random_walk(6, 100)
    entry = closes[-30]
    lookback = FEATURE_LOOKBACKS[index]
    base = _observe(closes, held=True, entry_close=entry)

    at_lookback = list(closes)
    at_lookback[-1 - lookback] *= 1.03
    moved = _observe(at_lookback, held=True, entry_close=entry)[index]
    if index == _HELD:
        assert moved == base[index]
    else:
        assert moved != base[index]

    beyond = list(closes)
    beyond[-2 - lookback] *= 1.03
    assert _observe(beyond, held=True, entry_close=entry)[index] == base[index]


# --- acceptance 5: position state ---------------------------------------------------------


def test_held_and_unrealized_encode_position_state() -> None:
    closes = _random_walk(7, 90)
    flat = _observe(closes, held=False, entry_close=None)
    held = _observe(closes, held=True, entry_close=closes[-20])
    assert flat[_HELD] == 0.0
    assert held[_HELD] == 1.0
    assert flat[:_HELD] == held[:_HELD]  # the price features do not see the position
    expected = (closes[-1] / closes[-20] - 1.0) / FeatureScales().unrealized
    assert held[_UNREALIZED] == pytest.approx(expected, rel=1e-12)
    # entry_close=None forces the unrealized P&L to exactly 0.0 — held or not.
    assert _observe(closes, held=True, entry_close=None)[_UNREALIZED] == 0.0
    assert flat[_UNREALIZED] == 0.0
    # Flat means no open P&L, whatever stale entry the caller passes.
    assert _observe(closes, held=False, entry_close=closes[-20])[_UNREALIZED] == 0.0
    # Entered at the decision close: zero open P&L.
    assert _observe(closes, held=True, entry_close=closes[-1])[_UNREALIZED] == 0.0


# --- acceptance 6: the BINDING returns-only rule ------------------------------------------


def _assert_level_free(
    closes: Sequence[float],
    *,
    factor: float,
    held: bool = False,
    entry_close: float | None = None,
    indices: Sequence[int] = range(OBS_DIM),
    builder: Builder = build_observation,
) -> None:
    """Multiply every close — and the entry close, which lives on the same price series — by
    ``factor``; every feature in ``indices`` must be unchanged to 1e-12."""
    base = builder(closes, held=held, entry_close=entry_close, spec=_SPEC)
    scaled = builder(
        [c * factor for c in closes],
        held=held,
        entry_close=None if entry_close is None else entry_close * factor,
        spec=_SPEC,
    )
    for i in indices:
        same = math.isclose(base[i], scaled[i], rel_tol=1e-12, abs_tol=1e-12)
        message = f"feature {FEATURE_NAMES[i]} depends on the price level: {base[i]} vs {scaled[i]}"
        assert same, message


def test_no_raw_price_level_leaks_through() -> None:
    # The spec's statement: scaling the entire series by 2.0 leaves indices 0-6 unchanged.
    for seed in range(20):
        closes = _random_walk(seed, 300, start=random.Random(seed).uniform(5.0, 900.0))
        for t in range(MAX_LOOKBACK, len(closes), 7):
            _assert_level_free(closes[: t + 1], factor=2.0, indices=_PRICE_FEATURES)


@pytest.mark.parametrize("factor", [2.0, 10.0, 1e-3, 7.3])
def test_the_full_observation_is_invariant_to_the_price_level(factor: float) -> None:
    # Extends the rule to all 9 features: the entry close scales with the series, so the
    # unrealized P&L is a ratio too; the position flag has no price input at all.
    for seed in range(10):
        closes = _random_walk(100 + seed, 200, start=random.Random(seed).uniform(5.0, 900.0))
        for t in range(MAX_LOOKBACK, len(closes), 11):
            window = closes[: t + 1]
            _assert_level_free(window, factor=factor)
            _assert_level_free(window, factor=factor, held=True, entry_close=closes[t // 2])
            _assert_level_free(window, factor=factor, held=True, entry_close=None)


def test_the_frozen_spy_observations_are_invariant_to_the_price_level() -> None:
    closes = [bar.close for bar in rl_dataset_from_json(_FROZEN.read_bytes()).bars]
    assert len(closes) > 1000
    for t in range(MAX_LOOKBACK, len(closes)):
        _assert_level_free(
            closes[: t + 1], factor=10.0, held=t % 2 == 0, entry_close=closes[t - t % 13]
        )


def test_price_level_detector_catches_a_level_feature() -> None:
    # Negative control — a check that cannot fail is decoration. This fake builder swaps r1 for
    # the log of the raw close (a LEVEL); the detector must reject it.
    def leaky(
        closes: Sequence[float], *, held: bool, entry_close: float | None, spec: FeatureSpec
    ) -> tuple[float, ...]:
        obs = list(build_observation(closes, held=held, entry_close=entry_close, spec=spec))
        obs[0] = clip(math.log(closes[-1]))
        return tuple(obs)

    closes = _random_walk(8, 120)
    _assert_level_free(closes, factor=2.0, indices=range(1, OBS_DIM), builder=leaky)
    with pytest.raises(AssertionError, match="r1 depends on the price level"):
        _assert_level_free(closes, factor=2.0, indices=_PRICE_FEATURES, builder=leaky)


def test_price_level_detector_catches_an_unscaled_entry_close() -> None:
    # Second negative control: an entry price NOT taken from the same (scaled) series is a level.
    def stale_entry(
        closes: Sequence[float], *, held: bool, entry_close: float | None, spec: FeatureSpec
    ) -> tuple[float, ...]:
        return build_observation(closes, held=held, entry_close=100.0, spec=spec)

    with pytest.raises(AssertionError, match="unrealized depends on the price level"):
        _assert_level_free(
            _random_walk(9, 120), factor=2.0, held=True, entry_close=100.0, builder=stale_entry
        )


# --- the formulas ---------------------------------------------------------------------------


def test_features_match_an_independent_reference_implementation() -> None:
    # Hand-written reference (math.fsum + statistics.stdev) for program doc §4.3's table.
    closes = _random_walk(10, 140)
    entry = closes[-17]
    obs = _observe(closes, held=True, entry_close=entry)
    c = closes
    s = FeatureScales()
    sma10 = math.fsum(c[-10:]) / 10
    sma50 = math.fsum(c[-50:]) / 50
    log_returns = [math.log(c[i] / c[i - 1]) for i in range(len(c) - 21, len(c))]
    reference = (
        math.log(c[-1] / c[-2]) / s.daily_vol,
        math.log(c[-1] / c[-6]) / (s.daily_vol * math.sqrt(5)),
        math.log(c[-1] / c[-22]) / (s.daily_vol * math.sqrt(21)),
        (c[-1] / sma10 - 1) / s.sma10_gap,
        (c[-1] / sma50 - 1) / s.sma50_gap,
        statistics.stdev(log_returns) / s.daily_vol - 1,
        (sma10 / sma50 - 1) / s.trend_gap,
        1.0,
        (c[-1] / entry - 1) / s.unrealized,
    )
    assert all(OBS_LOW < value < OBS_HIGH for value in reference)  # nothing clipped here
    for got, want in zip(obs, reference, strict=True):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_a_constant_series_is_all_zero_except_vol() -> None:
    obs = _observe([42.0] * 60)
    assert obs[:5] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs[5] == -1.0  # zero realized vol sits one daily_vol below the a-priori level
    assert obs[6:] == (0.0, 0.0, 0.0)


# --- input validation ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("position", [-1, -25, -WARMUP_BARS])
def test_a_bad_close_inside_the_window_raises(bad: float, position: int) -> None:
    closes = _random_walk(11, 70)
    closes[position] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        _observe(closes)


@pytest.mark.parametrize("bad", [0.0, -5.0, math.nan, math.inf])
def test_a_bad_entry_close_raises_when_held(bad: float) -> None:
    closes = _random_walk(12, 70)
    with pytest.raises(ValueError, match="entry_close"):
        _observe(closes, held=True, entry_close=bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [("daily_vol", 0.0), ("sma10_gap", -0.02), ("trend_gap", math.inf), ("unrealized", math.nan)],
)
def test_feature_scales_reject_non_positive_or_non_finite(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        FeatureScales(**{field: value})


def test_feature_scales_reject_a_bool() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        FeatureScales(daily_vol=True)


@pytest.mark.parametrize("window", [WARMUP_BARS - 1, WARMUP_BARS + 1, 0, True])
def test_feature_spec_pins_the_window(window: int) -> None:
    with pytest.raises(ValueError, match="WARMUP_BARS"):
        FeatureSpec(window=window, names=FEATURE_NAMES, scales=FeatureScales())


@pytest.mark.parametrize(
    "names", [FEATURE_NAMES[:-1], (*FEATURE_NAMES[:-2], "unrealized", "held"), ("date", "close")]
)
def test_feature_spec_pins_the_names(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="names"):
        FeatureSpec(window=WARMUP_BARS, names=names, scales=FeatureScales())


def test_feature_spec_normalizes_a_names_list_to_a_tuple() -> None:
    spec = FeatureSpec(window=WARMUP_BARS, names=list(FEATURE_NAMES), scales=FeatureScales())  # type: ignore[arg-type]
    assert spec.names == FEATURE_NAMES
    assert isinstance(spec.names, tuple)
    assert spec == _SPEC
    assert hash(spec) == hash(_SPEC)


def test_feature_spec_rejects_a_float_window() -> None:
    # 50.0 == 50, but it would not round-trip through to_dict/from_dict.
    with pytest.raises(ValueError, match="WARMUP_BARS"):
        FeatureSpec(window=50.0, names=FEATURE_NAMES, scales=FeatureScales())  # type: ignore[arg-type]


def test_feature_spec_rejects_scales_that_are_not_feature_scales() -> None:
    with pytest.raises(ValueError, match="scales must be a FeatureScales"):
        FeatureSpec(window=WARMUP_BARS, names=FEATURE_NAMES, scales={"daily_vol": 0.01})  # type: ignore[arg-type]


# --- serialization: the artifact / TypeScript-mirror contract -----------------------------


def test_feature_spec_round_trips_through_json_exactly() -> None:
    custom = FeatureSpec(
        window=WARMUP_BARS,
        names=FEATURE_NAMES,
        scales=FeatureScales(daily_vol=0.0123456789012345, unrealized=0.1 + 0.2),
    )
    for spec in (_SPEC, custom):
        payload = spec.to_dict()
        restored = FeatureSpec.from_dict(json.loads(json.dumps(payload)))
        assert restored == spec
        assert restored.to_dict() == payload
    assert _SPEC.to_dict() == {
        "window": 50,
        "names": list(FEATURE_NAMES),
        "scales": {
            "daily_vol": 0.01,
            "sma10_gap": 0.02,
            "sma50_gap": 0.05,
            "trend_gap": 0.03,
            "unrealized": 0.05,
        },
    }


def test_from_dict_accepts_integral_json_numbers() -> None:
    payload = _SPEC.to_dict()
    payload["scales"] = {
        "daily_vol": 1,
        "sma10_gap": 2,
        "sma50_gap": 5,
        "trend_gap": 3,
        "unrealized": 5,
    }
    assert FeatureSpec.from_dict(payload).scales == FeatureScales(1.0, 2.0, 5.0, 3.0, 5.0)


def _mutated(**changes: object) -> dict[str, object]:
    payload = _SPEC.to_dict()
    payload.update(changes)
    return payload


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"window": 50, "names": list(FEATURE_NAMES)}, "keys"),
        (_mutated(extra=1), "keys"),
        (_mutated(window=50.0), "window must be an int"),
        (_mutated(window=True), "window must be an int"),
        (_mutated(window=49), "WARMUP_BARS"),
        (_mutated(names="r1"), "names must be a list"),
        (_mutated(names=[1, 2]), "names must be a list"),
        (_mutated(names=list(reversed(FEATURE_NAMES))), "names must be"),
        (_mutated(scales=[0.01]), "scales must have exactly"),
        (_mutated(scales={"daily_vol": 0.01}), "scales must have exactly"),
        (
            _mutated(
                scales={
                    "daily_vol": "0.01",
                    "sma10_gap": 0.02,
                    "sma50_gap": 0.05,
                    "trend_gap": 0.03,
                    "unrealized": 0.05,
                }
            ),
            "daily_vol must be a number",
        ),
        (
            _mutated(
                scales={
                    "daily_vol": 0.01,
                    "sma10_gap": 0.02,
                    "sma50_gap": 0.05,
                    "trend_gap": False,
                    "unrealized": 0.05,
                }
            ),
            "trend_gap must be a number",
        ),
        (
            _mutated(
                scales={
                    "daily_vol": 0.01,
                    "sma10_gap": 0.02,
                    "sma50_gap": 0.05,
                    "trend_gap": 0.03,
                    "unrealized": -0.05,
                }
            ),
            "unrealized must be finite and positive",
        ),
    ],
)
def test_from_dict_fails_closed(payload: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FeatureSpec.from_dict(payload)


@pytest.mark.parametrize("payload", [[("window", 50)], "spec", None, 50])
def test_from_dict_rejects_a_non_mapping_with_value_error(payload: object) -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        FeatureSpec.from_dict(payload)  # type: ignore[arg-type]
