"""The RL observation: ONE shared feature builder (task-rl-04; docs/RL-SLEEVE-PROGRAM.md §4.3).

PURE domain code: stdlib only (no numpy — the env converts to ``np.float32`` at its own boundary),
zero I/O, no clock, no randomness. Used by BOTH the training env (``rl/env.py``) and any live or
in-browser consumer of the exported policy, so it is the train/serve-skew firewall: there is
exactly one implementation of the observation.

**Inputs are primitives, never a ``BarTable``.** ``trading.domain`` may not import ``trading.rl``
(import-linter "Heavy ML deps never enter the app import graph"), so the env adapts its
``BarTable`` to a plain sequence of float closes before calling in.

**The observation** — ``OBS_DIM = 9`` floats, each clipped to ``[OBS_LOW, OBS_HIGH] = [-10, 10]``.
``C_t`` is ``closes[-1]``; ``C_{t-k}`` is ``closes[-1-k]``; ``s`` is the spec's ``FeatureScales``:

=====  ==============  =====================================================  ==========
idx    name            value (before the clip)                                lookback k
=====  ==============  =====================================================  ==========
0      ``r1``          ``ln(C_t / C_{t-1}) / s.daily_vol``                    1
1      ``r5``          ``ln(C_t / C_{t-5}) / (s.daily_vol * sqrt(5))``        5
2      ``r21``         ``ln(C_t / C_{t-21}) / (s.daily_vol * sqrt(21))``      21
3      ``sma10_gap``   ``(C_t / SMA_10 - 1) / s.sma10_gap``                   9
4      ``sma50_gap``   ``(C_t / SMA_50 - 1) / s.sma50_gap``                   49
5      ``vol21``       ``vol_21 / s.daily_vol - 1``                           21
6      ``trend``       ``(SMA_10 / SMA_50 - 1) / s.trend_gap``                49
7      ``held``        ``1.0`` if a position is held else ``0.0``             0
8      ``unrealized``  ``(C_t / entry_close - 1) / s.unrealized``; ``0.0``    0
                       when flat or when ``entry_close`` is ``None``
=====  ==============  =====================================================  ==========

``lookback k`` is the oldest close a feature reads, as an offset before the decision bar
(``FEATURE_LOOKBACKS``). The arithmetic, specified so a TypeScript mirror can reproduce it (the
left-to-right sums, divisions and ``sqrt`` mirror exactly; ``ln`` is NOT guaranteed bit-identical
across libm / V8 — ~1 ulp — so the rl-09 / rl-12 parity comparisons must use a tolerance, never
bit-for-bit equality):

- ``SMA_n`` = left-to-right float sum of the last ``n`` closes ``C_{t-n+1} .. C_t``, divided by
  ``n``.
- ``vol_21`` = SAMPLE standard deviation (divide by ``n - 1 = 20``) of the 21 one-session log
  returns ``ln(C_i / C_{i-1})`` for ``i = t-20 .. t`` (so it reads ``C_{t-21} .. C_t``): mean =
  left-to-right sum / 21; variance = left-to-right sum of squared deviations / 20; ``sqrt``.
- Every price-derived quantity is a RATIO of closes (or a log/mean of ratios), never a level.

**Returns only — BINDING (program doc §4.2 "Bar adjustment").** The canonical closes are
``adjustment=all``: their ratios are point-in-time total returns but their LEVELS are back-scaled by
dividends paid later, so a level encodes future information (and lets the net memorize the year).
Every feature above is homogeneous of degree zero in the closes: multiplying every close — and
``entry_close``, which must come from the SAME series — by any constant leaves the whole vector
unchanged (pinned by ``test_no_raw_price_level_leaks_through`` with a negative control). There is
no calendar/date input at all, so no date feature can enter.

**Point-in-time contract.** ``build_observation`` takes a SLICE ending at the decision bar
(``closes[: t + 1]``), never ``(full_list, index)``: the future is physically absent from its
inputs. It reads ONLY the last ``spec.window == WARMUP_BARS`` closes ``C_{t-49} .. C_t`` (anything
earlier is ignored, not even validated); ``closes[-1]`` must be a CLOSED bar — the caller (the env,
task-rl-05) owns which bars are known, i.e. every close it passes must have ``known_at <=`` the
decision time. ``MAX_LOOKBACK = 49``: the first legal decision index is ``49`` and a train/eval
purge must keep every 50-bar window on one side of a split (rl-05 episode start, rl-10 purge).

The position state is part of the same contract (program doc §4.7, whose fill timing is still
flagged for task-rl-05 to resolve):

- ``entry_close`` must be the price at which the open position was ACTUALLY filled under the env's
  timing contract (§4.7 / ``execution_lag``). It must be a close no later than ``closes[-1]``,
  already known at decision time. Never pass the decision bar's close when fills are lagged, and
  never a close that is not yet known (e.g. ``C_{t+1}``) — either would make the observation and the
  reward disagree, or leak the fill price.
- ``held`` reflects only fills completed by the decision bar; a pending order is not held.
- ``held=True`` with ``entry_close=None`` yields an unrealized P&L of 0.0 (spec acceptance test 5),
  but the env must never pass that combination — a held position always has a known fill price.

**Normalization is FIXED a-priori constants (``FeatureScales``), never a fitted statistic.**
Parameters fitted over the dataset would leak the future's mean and variance into every
observation; constants use no data at all, so they are trivially free of that leak — there is
deliberately NO ``fit`` step. ``FeatureSpec.to_dict`` / ``FeatureSpec.from_dict`` round-trip the
spec through plain JSON types (ints, floats, strings, lists, dicts) exactly, for the exported
policy artifact and the demo bundle's TypeScript mirror.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Final

FEATURE_NAMES: Final[tuple[str, ...]] = (
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
OBS_DIM: Final = 9
OBS_LOW: Final = -10.0
OBS_HIGH: Final = 10.0

# Window lengths (in closes / returns) of the price-derived features.
_R1: Final = 1
_R5: Final = 5
_R21: Final = 21
_SMA_SHORT: Final = 10
_SMA_LONG: Final = 50
_VOL_RETURNS: Final = 21

# SMA_50 is the longest lookback: the observation at t reads C_{t-49} .. C_t.
WARMUP_BARS: Final = _SMA_LONG
# Oldest close read, as an offset before the decision bar; == first legal decision index.
MAX_LOOKBACK: Final = WARMUP_BARS - 1
# Per-feature lookback (aligned with FEATURE_NAMES): the oldest offset each feature reads.
FEATURE_LOOKBACKS: Final[tuple[int, ...]] = (
    _R1,
    _R5,
    _R21,
    _SMA_SHORT - 1,
    _SMA_LONG - 1,
    _VOL_RETURNS,
    _SMA_LONG - 1,
    0,
    0,
)

_SPEC_KEYS: Final = frozenset({"window", "names", "scales"})


@dataclass(frozen=True, slots=True)
class FeatureScales:
    """FIXED a-priori divisors (~1%/day equity vol). Constants ON PURPOSE: a z-score fitted
    over the dataset would leak the future's mean and variance into every observation."""

    daily_vol: float = 0.01
    sma10_gap: float = 0.02
    sma50_gap: float = 0.05
    trend_gap: float = 0.03
    unrealized: float = 0.05

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"scale {field.name} must be a number, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"scale {field.name} must be finite and positive, got {value}")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Everything that defines the observation function's parameters; embedded in artifacts."""

    window: int  # closed bars required == WARMUP_BARS
    names: tuple[str, ...]
    scales: FeatureScales

    def __post_init__(self) -> None:
        # An int, never a bool or 50.0: a float window would not round-trip through to_dict.
        window: object = self.window
        if isinstance(window, bool) or not isinstance(window, int) or window != WARMUP_BARS:
            raise ValueError(f"window must be WARMUP_BARS ({WARMUP_BARS}), got {self.window!r}")
        names: object = self.names
        if not isinstance(names, tuple | list) or tuple(names) != FEATURE_NAMES:
            raise ValueError(f"names must be {FEATURE_NAMES}, got {self.names!r}")
        # Normalize a list to a tuple so equality and hashing behave (frozen: bypass __setattr__).
        object.__setattr__(self, "names", tuple(names))
        scales: object = self.scales
        if not isinstance(scales, FeatureScales):
            raise ValueError(f"scales must be a FeatureScales, got {self.scales!r}")

    def to_dict(self) -> dict[str, object]:
        """Plain JSON types only; ``from_dict(json.loads(json.dumps(d)))`` == ``self`` exactly."""
        return {
            "window": self.window,
            "names": list(self.names),
            "scales": {name: float(value) for name, value in asdict(self.scales).items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeatureSpec":
        """Inverse of ``to_dict``; fails closed on any missing, extra or mistyped key."""
        if not isinstance(data, Mapping):
            raise ValueError(f"feature spec must be a mapping, got {type(data).__name__}")
        if set(data) != _SPEC_KEYS:
            raise ValueError(f"feature spec keys must be {sorted(_SPEC_KEYS)}, got {sorted(data)}")
        window, names, scales = data["window"], data["names"], data["scales"]
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError(f"window must be an int, got {window!r}")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"names must be a list of strings, got {names!r}")
        expected = {field.name for field in fields(FeatureScales)}
        if not isinstance(scales, Mapping) or set(scales) != expected:
            raise ValueError(f"scales must have exactly the keys {sorted(expected)}")
        values: dict[str, float] = {}
        for key in sorted(expected):
            value = scales[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"scale {key} must be a number, got {value!r}")
            values[key] = float(value)
        return cls(window=window, names=tuple(names), scales=FeatureScales(**values))


def clip(value: float) -> float:
    """Clamp into ``[OBS_LOW, OBS_HIGH]`` (infinities saturate). NaN raises: it would otherwise be
    silently mapped to a bound and hide an arithmetic defect."""
    if math.isnan(value):
        raise ValueError("cannot clip NaN into the observation box")
    return max(OBS_LOW, min(OBS_HIGH, value))


def _sma(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _sample_stdev(values: Sequence[float]) -> float:
    centre = _sma(values)
    squares = 0.0
    for value in values:
        squares += (value - centre) * (value - centre)
    return math.sqrt(squares / (len(values) - 1))


def build_observation(
    closes: Sequence[float], *, held: bool, entry_close: float | None, spec: FeatureSpec
) -> tuple[float, ...]:
    """`closes` is a SLICE ending at the decision bar (closes[-1] == C_t). Takes a slice, never
    (full_list, index): the future is physically absent from this function's inputs, so leakage
    here is impossible by construction rather than by review.
    Raises ValueError if len(closes) < spec.window.

    Only the last ``spec.window`` closes are read; each must be finite and positive. When ``held``
    and ``entry_close`` is given, ``entry_close`` must be finite, positive and from the same price
    series as ``closes`` (a ratio against it is level-free only then).

    PIT (see the module docstring): ``entry_close`` is the price the open position was ACTUALLY
    filled at under the env's timing contract (§4.7 / ``execution_lag``) — a close no later than
    ``closes[-1]``, already known at decision time; never the decision bar's close when fills are
    lagged. ``held`` reflects only fills completed by the decision bar; a pending order is not held.
    ``held=True`` with ``entry_close=None`` yields 0.0 at index 8 (spec acceptance test 5), but the
    env must never pass that combination.
    """
    if len(closes) < spec.window:
        raise ValueError(f"need at least {spec.window} closes, got {len(closes)}")
    window = tuple(float(c) for c in closes[len(closes) - spec.window :])
    for close in window:
        if not math.isfinite(close) or close <= 0:
            raise ValueError(f"closes must be finite and positive, got {close}")
    scales = spec.scales
    c_t = window[-1]

    r1 = math.log(c_t / window[-1 - _R1])
    r5 = math.log(c_t / window[-1 - _R5])
    r21 = math.log(c_t / window[-1 - _R21])
    sma_short = _sma(window[-_SMA_SHORT:])
    sma_long = _sma(window[-_SMA_LONG:])
    returns = [
        math.log(window[i] / window[i - 1]) for i in range(len(window) - _VOL_RETURNS, len(window))
    ]
    vol = _sample_stdev(returns)

    unrealized = 0.0
    if held and entry_close is not None:
        if not math.isfinite(entry_close) or entry_close <= 0:
            raise ValueError(f"entry_close must be finite and positive, got {entry_close}")
        unrealized = (c_t / entry_close - 1.0) / scales.unrealized

    return (
        clip(r1 / scales.daily_vol),
        clip(r5 / (scales.daily_vol * math.sqrt(_R5))),
        clip(r21 / (scales.daily_vol * math.sqrt(_R21))),
        clip((c_t / sma_short - 1.0) / scales.sma10_gap),
        clip((c_t / sma_long - 1.0) / scales.sma50_gap),
        clip(vol / scales.daily_vol - 1.0),
        clip((sma_short / sma_long - 1.0) / scales.trend_gap),
        1.0 if held else 0.0,
        clip(unrealized),
    )
