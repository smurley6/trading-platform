"""Conviction scoring (FR-POL-4/5): corroborated, weighted signal aggregation.

This module is the core conviction engine. It is intentionally pure, deterministic,
and side-effect free (no I/O, no wall-clock, no randomness) so the *same* code path
runs live and in backtest replay (FR-BT-1).

The score for a ticker is built from three composable, independently tunable pieces,
all parameterised by :class:`ConvictionParams`:

1. **Time decay** (FR-POL-4): each in-window signal is weighted by an exponential
   decay of its age relative to a deterministic reference time. Older signals matter
   less. A half-life of ``<= 0`` disables decay (every signal gets weight ``1.0``),
   which reproduces the original M1 behaviour exactly.

2. **Weighted aggregation**: the decayed signals are summed as
   ``decay_weight · sentiment · politician_weight · source_reliability``.

3. **Corroboration** (FR-POL-4): conviction rises as *independent source types*
   align, with configurable diminishing returns and an optional requirement that a
   minimum number of distinct source types corroborate before the full multiplier is
   reached. The factor is monotonic non-decreasing in the number of sources and
   bounded above.

The aggregate is scaled, clamped to ``[0, 100]``, and mapped to a
:class:`ConvictionAction` by threshold (FR-POL-5).

For auditability (spec section 12 — "what conviction, what signals") the returned
:class:`ConvictionResult` exposes the full component breakdown: the raw and
decay-adjusted weighted sums, the corroboration factor, the distinct source types,
and a per-signal contribution list (signal id -> decayed weighted contribution).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from trading.domain.signals import Signal, SignalSource

# Default scale: maps the weighted sentiment sum onto the 0-100 conviction range.
_DEFAULT_SCALE = 25.0
# Default corroboration bonus per independent source type beyond the first.
_DEFAULT_CORROBORATION_BONUS_PER_SOURCE = 0.25


class ConvictionAction(StrEnum):
    TRADE = "trade"
    WATCH = "watch"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class ConvictionThresholds:
    """Action thresholds on the 0-100 conviction scale (FR-POL-5)."""

    trade: float = 70.0
    watch: float = 40.0

    def __post_init__(self) -> None:
        if self.watch > self.trade:
            raise ValueError("watch threshold must not exceed trade threshold")


@dataclass(frozen=True, slots=True)
class ConvictionParams:
    """Tunable knobs for the conviction engine (FR-POL-4).

    Defaults reproduce the original M1 formula exactly: with ``decay_half_life_days <= 0``
    (decay disabled, every signal weighted ``1.0``) and ``min_sources_for_full_corroboration
    == 1``, the corroboration factor collapses to ``1 + bonus_per_source · (n_sources - 1)``
    and the score equals ``clamp(weighted_sum · corroboration · scale, 0, 100)``.

    Attributes:
        scale: Multiplier mapping the corroborated weighted sum onto the 0-100 range.
        corroboration_bonus_per_source: Marginal lift per fully-counted independent
            source type beyond the first.
        decay_half_life_days: Exponential half-life (in days) for time-decay of signal
            age. ``<= 0`` disables decay (equal weights). Must be finite.
        min_sources_for_full_corroboration: Number of distinct source types that must
            corroborate before each source contributes its *full* bonus. Below this
            count, each present source's marginal bonus is scaled down linearly, so
            high conviction is harder to reach on thin corroboration. ``1`` (the
            default) means every source counts fully — i.e. no ramp.
        max_corroboration_factor: Upper bound on the corroboration multiplier
            (diminishing-returns cap). ``None`` leaves it unbounded above. Must be
            ``>= 1`` when set.
    """

    scale: float = _DEFAULT_SCALE
    corroboration_bonus_per_source: float = _DEFAULT_CORROBORATION_BONUS_PER_SOURCE
    decay_half_life_days: float = 0.0
    min_sources_for_full_corroboration: int = 1
    max_corroboration_factor: float | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.corroboration_bonus_per_source < 0.0:
            raise ValueError("corroboration_bonus_per_source must be non-negative")
        if not math.isfinite(self.decay_half_life_days):
            raise ValueError("decay_half_life_days must be finite")
        if self.min_sources_for_full_corroboration < 1:
            raise ValueError("min_sources_for_full_corroboration must be >= 1")
        if self.max_corroboration_factor is not None and self.max_corroboration_factor < 1.0:
            raise ValueError("max_corroboration_factor must be >= 1 when set")

    @property
    def decay_enabled(self) -> bool:
        return self.decay_half_life_days > 0.0


@dataclass(frozen=True, slots=True)
class ConvictionResult:
    """Conviction outcome plus its full component breakdown (spec section 12).

    The first four fields preserve the original public surface so existing callers keep
    working unchanged. The remaining fields are the audit breakdown answering "what
    conviction, and from which signals".

    Attributes:
        ticker: The ticker scored.
        window_days: Corroboration window the caller pre-filtered to (carried through
            for the audit trail; the engine does not re-filter).
        contributing_signal_ids: Ids of the signals that fed this score, in input order.
        score: Final conviction on the 0-100 scale.
        action: Threshold-mapped action (FR-POL-5).
        raw_weighted_sum: Weighted sum with *no* time decay (decay weights forced to 1).
        decayed_weighted_sum: Weighted sum *with* time decay applied. Equals
            ``raw_weighted_sum`` when decay is disabled.
        corroboration_factor: The corroboration multiplier actually applied.
        distinct_source_types: Distinct source types observed among contributing signals.
        contributions: Per-signal decayed weighted contribution
            (``decay_weight · sentiment · politician_weight · source_reliability``),
            keyed by signal id, in input order. Sums to ``decayed_weighted_sum``.
        reference_time: The deterministic "now" used for decay (max ``created_at`` of the
            contributing signals), or ``None`` when there were no signals or decay is off.
    """

    ticker: str
    window_days: int
    contributing_signal_ids: tuple[str, ...]
    score: float  # 0 .. 100
    action: ConvictionAction
    raw_weighted_sum: float = 0.0
    decayed_weighted_sum: float = 0.0
    corroboration_factor: float = 1.0
    distinct_source_types: tuple[SignalSource, ...] = ()
    contributions: Mapping[str, float] = field(default_factory=dict)
    reference_time: datetime | None = None


def _reference_time(signals: Sequence[Signal]) -> datetime | None:
    """Deterministic "now" for time decay: the most recent signal's ``created_at``.

    The strategy seam (``evaluate(signals, portfolio)``) carries no wall-clock, and using
    the real clock would break backtest/live determinism (FR-BT-1). We therefore anchor
    decay to the newest in-window signal: the freshest signal always has age 0, and the
    relative weighting of any two signals depends only on the data, not on when the
    engine happens to run.
    """
    if not signals:
        return None
    return max(s.created_at for s in signals)


def _decay_weight(signal: Signal, reference: datetime | None, params: ConvictionParams) -> float:
    """Exponential time-decay weight in ``(0, 1]`` for one signal.

    ``weight = 0.5 ** (age_days / half_life)``. Returns ``1.0`` when decay is disabled,
    so the original equal-weight behaviour is preserved exactly. Signals dated in the
    future relative to the reference (should not happen post-filtering) are clamped to
    weight ``1.0`` rather than amplified.
    """
    if reference is None or not params.decay_enabled:
        return 1.0
    age_days = (reference - signal.created_at).total_seconds() / 86_400.0
    if age_days <= 0.0:
        return 1.0
    return float(0.5 ** (age_days / params.decay_half_life_days))


def _corroboration_factor(n_sources: int, params: ConvictionParams) -> float:
    """Monotonic, bounded corroboration multiplier for ``n_sources`` distinct types.

    Construction (FR-POL-4):

    * The first source type establishes a baseline factor of ``1.0`` (no corroboration).
    * Each *additional* distinct source type adds ``corroboration_bonus_per_source``,
      i.e. ``1 + bonus · (n - 1)`` — identical to the original M1 formula when
      ``min_sources_for_full_corroboration == 1``.
    * **Minimum-sources ramp:** while fewer than ``min_sources_for_full_corroboration``
      types have corroborated, each newly-added type contributes only a *fraction* of
      the bonus that grows linearly with corroboration depth
      (``ramp_position / (min_sources - 1)``). This makes high conviction unreachable on
      thin corroboration without ever decreasing the factor as sources are added.
    * **Diminishing-returns cap:** the result is clamped to ``max_corroboration_factor``
      when set.

    The factor is non-decreasing in ``n_sources`` and always ``>= 1.0``.
    """
    extra = max(0, n_sources - 1)
    bonus = params.corroboration_bonus_per_source
    min_full = params.min_sources_for_full_corroboration

    if min_full <= 1 or bonus == 0.0:
        factor = 1.0 + bonus * extra
    else:
        # Ramp the marginal bonus from 0 up to full over the first (min_full - 1)
        # additional sources; every source past min_full contributes the full bonus.
        ramp_span = min_full - 1
        accumulated = 0.0
        for step in range(1, extra + 1):
            fraction = min(1.0, step / ramp_span)
            accumulated += bonus * fraction
        factor = 1.0 + accumulated

    if params.max_corroboration_factor is not None:
        factor = min(factor, params.max_corroboration_factor)
    return factor


def compute_conviction(
    ticker: str,
    signals: Sequence[Signal],
    window_days: int,
    thresholds: ConvictionThresholds,
    params: ConvictionParams | None = None,
    now: datetime | None = None,
) -> ConvictionResult:
    """Aggregate signals for one ticker into a conviction score and action (FR-POL-4/5).

    Callers are responsible for pre-filtering ``signals`` to the corroboration window;
    the engine scores whatever it is given (filtering only by ``ticker``).

    Args:
        ticker: The ticker to score. Only signals whose ``ticker`` matches contribute.
        signals: In-window signals (any tickers; mismatched tickers are ignored).
        window_days: Carried through onto the result for the audit trail.
        thresholds: Action thresholds (FR-POL-5).
        params: Tuning knobs. ``None`` uses defaults that reproduce M1 behaviour.
        now: Deterministic reference time for decay. ``None`` derives it from the data
            as the max ``created_at`` among contributing signals (see ``_reference_time``);
            passing it explicitly lets a strategy share one reference across tickers.

    Returns:
        A :class:`ConvictionResult` with the score, action, and full audit breakdown.
    """
    if params is None:
        params = ConvictionParams()

    relevant = [s for s in signals if s.ticker == ticker]
    reference = now if now is not None else _reference_time(relevant)

    raw_weighted_sum = 0.0
    decayed_weighted_sum = 0.0
    contributions: dict[str, float] = {}
    for s in relevant:
        base = s.sentiment * s.politician_weight * s.source_reliability
        contribution = _decay_weight(s, reference, params) * base
        raw_weighted_sum += base
        decayed_weighted_sum += contribution
        contributions[s.id] = contribution

    distinct_sources = {s.source_type for s in relevant}
    corroboration = _corroboration_factor(len(distinct_sources), params)
    score = min(100.0, max(0.0, decayed_weighted_sum * corroboration * params.scale))

    if score >= thresholds.trade:
        action = ConvictionAction.TRADE
    elif score >= thresholds.watch:
        action = ConvictionAction.WATCH
    else:
        action = ConvictionAction.IGNORE

    return ConvictionResult(
        ticker=ticker,
        window_days=window_days,
        contributing_signal_ids=tuple(s.id for s in relevant),
        score=score,
        action=action,
        raw_weighted_sum=raw_weighted_sum,
        decayed_weighted_sum=decayed_weighted_sum,
        corroboration_factor=corroboration,
        distinct_source_types=tuple(sorted(distinct_sources)),
        contributions=contributions,
        reference_time=reference if params.decay_enabled else None,
    )
