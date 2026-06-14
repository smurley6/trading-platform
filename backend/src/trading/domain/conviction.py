"""Conviction scoring (FR-POL-4/5): corroborated, weighted signal aggregation.

M1 ships a deliberately simple, deterministic formula; calibration is M2 work.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from trading.domain.signals import Signal

# Scales the weighted sentiment sum onto the 0-100 conviction range.
_SCALE = 25.0
# Conviction rises when independent source types align (FR-POL-4).
_CORROBORATION_BONUS_PER_SOURCE = 0.25


class ConvictionAction(StrEnum):
    TRADE = "trade"
    WATCH = "watch"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class ConvictionThresholds:
    trade: float = 70.0
    watch: float = 40.0

    def __post_init__(self) -> None:
        if self.watch > self.trade:
            raise ValueError("watch threshold must not exceed trade threshold")


@dataclass(frozen=True, slots=True)
class ConvictionResult:
    ticker: str
    window_days: int
    contributing_signal_ids: tuple[str, ...]
    score: float  # 0 .. 100
    action: ConvictionAction


def compute_conviction(
    ticker: str,
    signals: Sequence[Signal],
    window_days: int,
    thresholds: ConvictionThresholds,
) -> ConvictionResult:
    """Aggregate signals for one ticker into a conviction score and action.

    Callers are responsible for pre-filtering `signals` to the corroboration window.
    """
    relevant = [s for s in signals if s.ticker == ticker]
    weighted_sum = sum(s.sentiment * s.politician_weight * s.source_reliability for s in relevant)
    distinct_sources = {s.source_type for s in relevant}
    corroboration = 1.0 + _CORROBORATION_BONUS_PER_SOURCE * max(0, len(distinct_sources) - 1)
    score = min(100.0, max(0.0, weighted_sum * corroboration * _SCALE))

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
    )
