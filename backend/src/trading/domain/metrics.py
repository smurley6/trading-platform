"""Backtest performance metrics (FR-BT-2): equity curve, return, drawdown. Pure."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class EquityPoint:
    when: datetime
    equity: Decimal


def total_return_pct(start: Decimal, end: Decimal) -> float:
    if start <= 0:
        return 0.0
    return float((end - start) / start * 100)


def max_drawdown_pct(values: Sequence[Decimal]) -> float:
    """Largest peak-to-trough decline over the series, as a positive percentage."""
    peak: Decimal | None = None
    worst = Decimal("0")
    for value in values:
        if peak is None or value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak * 100
            if drawdown > worst:
                worst = drawdown
    return float(worst)
