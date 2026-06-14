"""Historical price series — the market-context input a backtest replays against."""

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """Per-symbol time series of prices, each sorted ascending by time."""

    series: Mapping[str, tuple[tuple[datetime, Decimal], ...]]

    def price_at(self, symbol: str, when: datetime) -> Decimal | None:
        """The most recent price at or before `when`; the earliest if `when` precedes all."""
        points = self.series.get(symbol)
        if not points:
            return None
        times = [t for t, _ in points]
        index = bisect_right(times, when) - 1
        if index < 0:
            return points[0][1]
        return points[index][1]

    def symbols(self) -> tuple[str, ...]:
        return tuple(self.series)
