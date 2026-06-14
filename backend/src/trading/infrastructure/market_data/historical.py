"""Historical market data adapter for backtesting: prices a symbol at the clock's time.

Also provides `synthetic_price_series`, a deterministic illustrative price path used by the
backtest API when real historical data is not configured. It is reproducible (seeded by
symbol) so a given set of signals always backtests to the same result, and `SyntheticHistorical
Prices` exposes it behind the `HistoricalPricesPort` so it is the offline, network-free default.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading.application.ports.clock import ClockPort
from trading.application.ports.market_data import Quote
from trading.domain.prices import PriceSeries
from trading.infrastructure.market_data.static import UnknownSymbolError

_DEFAULT_SYNTHETIC_BASE = Decimal("100")


class HistoricalMarketData:
    def __init__(self, prices: PriceSeries, clock: ClockPort) -> None:
        self._prices = prices
        self._clock = clock

    async def get_quote(self, symbol: str) -> Quote:
        now = self._clock.now()
        price = self._prices.price_at(symbol, now)
        if price is None:
            raise UnknownSymbolError(symbol)
        return Quote(symbol=symbol, bid=price, ask=price, last=price, as_of=now)


def synthetic_price_series(base: Mapping[str, Decimal], times: Sequence[datetime]) -> PriceSeries:
    """Deterministic random-walk-ish price path per symbol across `times` (illustrative)."""
    series: dict[str, tuple[tuple[datetime, Decimal], ...]] = {}
    for symbol, base_price in base.items():
        seed = sum(ord(c) for c in symbol)
        price = base_price
        points: list[tuple[datetime, Decimal]] = []
        for index, when in enumerate(times):
            # Bounded deterministic step in roughly [-2%, +2%].
            ratio = (((seed * 9301 + index * 49297) % 233280) / 233280.0 - 0.5) * 0.04
            price = (price * Decimal(str(1.0 + ratio))).quantize(Decimal("0.01"))
            points.append((when, price))
        series[symbol] = tuple(points)
    return PriceSeries(series=series)


def _daily_grid(start: datetime, end: datetime) -> list[datetime]:
    """Inclusive daily timestamps from ``start`` to ``end`` (at least the start point)."""
    if end < start:
        start, end = end, start
    times: list[datetime] = []
    when = start
    while when <= end:
        times.append(when)
        when += timedelta(days=1)
    return times or [start]


class SyntheticHistoricalPrices:
    """`HistoricalPricesPort` over `synthetic_price_series` — the offline default.

    Builds a deterministic illustrative path across a daily grid of the requested window, so
    backtests run with zero network calls. `base` supplies a per-symbol starting price (a
    flat default covers any symbol not listed).
    """

    def __init__(self, base: Mapping[str, Decimal] | None = None) -> None:
        self._base = dict(base or {})

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        wanted = [s.strip().upper() for s in symbols if s.strip()]
        if not wanted:
            return PriceSeries(series={})
        base = {s: self._base.get(s, _DEFAULT_SYNTHETIC_BASE) for s in wanted}
        return synthetic_price_series(base, _daily_grid(start, end))
