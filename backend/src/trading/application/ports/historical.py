"""Historical prices port — the market-context source a backtest replays against.

Separate from the live `MarketDataPort`/`QuoteFeedPort` seams: a backtest does not want a
single latest quote, it wants a whole `PriceSeries` (per-symbol time series of closes) over a
date range. The composition root injects a concrete provider; the default is synthetic
(offline, no network), and a real Alpaca daily-bars adapter is wired only when explicitly
configured. Keeping this a port keeps the application layer pure — it never imports either
concrete implementation.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from trading.domain.prices import PriceSeries


class HistoricalPricesPort(Protocol):
    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        """Per-symbol price series over ``[start, end]``. Unknown symbols are simply absent."""
        ...
