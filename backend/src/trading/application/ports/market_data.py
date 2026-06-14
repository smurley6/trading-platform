"""Market data ports.

`MarketDataPort` is the single-symbol quote seam used by the paper broker and backtests.
`QuoteFeedPort` is a separate, additive seam for the live watchlist/tracking feed: it fetches
many symbols at once and is allowed to be a no-op (disabled) so the app always has something
injected and makes no network calls by default.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    as_of: datetime


class MarketDataPort(Protocol):
    async def get_quote(self, symbol: str) -> Quote: ...


class QuoteFeedPort(Protocol):
    """Live, batch quote feed for tracking/watchlist and live position marks.

    `enabled` lets the interface layer report a disabled feed (HTTP 200, empty quotes)
    instead of erroring, keeping the default-off contract.
    """

    @property
    def enabled(self) -> bool: ...

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]: ...
