"""Market data ports.

`MarketDataPort` is the single-symbol quote seam used by the paper broker and backtests.
`QuoteFeedPort` is a separate, additive seam for the live watchlist/tracking feed: it fetches
many symbols at once and is allowed to be a no-op (disabled) so the app always has something
injected and makes no network calls by default.
`QuoteStreamPort` is the real-time push seam behind the `/ws/quotes` WebSocket: subscribers
fan out off per-connection queues (mirroring the event broadcaster) and may extend the
upstream symbol set. Like `QuoteFeedPort`, it is allowed to be a disabled no-op so the app
always has something injected and opens no socket by default.
"""

import asyncio
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


class QuoteStreamPort(Protocol):
    """Real-time, push-based quote source behind the `/ws/quotes` WebSocket.

    Subscribers register a per-connection queue and drain `Quote`s as they arrive, exactly
    like the event broadcaster's fan-out. `add_symbols` extends the upstream subscription so
    a client can ask for tickers beyond the platform's tracked set. `enabled` lets the
    interface report a disabled stream (status frame, no quotes) instead of erroring, keeping
    the default-off contract: the disabled implementation opens no socket and yields nothing.
    """

    @property
    def enabled(self) -> bool: ...

    def subscribe(self) -> "asyncio.Queue[Quote]":
        """Register a subscriber and return its queue of newly received quotes."""
        ...

    def unsubscribe(self, queue: "asyncio.Queue[Quote]") -> None: ...

    async def add_symbols(self, symbols: Sequence[str]) -> None:
        """Ensure `symbols` are part of the upstream subscription (idempotent)."""
        ...
