"""Null market-data adapters used when the live feed/stream is not configured.

Default-off by construction: a bare `python -m trading.main` injects these so the app makes
no network or socket calls. Each reports `enabled == False`, which the interface layer turns
into a benign disabled response (`{"enabled": false, ...}` / a `{"type":"status",...}` frame)
rather than a 5xx.
"""

import asyncio
from collections.abc import Sequence

from trading.application.ports.market_data import Quote


class DisabledQuoteFeed:
    @property
    def enabled(self) -> bool:
        return False

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        return []


class DisabledQuoteStream:
    """Null QuoteStreamPort: never connects, never yields. The default for `/ws/quotes`.

    `subscribe` still hands back a real queue so the endpoint code path is uniform, but no
    producer ever pushes to it; `add_symbols` is a no-op. Importing or constructing this
    opens no socket and makes no network call.
    """

    @property
    def enabled(self) -> bool:
        return False

    def subscribe(self) -> "asyncio.Queue[Quote]":
        return asyncio.Queue()

    def unsubscribe(self, queue: "asyncio.Queue[Quote]") -> None:
        return None

    async def add_symbols(self, symbols: Sequence[str]) -> None:
        return None
