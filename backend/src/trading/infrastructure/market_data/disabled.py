"""Null QuoteFeedPort used when the live feed is not configured.

Default-off by construction: a bare `python -m trading.main` injects this so the app makes
no network calls. It reports `enabled == False` and returns no quotes, which the interface
layer turns into a benign `{"enabled": false, ...}` response rather than a 5xx.
"""

from collections.abc import Sequence

from trading.application.ports.market_data import Quote


class DisabledQuoteFeed:
    @property
    def enabled(self) -> bool:
        return False

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        return []
