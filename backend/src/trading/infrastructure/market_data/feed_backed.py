"""Bridge the live batch `QuoteFeedPort` onto the single-symbol `MarketDataPort`.

`MarketDataPort` is the seam the paper broker and the decision/risk path use to price a
symbol. By default that seam is static dev prices; this adapter routes it through the live
Alpaca feed so the strategy *decides and fills at real prices* when the feed is on, while
falling back to `fallback` whenever the feed is disabled or has no usable quote for a symbol
(off-hours, illiquid, or not on the IEX feed). The decision/fill path therefore never fails
just because one symbol has no live quote.
"""

from decimal import Decimal

from trading.application.ports.market_data import MarketDataPort, Quote, QuoteFeedPort


class FeedBackedMarketData:
    def __init__(self, feed: QuoteFeedPort, fallback: MarketDataPort) -> None:
        self._feed = feed
        self._fallback = fallback

    async def get_quote(self, symbol: str) -> Quote:
        if self._feed.enabled:
            quotes = await self._feed.get_quotes([symbol])
            # A non-positive `last` means the feed had no real price (e.g. empty book
            # off-hours); fall back rather than hand the risk gate an unusable price.
            if quotes and quotes[0].last > Decimal("0"):
                return quotes[0]
        return await self._fallback.get_quote(symbol)
