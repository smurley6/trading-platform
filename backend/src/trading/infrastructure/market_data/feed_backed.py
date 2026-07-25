"""Bridge the live batch `QuoteFeedPort` onto the single-symbol `MarketDataPort`.

`MarketDataPort` is the seam the paper broker and the decision/risk path use to price a
symbol. By default that seam is static dev prices; this adapter routes it through the live
Alpaca feed so the strategy *decides and fills at real prices* when the feed is on.

Two modes, chosen by the composition root at wiring time (`main.py`):

- **Dev / display mode** (`fallback` is a `MarketDataPort`): when the feed is disabled or has
  no usable quote for a symbol (off-hours, illiquid, or not on the IEX feed), fall back to the
  static prices so the decision/fill path never fails just because one symbol has no live
  quote. Safe because no real broker is wired.

- **Real-data mode** (`fallback is None`, set whenever `require_real_data`): the static
  `_DEV_PRICES` fallback is NOT wired at all. A missing or non-positive live quote raises
  `NoRealQuoteError` instead of substituting a synthetic price — the decision path catches it
  and skips that symbol (review F4: a real account must never trade or mark against a guessed
  price, precisely in the pre-open window when the IEX book is often empty).
"""

from decimal import Decimal

from trading.application.ports.market_data import (
    MarketDataPort,
    NoRealQuoteError,
    Quote,
    QuoteFeedPort,
)


class FeedBackedMarketData:
    def __init__(self, feed: QuoteFeedPort, fallback: MarketDataPort | None) -> None:
        self._feed = feed
        # None => real-data mode: no synthetic fallback is wired, so no synthetic price can
        # ever reach a real order/mark. A real price is required or the symbol is skipped.
        self._fallback = fallback

    async def get_quote(self, symbol: str) -> Quote:
        if self._feed.enabled:
            quotes = await self._feed.get_quotes([symbol])
            # A non-positive `last` means the feed had no real price (e.g. empty book
            # off-hours); do not hand the risk gate an unusable price.
            if quotes and quotes[0].last > Decimal("0"):
                return quotes[0]
        if self._fallback is None:
            # Real-data mode: fail closed. No static price is substituted for a real order.
            raise NoRealQuoteError(symbol)
        return await self._fallback.get_quote(symbol)
