"""Static market data adapter for development and tests (deterministic quotes)."""

from decimal import Decimal

from trading.application.ports.clock import ClockPort
from trading.application.ports.market_data import Quote


class UnknownSymbolError(LookupError):
    def __init__(self, symbol: str) -> None:
        super().__init__(f"no quote configured for symbol '{symbol}'")


class StaticMarketData:
    def __init__(self, prices: dict[str, Decimal], clock: ClockPort) -> None:
        self._prices = dict(prices)
        self._clock = clock

    def set_price(self, symbol: str, price: Decimal) -> None:
        self._prices[symbol] = price

    async def get_quote(self, symbol: str) -> Quote:
        try:
            price = self._prices[symbol]
        except KeyError:
            raise UnknownSymbolError(symbol) from None
        return Quote(symbol=symbol, bid=price, ask=price, last=price, as_of=self._clock.now())
