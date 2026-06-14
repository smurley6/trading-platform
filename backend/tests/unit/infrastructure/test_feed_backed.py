"""FeedBackedMarketData: live feed for decision/fill prices, static fallback otherwise."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.application.ports.market_data import Quote
from trading.infrastructure.market_data.feed_backed import FeedBackedMarketData

_NOW = datetime(2026, 6, 12, tzinfo=UTC)


def _quote(symbol: str, last: str) -> Quote:
    return Quote(
        symbol=symbol,
        bid=Decimal(last),
        ask=Decimal(last),
        last=Decimal(last),
        as_of=_NOW,
    )


class _FakeFeed:
    def __init__(self, *, enabled: bool, quotes: dict[str, Quote]) -> None:
        self._enabled = enabled
        self._quotes = quotes

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        return [self._quotes[s] for s in symbols if s in self._quotes]


class _FakeFallback:
    async def get_quote(self, symbol: str) -> Quote:
        return _quote(symbol, "200")


async def test_uses_live_feed_when_enabled_and_priced() -> None:
    bridge = FeedBackedMarketData(
        _FakeFeed(enabled=True, quotes={"AAPL": _quote("AAPL", "290.22")}),
        _FakeFallback(),
    )
    assert (await bridge.get_quote("AAPL")).last == Decimal("290.22")


async def test_falls_back_when_feed_disabled() -> None:
    bridge = FeedBackedMarketData(
        _FakeFeed(enabled=False, quotes={"AAPL": _quote("AAPL", "290.22")}),
        _FakeFallback(),
    )
    assert (await bridge.get_quote("AAPL")).last == Decimal("200")


async def test_falls_back_when_feed_has_no_quote() -> None:
    bridge = FeedBackedMarketData(_FakeFeed(enabled=True, quotes={}), _FakeFallback())
    assert (await bridge.get_quote("OBSCURE")).last == Decimal("200")


@pytest.mark.parametrize("bad_last", ["0", "-1"])
async def test_falls_back_on_nonpositive_price(bad_last: str) -> None:
    bridge = FeedBackedMarketData(
        _FakeFeed(enabled=True, quotes={"AAPL": _quote("AAPL", bad_last)}),
        _FakeFallback(),
    )
    assert (await bridge.get_quote("AAPL")).last == Decimal("200")
