"""Market data port."""

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
