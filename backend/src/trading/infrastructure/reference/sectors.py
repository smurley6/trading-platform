"""Ticker-to-sector classification backed by the seed company directory."""

from trading.domain.sectors import Sector
from trading.infrastructure.reference import COMPANY_DIRECTORY


class StaticSectorMap:
    def sector_for(self, ticker: str) -> Sector | None:
        entry = COMPANY_DIRECTORY.get(ticker.upper())
        return entry[0] if entry else None
