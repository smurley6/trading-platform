"""Reference-data ports: ticker-to-sector classification."""

from typing import Protocol

from trading.domain.sectors import Sector


class SectorMapPort(Protocol):
    def sector_for(self, ticker: str) -> Sector | None:
        """Classify a ticker into a sector, or None if unknown."""
        ...
