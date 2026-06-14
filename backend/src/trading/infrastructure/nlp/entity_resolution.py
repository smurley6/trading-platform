"""Gazetteer entity resolution (FR-POL-1): company mentions and cashtags -> tickers.

Deterministic and offline. Matches the seed company directory's aliases and `$TICKER`
cashtags. Word-boundary matching avoids spurious hits inside longer words.
"""

import re
from collections.abc import Sequence

from trading.infrastructure.reference import COMPANY_DIRECTORY

_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")


class GazetteerEntityResolver:
    def __init__(self) -> None:
        # alias -> ticker, plus a compiled word-boundary pattern per alias.
        self._aliases: list[tuple[re.Pattern[str], str]] = []
        for ticker, (_sector, aliases) in COMPANY_DIRECTORY.items():
            for alias in aliases:
                pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                self._aliases.append((pattern, ticker))
        self._known_tickers = set(COMPANY_DIRECTORY)

    def resolve_tickers(self, text: str) -> Sequence[str]:
        found: list[str] = []
        for pattern, ticker in self._aliases:
            if pattern.search(text) and ticker not in found:
                found.append(ticker)
        for match in _CASHTAG.finditer(text):
            ticker = match.group(1)
            if ticker in self._known_tickers and ticker not in found:
                found.append(ticker)
        return found
