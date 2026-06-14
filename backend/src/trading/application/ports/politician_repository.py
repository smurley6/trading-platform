"""Politician repository port: lookups for influence weighting (FR-POL-3)."""

from collections.abc import Sequence
from typing import Protocol

from trading.domain.politicians import Politician


class PoliticianRepositoryPort(Protocol):
    def get(self, politician_id: str) -> Politician | None: ...

    def find_in_text(self, text: str) -> Politician | None:
        """Return the first known politician whose name appears in the text, if any."""
        ...

    def all(self) -> Sequence[Politician]: ...
