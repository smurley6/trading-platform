"""Data source port: every external provider is an adapter behind this (FR-ING-1)."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from trading.domain.signals import Signal


class DataSourcePort(Protocol):
    @property
    def name(self) -> str: ...

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        """Fetch and normalize new observations into Signals, retaining raw payloads."""
        ...
