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


class HistoricalDataSourcePort(Protocol):
    """A source that can reconstruct historical signals over a date window (the backfill port).

    Kept SEPARATE from :class:`DataSourcePort` so the live-ingest contract is untouched: a live
    adapter polls forward from a cursor (``fetch_signals``), whereas a backfill adapter walks an
    archive for a bounded ``[start, end)`` window (``fetch_historical``). Each returned Signal
    carries its source-native id in ``raw_payload["source_uid"]`` (EDGAR accession number, House
    DocID + row, …) and its ``created_at`` set to the PUBLIC disclosure timestamp — the backfill
    use case reads both. No live adapter needs to implement this; the historical fetchers
    (Tasks 04a/04b) add it, and the composition root exposes a registry of the ones that do.
    """

    @property
    def name(self) -> str: ...

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        """Reconstruct signals disclosed within ``[start, end)``, retaining raw payloads.

        Every Signal's ``created_at`` is the PUBLIC disclosure timestamp (never the underlying
        event/trade date), and ``raw_payload`` includes a ``source_uid`` for dedupe.
        """
        ...
