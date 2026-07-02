"""Data source port: every external provider is an adapter behind this (FR-ING-1)."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

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
    archive for a bounded ``[start, end]`` window, inclusive of both endpoints
    (``fetch_historical``). Each returned Signal carries its source-native id in
    ``raw_payload["source_uid"]`` (EDGAR accession number, House DocID + row, …) and its
    ``created_at`` set to the PUBLIC disclosure timestamp — the backfill use case reads both.
    No live adapter needs to implement this; the historical fetchers (Tasks 04a/04b) add it,
    and the composition root exposes a registry of the ones that do.
    """

    @property
    def name(self) -> str: ...

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        """Reconstruct signals disclosed in ``[start, end]`` (inclusive), keeping raw payloads.

        Every Signal's ``created_at`` is the PUBLIC disclosure timestamp (never the underlying
        event/trade date), and ``raw_payload`` includes a ``source_uid`` for dedupe.
        """
        ...


@runtime_checkable
class FilterableHistoricalSource(Protocol):
    """Optional capability: a historical source whose fetch volume is restricted by a filter
    (e.g. an issuer CIK allow-list) that can be toggled and described for auditability.

    Kept SEPARATE from :class:`HistoricalDataSourcePort` (rather than adding these methods to
    it directly) because not every historical source needs a filter — this is a capability a
    source MAY additionally implement, checked at runtime via ``isinstance`` against this
    ``runtime_checkable`` Protocol (no ``hasattr`` probing). Task 04a's ``SecForm4Source`` is the
    first implementer: the default-ON CIK filter restricts the EDGAR index walk to issuers whose
    tickers are in ``COMPANY_DIRECTORY``; ``--no-filter`` on the backfill job disables it. The
    job/use-case that drives a backfill run is responsible for calling ``set_filter_enabled``
    (from CLI flags) and recording ``filter_description()`` in the run's ``user.action`` event
    so the active filter is auditable per RESEARCH-PLAN §3/§4.
    """

    def set_filter_enabled(self, enabled: bool) -> None:
        """Turn the source's volume filter on or off for subsequent ``fetch_historical`` calls."""
        ...

    def filter_description(self) -> str | None:
        """A short human-readable description of the ACTIVE filter, or ``None`` if disabled."""
        ...
