"""IngestSignals use case (FR-ING): poll data sources, record signals as events.

Each source is a `DataSourcePort` adapter; adding or removing one changes nothing here
(FR-ING-1). A per-source cursor advances by the newest signal seen, so repeated polls do
not re-ingest the same observation. A failure in one source MUST NOT stop the others
(spec section 12, reliability), so each source is polled independently and its errors are
logged and isolated.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from trading.application.ports.clock import ClockPort
from trading.application.ports.data_source import DataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.application.signal_recording import record_signals
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    correlation_id: str
    signals: tuple[Signal, ...]
    failed_sources: tuple[str, ...]


class IngestSignals:
    def __init__(
        self,
        *,
        sources: Sequence[DataSourcePort],
        event_store: EventStorePort,
        clock: ClockPort,
        lookback_start: datetime,
        require_real_data: bool = False,
    ) -> None:
        self._sources = list(sources)
        self._event_store = event_store
        self._clock = clock
        # When True, SYNTHETIC signals are dropped before they ever enter the event log.
        self._require_real_data = require_real_data
        # Per-source cursor: only signals created strictly after this are ingested.
        self._cursors: dict[str, datetime] = {s.name: lookback_start for s in sources}

    async def execute(self) -> IngestionResult:
        correlation_id = str(uuid4())
        collected: list[Signal] = []
        failed: list[str] = []
        for source in self._sources:
            try:
                fetched = await source.fetch_signals(self._cursors[source.name])
            except Exception:
                logger.exception(
                    "data source failed; isolating and continuing",
                    extra={"correlation_id": correlation_id},
                )
                failed.append(source.name)
                continue
            fresh = [s for s in fetched if s.created_at > self._cursors[source.name]]
            if fresh:
                self._cursors[source.name] = max(s.created_at for s in fresh)
            collected.extend(fresh)

        # Ingestion gate: in real-data mode, non-REAL signals never enter the log.
        if self._require_real_data:
            real = [s for s in collected if s.provenance is Provenance.REAL]
            dropped = len(collected) - len(real)
            if dropped:
                logger.warning(
                    "dropped %d non-REAL signal(s) before recording (real-data mode)",
                    dropped,
                    extra={"correlation_id": correlation_id},
                )
            collected = real

        await record_signals(
            event_store=self._event_store,
            clock=self._clock,
            signals=collected,
            correlation_id=correlation_id,
        )
        return IngestionResult(
            correlation_id=correlation_id,
            signals=tuple(collected),
            failed_sources=tuple(failed),
        )
