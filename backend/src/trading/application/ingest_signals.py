"""IngestSignals use case (FR-ING): poll data sources, record signals as events.

Each source is a `DataSourcePort` adapter; adding or removing one changes nothing here
(FR-ING-1). A per-source cursor advances by the newest signal seen, so repeated polls do
not re-ingest the same observation. A failure in one source MUST NOT stop the others
(spec section 12, reliability), so each source is polled independently and its errors are
logged and isolated.

Cursors are PERSISTED in the event log (review F1). Each successful ingest appends an
``ingest.cursor_advanced`` event per advanced source; the next process (the daily cron is a
fresh process every run) rehydrates each source's cursor from the latest such event before
polling, so it resumes where the last run left off instead of re-scanning — and re-appending —
the whole 21-day lookback. The event log is the existing system of record, so this needs no
new table or migration and mirrors the kill-switch/peak-equity replay-at-startup idiom.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from trading.application.ports.clock import ClockPort
from trading.application.ports.data_source import DataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.application.signal_recording import record_signals
from trading.domain.events import DomainEvent, EventType
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal

logger = logging.getLogger(__name__)

# Some data-source adapters (congress_bills, fda_approvals, federal_register, congress_trades)
# authenticate with an API key passed as a URL query parameter, because their upstreams
# (api.data.gov / openFDA / FMP) accept the key only that way. On a routine 429/5xx, httpx raises
# an HTTPStatusError whose message embeds the FULL request URL — key and all. Logging that verbatim
# would persist the secret in Render's logs (review F22). We redact key-bearing query params from
# any message BEFORE it reaches a log record, and never log the raw traceback (which carries the
# same URL). The owner still rotates any key that a prior deploy may already have leaked.
_SECRET_QUERY_PARAM_RE = re.compile(
    r"(?i)([?&](?:api_?key|apikey|key|secret|token|password|access_token)=)[^&\s\"'>]+"
)


def _redact_secrets(message: str) -> str:
    """Redact key-bearing URL query-parameter VALUES from a log message (review F22)."""
    return _SECRET_QUERY_PARAM_RE.sub(r"\1REDACTED", message)


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
        # Per-source cursor: only signals created strictly after this are ingested. Seeded to
        # the lookback floor (the per-source resume point when no cursor is persisted yet) and
        # rehydrated from the persisted cursor events on first execute.
        self._cursors: dict[str, datetime] = {s.name: lookback_start for s in sources}
        self._hydrated = False

    async def _hydrate_cursors(self) -> None:
        """Seed each source's cursor from its latest persisted ``ingest.cursor_advanced`` event.

        Runs once per process (the cron's single run, or the first scheduled ingest of the
        always-on API); afterwards cursors advance in memory and via freshly appended events.
        A source with no persisted cursor keeps the lookback floor. The persisted value only
        ever moves the cursor FORWARD (``max`` against the floor), so a resumed cron never
        re-scans behind where it durably left off.
        """
        if self._hydrated:
            return
        self._hydrated = True
        latest: dict[str, datetime] = {}
        for stored in await self._event_store.read_all():
            if stored.event.event_type != EventType.INGEST_CURSOR_ADVANCED:
                continue
            payload = stored.event.payload
            source = str(payload["source"])
            cursor = datetime.fromisoformat(str(payload["cursor"]))
            # read_all is in recording order, so the last event per source is the newest;
            # max() is belt-and-suspenders against any out-of-order replay.
            latest[source] = max(cursor, latest[source]) if source in latest else cursor
        for name, cursor in latest.items():
            if name in self._cursors:
                self._cursors[name] = max(self._cursors[name], cursor)

    async def execute(self) -> IngestionResult:
        correlation_id = str(uuid4())
        await self._hydrate_cursors()
        collected: list[Signal] = []
        failed: list[str] = []
        advanced: dict[str, datetime] = {}
        for source in self._sources:
            try:
                fetched = await source.fetch_signals(self._cursors[source.name])
            except Exception as exc:
                # Redact any key-bearing URL and do NOT attach the raw traceback (exc_info): an
                # httpx error message embeds the full request URL including a query-param API key
                # (review F22). The redacted message keeps the failure diagnosable (status, host,
                # path) without persisting the secret.
                logger.error(
                    "data source '%s' failed; isolating and continuing: %s",
                    source.name,
                    _redact_secrets(f"{type(exc).__name__}: {exc}"),
                    extra={"correlation_id": correlation_id},
                )
                failed.append(source.name)
                continue
            fresh = [s for s in fetched if s.created_at > self._cursors[source.name]]
            if fresh:
                new_cursor = max(s.created_at for s in fresh)
                self._cursors[source.name] = new_cursor
                advanced[source.name] = new_cursor
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

        # Append-time idempotency guard (review F1): a re-scan of the lookback never grows a
        # second copy of an already-recorded signal in the append-only log.
        await record_signals(
            event_store=self._event_store,
            clock=self._clock,
            signals=collected,
            correlation_id=correlation_id,
            skip_existing_streams=True,
        )
        # Persist the advanced cursors ONLY after the signals were successfully recorded, so a
        # crash before/into the append leaves the durable cursor untouched and the next run
        # re-fetches (the guard above then keeps the re-fetch idempotent).
        await self._persist_cursors(advanced, correlation_id)
        return IngestionResult(
            correlation_id=correlation_id,
            signals=tuple(collected),
            failed_sources=tuple(failed),
        )

    async def _persist_cursors(self, advanced: dict[str, datetime], correlation_id: str) -> None:
        """Append one ``ingest.cursor_advanced`` watermark per source that moved this run."""
        for name, cursor in advanced.items():
            await self._event_store.append(
                DomainEvent(
                    stream_id=f"ingest-cursor-{name}",
                    event_type=EventType.INGEST_CURSOR_ADVANCED,
                    payload={"source": name, "cursor": cursor.isoformat()},
                    occurred_at=self._clock.now(),
                    correlation_id=correlation_id,
                )
            )
