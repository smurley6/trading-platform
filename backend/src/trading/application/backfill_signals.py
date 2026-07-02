"""BackfillSignals use case (Task 02, docs/RESEARCH-PLAN.md §2-§4).

Drives one historical-source's ``fetch_historical`` over a ``[start, end)`` window and records
the reconstructed signals as ``signal.received`` events, stamped for research correctness:

  * ``occurred_at`` of the event = the signal's PUBLIC disclosure timestamp (``created_at`` set
    by the fetcher), NOT the ingest wall-clock (RESEARCH-PLAN §2.1 — point-in-time integrity).
  * ``provenance = Provenance.BACKFILLED`` — real historical data that Task 01's execution wall
    keeps out of order placement while admitting it to projections/labeler/learner.
  * ``pit_confidence`` from the per-source tier map below (RESEARCH-PLAN §3), so the learner can
    weight or filter by timestamp trust.

Idempotency covers the EVENT LOG, not just the read model: the source-native id
(``raw_payload["source_uid"]`` — EDGAR accession number, House DocID + row, …) is checked
against :class:`~trading.application.ports.backfill_seen.BackfillSeenPort` BEFORE any event is
appended, so re-running a window double-appends nothing (projections and the future labeler
would otherwise double-count). ``source_uid`` also travels inside ``raw_payload``, so the dedupe
table is rebuildable from the append-only event log.

The use case receives ports (historical source, event store, dedupe store, clock) — never
concrete adapters; the composition root wires them (hexagonal boundary).
"""

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from trading.application.ports.backfill_seen import BackfillSeenPort
from trading.application.ports.clock import ClockPort
from trading.application.ports.data_source import HistoricalDataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.application.signal_recording import record_signals
from trading.domain.provenance import PitConfidence, Provenance
from trading.domain.signals import Signal

logger = logging.getLogger(__name__)

# Per-source point-in-time confidence (RESEARCH-PLAN §3 source tier table). The backfill use case
# stamps every fetched signal with its source's value; a source absent here cannot be backfilled.
# `news` is deliberately EXCLUDED (its provider serves a current index, not a dated archive), so
# it is not a key — attempting to backfill it fails cleanly.
BACKFILL_PIT_CONFIDENCE: dict[str, PitConfidence] = {
    "sec_form4": "exact",
    "sec_edgar_atom": "exact",
    "house_disclosures": "exact",
    "federal_register": "exact",
    "congress_bills": "exact",
    "clinical_trials": "exact",
    "senate_efd": "approximate",
    "congress_trades": "approximate",
    "lobbying": "approximate",
    "fda_approvals": "approximate",
    "regulations_gov": "approximate",
    "usaspending": "approximate",
}


class BackfillError(Exception):
    """A backfill run could not proceed (unknown source, unsupported source, or bad window)."""


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """Per-run outcome: what was fetched and (unless dry-run) what was recorded."""

    source_id: str
    start: datetime
    end: datetime
    correlation_id: str | None  # None on a dry-run (no events written)
    fetched: int
    new: int
    duplicate: int
    refused: int  # fetched signals with no usable source_uid (cannot be deduped → not recorded)
    committed: bool


def pit_confidence_for(source_id: str) -> PitConfidence:
    """The point-in-time tier for a backfillable source, or raise if it cannot be backfilled."""
    try:
        return BACKFILL_PIT_CONFIDENCE[source_id]
    except KeyError:
        supported = ", ".join(sorted(BACKFILL_PIT_CONFIDENCE))
        raise BackfillError(
            f"source '{source_id}' cannot be backfilled (no point-in-time tier); "
            f"backfillable sources: {supported}"
        ) from None


def _source_uid(signal: Signal) -> str | None:
    uid = signal.raw_payload.get("source_uid")
    return str(uid) if uid not in (None, "") else None


class BackfillSignals:
    """Backfill one historical source over a window, idempotently and point-in-time-correctly."""

    def __init__(
        self,
        *,
        source: HistoricalDataSourcePort,
        event_store: EventStorePort,
        seen: BackfillSeenPort,
        clock: ClockPort,
    ) -> None:
        self._source = source
        self._event_store = event_store
        self._seen = seen
        self._clock = clock
        # Fail fast if this source has no PIT tier (i.e. is not backfillable, e.g. news).
        self._pit_confidence = pit_confidence_for(source.name)

    async def execute(
        self, *, start: datetime, end: datetime, correlation_id: str, dry_run: bool
    ) -> BackfillReport:
        """Fetch the window, stamp research invariants, dedupe, and (unless dry_run) record.

        Returns a :class:`BackfillReport` with per-outcome counts. On a dry run NO events are
        appended and NO ids are marked seen, but the same dedupe check runs so the report's
        ``new``/``duplicate`` split reflects exactly what a ``--confirm`` run would write.
        """
        fetched = list(await self._source.fetch_historical(start, end))
        stamped = [self._stamp(s) for s in fetched]

        # Refuse signals with no usable source_uid: without one they cannot be deduped, so
        # recording them would break idempotency on a re-run. Log + count, do not record.
        recordable: list[Signal] = []
        refused = 0
        for signal in stamped:
            if _source_uid(signal) is None:
                refused += 1
                logger.warning(
                    "backfill: dropping signal with no source_uid (cannot dedupe)",
                    extra={"correlation_id": correlation_id},
                )
                continue
            recordable.append(signal)

        # De-duplicate within this batch first (a fetcher may emit the same uid twice), then
        # against the persistent seen-set — the event-log-level idempotency check.
        by_uid: dict[str, Signal] = {}
        for signal in recordable:
            uid = _source_uid(signal)
            assert uid is not None  # guaranteed by the refused filter above
            by_uid.setdefault(uid, signal)  # keep the first occurrence of a repeated uid
        already = await self._seen.seen(self._source.name, list(by_uid))
        new_signals = [s for uid, s in by_uid.items() if uid not in already]
        # Intra-batch repeats + cross-run repeats both count as duplicates against `fetched`.
        duplicate = len(recordable) - len(new_signals)

        committed_correlation: str | None = None
        if not dry_run and new_signals:
            # One event per signal so occurred_at is that signal's own disclosure timestamp
            # (record_signals stamps every event in a call with the single override passed).
            for signal in new_signals:
                await record_signals(
                    event_store=self._event_store,
                    clock=self._clock,
                    signals=[signal],
                    correlation_id=correlation_id,
                    occurred_at=signal.created_at,
                )
            await self._seen.mark(self._source.name, [_uid(s) for s in new_signals])
            committed_correlation = correlation_id

        return BackfillReport(
            source_id=self._source.name,
            start=start,
            end=end,
            correlation_id=committed_correlation,
            fetched=len(fetched),
            new=len(new_signals),
            duplicate=duplicate,
            refused=refused,
            committed=not dry_run and bool(new_signals),
        )

    def _stamp(self, signal: Signal) -> Signal:
        """Enforce the research invariants on a fetched signal (use case, not the adapter)."""
        return dataclasses.replace(
            signal,
            provenance=Provenance.BACKFILLED,
            pit_confidence=self._pit_confidence,
        )


def _uid(signal: Signal) -> str:
    uid = _source_uid(signal)
    assert uid is not None  # only ever called on signals that passed the refused filter
    return uid


def supported_sources(sources: Sequence[HistoricalDataSourcePort]) -> list[str]:
    """The names of the wired historical sources, sorted, for the job's error message."""
    return sorted(s.name for s in sources)
