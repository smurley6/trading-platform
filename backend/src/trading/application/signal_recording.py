"""Recording signals as immutable events.

A signal is captured the moment it is observed — including below-threshold "watch" signals,
with their raw provider payload (spec section 9). Both the ingestion job and the manual
push endpoint funnel through here so emission lives in exactly one place. Evaluation reads
these events back; it never re-emits them.
"""

import logging
from collections.abc import Sequence
from datetime import datetime

from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.domain.events import DomainEvent, EventType
from trading.domain.signals import Signal

logger = logging.getLogger(__name__)


def signal_payload(signal: Signal) -> dict[str, object]:
    return {
        "signal_id": signal.id,
        "created_at": signal.created_at.isoformat(),
        "source_type": signal.source_type.value,
        "ticker": signal.ticker,
        "politician_id": signal.politician_id,
        "sentiment": signal.sentiment,
        "politician_weight": signal.politician_weight,
        "source_reliability": signal.source_reliability,
        "raw_payload": signal.raw_payload,
        "provenance": signal.provenance.value,
        "pit_confidence": signal.pit_confidence,
    }


async def record_signals(
    *,
    event_store: EventStorePort,
    clock: ClockPort,
    signals: Sequence[Signal],
    correlation_id: str,
    occurred_at: datetime | None = None,
    skip_existing_streams: bool = False,
) -> None:
    """Append one ``signal.received`` event per signal (the single emission path).

    ``occurred_at`` defaults to ``clock.now()`` — the moment the signal was OBSERVED — for live
    ingestion. Backfill overrides it with the true public disclosure timestamp so the event's
    ``occurred_at`` is point-in-time correct, never the ingest wall-clock (RESEARCH-PLAN §2). The
    override applies to every signal in this call, so a backfill caller records one signal at a
    time with that signal's own ``created_at`` (its disclosure timestamp).

    ``skip_existing_streams`` (opt-in; default OFF) is the append-time idempotency guard for the
    LIVE ingest path (review F1): when set, a signal whose ``signal-{id}`` stream already holds an
    event is skipped rather than re-appended, so a re-scan of the lookback window never grows a
    second copy in the log. It is DEFAULT OFF so the historical backfill (which has its own
    ``backfill_seen`` dedupe), the backtest recorder, and the manual push endpoint keep byte-for-
    byte identical behaviour — only ``IngestSignals`` opts in.
    """
    stamp = clock.now() if occurred_at is None else occurred_at
    for signal in signals:
        stream_id = f"signal-{signal.id}"
        if skip_existing_streams and await event_store.read_stream(stream_id):
            logger.debug(
                "skipping already-recorded signal stream %s (idempotent append)", stream_id
            )
            continue
        await event_store.append(
            DomainEvent(
                stream_id=stream_id,
                event_type=EventType.SIGNAL_RECEIVED,
                payload=signal_payload(signal),
                occurred_at=stamp,
                correlation_id=correlation_id,
            )
        )
