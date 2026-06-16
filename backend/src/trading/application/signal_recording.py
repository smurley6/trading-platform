"""Recording signals as immutable events.

A signal is captured the moment it is observed — including below-threshold "watch" signals,
with their raw provider payload (spec section 9). Both the ingestion job and the manual
push endpoint funnel through here so emission lives in exactly one place. Evaluation reads
these events back; it never re-emits them.
"""

from collections.abc import Sequence

from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.domain.events import DomainEvent, EventType
from trading.domain.signals import Signal


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
    }


async def record_signals(
    *,
    event_store: EventStorePort,
    clock: ClockPort,
    signals: Sequence[Signal],
    correlation_id: str,
) -> None:
    for signal in signals:
        await event_store.append(
            DomainEvent(
                stream_id=f"signal-{signal.id}",
                event_type=EventType.SIGNAL_RECEIVED,
                payload=signal_payload(signal),
                occurred_at=clock.now(),
                correlation_id=correlation_id,
            )
        )
