// Signal → event-log join (UI Phase C). GET /signals/feed's SignalFeedItem carries NO
// correlation id and NO provenance (interface/schemas.py) — both live on the `signal.received`
// event that recorded the signal (application/signal_recording.py's payload: signal_id,
// provenance, …). This pure helper indexes a loaded /events window by payload signal_id so the
// Signals grid can badge provenance and open the Chain Inspector on a row, and Review can open
// the chain of a proposal's contributing signal.
//
// Honest limitation: the join only covers signals whose `signal.received` event falls inside the
// loaded events window (GET /events caps at 1000, mixed event types). Rows outside it get
// meta = undefined — rendered as unknown provenance ("—") and no chain, never guessed.

import type { Provenance, StoredEventDto } from "./types";

export interface SignalMeta {
  correlationId: string;
  provenance: Provenance | null;
}

const PROVENANCE_VALUES: ReadonlySet<string> = new Set(["REAL", "BACKFILLED", "SYNTHETIC"]);

/** Index `signal.received` events by their payload signal_id. Later events win (idempotent
 *  re-records supersede), though in practice a signal is recorded exactly once. */
export function signalMetaFromEvents(
  events: readonly StoredEventDto[],
): Map<string, SignalMeta> {
  const meta = new Map<string, SignalMeta>();
  for (const e of events) {
    if (e.event_type !== "signal.received") continue;
    const signalId = e.payload.signal_id;
    if (typeof signalId !== "string" || signalId.length === 0) continue;
    const rawProvenance = e.payload.provenance;
    const provenance =
      typeof rawProvenance === "string" && PROVENANCE_VALUES.has(rawProvenance.toUpperCase())
        ? (rawProvenance.toUpperCase() as Provenance)
        : null;
    meta.set(signalId, { correlationId: e.correlation_id, provenance });
  }
  return meta;
}
