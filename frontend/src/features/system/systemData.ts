// System screen (UI Phase D, docs/UI-DESIGN.md §5.5) — pure projections over the event log for the
// Risk pane's kill-switch history and the Reconciliation pane's last-run + drift. Same idea as
// lib/projections.ts: current-state views derived from immutable events, unit-tested in isolation.
// The drawdown gauge reuses lib/projections' poolEquitySeries + maxDrawdownPct (no duplication).

import type { StoredEventDto } from "../../lib/types";

function str(value: unknown, fallback = ""): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
}

export interface KillSwitchChange {
  time: string;
  engaged: boolean;
  correlationId: string;
}

/** Kill-switch history (§5.5 Risk pane), newest-first, from `risk.kill_switch_changed` events. */
export function killSwitchHistory(events: StoredEventDto[]): KillSwitchChange[] {
  return events
    .filter((e) => e.event_type === "risk.kill_switch_changed")
    .map((e) => ({
      time: e.occurred_at,
      engaged: e.payload.engaged === true,
      correlationId: e.correlation_id,
    }))
    .reverse();
}

export interface ReconciliationDrift {
  kind: string;
  key: string;
  local: string;
  broker: string;
}

export interface ReconciliationSummary {
  /** Newest `reconciliation.completed` occurred_at, or null if reconciliation never ran. */
  lastRunAt: string | null;
  driftCount: number;
  checkedPositions: number;
  checkedOrders: number;
  /** The drift rows from the LATEST run only (matched by correlation id), not the whole log. */
  drifts: ReconciliationDrift[];
}

/** Reconciliation status (§5.5): the last completed run + its drift rows, from the event log. */
export function reconciliationSummary(events: StoredEventDto[]): ReconciliationSummary {
  let latestCompleted: StoredEventDto | null = null;
  for (const e of events) {
    if (e.event_type === "reconciliation.completed") {
      // Events arrive oldest-first; keep scanning so the last match is the newest run.
      latestCompleted = e;
    }
  }
  if (latestCompleted === null) {
    return { lastRunAt: null, driftCount: 0, checkedPositions: 0, checkedOrders: 0, drifts: [] };
  }
  const runCorrelation = latestCompleted.correlation_id;
  const drifts = events
    .filter(
      (e) => e.event_type === "reconciliation.drift" && e.correlation_id === runCorrelation,
    )
    .map((e) => ({
      kind: str(e.payload.kind),
      key: str(e.payload.key),
      local: str(e.payload.local),
      broker: str(e.payload.broker),
    }));
  return {
    lastRunAt: latestCompleted.occurred_at,
    driftCount: Number(str(latestCompleted.payload.drift_count, "0")),
    checkedPositions: Number(str(latestCompleted.payload.checked_positions, "0")),
    checkedOrders: Number(str(latestCompleted.payload.checked_orders, "0")),
    drifts,
  };
}
