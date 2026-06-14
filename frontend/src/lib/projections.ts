// Client-side projections over the event log — the same idea as the backend's
// application/projections.py: current-state views derived from immutable events.

import type { StoredEventDto } from "./types";

export interface BlotterRow {
  occurred_at: string;
  client_order_id: string;
  sleeve_id: string;
  symbol: string;
  side: string;
  quantity: string;
  status: string;
  broker_order_id: string;
}

export interface PositionRow {
  sleeve_id: string;
  symbol: string;
  quantity: number;
}

export interface RiskRow {
  occurred_at: string;
  decision_id: string;
  verdict: string;
  reason: string;
  requested_quantity: string;
  approved_quantity: string;
}

export interface EquityPoint {
  time: number; // unix seconds
  value: number;
}

function str(value: unknown, fallback = ""): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
    return String(value);
  }
  return JSON.stringify(value) ?? fallback;
}

/** Latest lifecycle state per client_order_id, newest-first for display. */
export function blotterRows(events: StoredEventDto[]): BlotterRow[] {
  const byOrder = new Map<string, BlotterRow>();
  for (const e of events) {
    const p = e.payload;
    const id = str(p.client_order_id);
    if (!id) continue;
    if (e.event_type === "order.intent") {
      byOrder.set(id, {
        occurred_at: e.occurred_at,
        client_order_id: id,
        sleeve_id: str(p.sleeve_id),
        symbol: str(p.symbol),
        side: str(p.side),
        quantity: str(p.quantity),
        status: "intent",
        broker_order_id: "",
      });
    } else if (e.event_type === "order.submitted") {
      const row = byOrder.get(id);
      if (row) {
        row.status = "submitted";
        row.broker_order_id = str(p.broker_order_id);
      }
    } else if (e.event_type === "order.status_changed") {
      const row = byOrder.get(id);
      if (row) {
        row.status = str(p.status, row.status);
        row.broker_order_id = str(p.broker_order_id, row.broker_order_id);
      }
    } else if (e.event_type === "order.submit_failed") {
      const row = byOrder.get(id);
      if (row) row.status = "submit_failed";
    }
  }
  return [...byOrder.values()].reverse();
}

/** Net position per (sleeve, symbol) from recorded fills. */
export function positionRows(events: StoredEventDto[]): PositionRow[] {
  const totals = new Map<string, PositionRow>();
  for (const e of events) {
    if (e.event_type !== "order.filled") continue;
    const p = e.payload;
    const sleeve = str(p.sleeve_id);
    const symbol = str(p.symbol);
    const qty = Number(str(p.quantity, "0")) * (p.side === "sell" ? -1 : 1);
    const key = `${sleeve}|${symbol}`;
    const row = totals.get(key) ?? { sleeve_id: sleeve, symbol, quantity: 0 };
    row.quantity += qty;
    totals.set(key, row);
  }
  return [...totals.values()].filter((r) => r.quantity !== 0);
}

export function riskRows(events: StoredEventDto[]): RiskRow[] {
  return events
    .filter((e) => e.event_type === "risk.evaluated")
    .map((e) => ({
      occurred_at: e.occurred_at,
      decision_id: str(e.payload.decision_id),
      verdict: str(e.payload.verdict),
      reason: str(e.payload.reason),
      requested_quantity: str(e.payload.requested_quantity),
      approved_quantity: str(e.payload.approved_quantity),
    }))
    .reverse();
}

export interface SleeveSnapshot {
  sleeve_id: string;
  cash: number;
  equity: number;
  positionCount: number;
}

/** The most recent portfolio snapshot per sleeve. */
export function latestSnapshots(events: StoredEventDto[]): Map<string, SleeveSnapshot> {
  const latest = new Map<string, SleeveSnapshot>();
  for (const e of events) {
    if (e.event_type !== "portfolio.snapshot") continue;
    const p = e.payload;
    const positions = (p.positions ?? {}) as Record<string, unknown>;
    latest.set(str(p.sleeve_id), {
      sleeve_id: str(p.sleeve_id),
      cash: Number(str(p.cash, "0")),
      equity: Number(str(p.equity, "0")),
      positionCount: Object.keys(positions).length,
    });
  }
  return latest;
}

/** Equity curve for one sleeve from its portfolio snapshots, oldest-first, de-duplicated. */
export function equitySeries(events: StoredEventDto[], sleeveId: string): EquityPoint[] {
  const points: EquityPoint[] = [];
  let lastTime = 0;
  for (const e of events) {
    if (e.event_type !== "portfolio.snapshot") continue;
    if (str(e.payload.sleeve_id) !== sleeveId) continue;
    const value = Number(str(e.payload.equity, "0"));
    let time = Math.floor(new Date(e.occurred_at).getTime() / 1000);
    if (time <= lastTime) time = lastTime + 1; // lightweight-charts requires strictly increasing time
    lastTime = time;
    points.push({ time, value });
  }
  return points;
}
