// Client-side projections over the event log — the same idea as the backend's
// application/projections.py: current-state views derived from immutable events.

import type { BacktestPoint, StoredEventDto } from "./types";

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

/**
 * Normalize a backtest curve (strategy or benchmark) into EquityPoint[] for the EquityCurve
 * chart. Accepts both the legacy `{ time: <unix seconds>, equity }` and the new
 * `{ t: <ISO timestamp>, equity }` point shapes, coerces `equity` from string-or-number, and
 * forces strictly increasing timestamps (lightweight-charts requires it; backtest snapshots can
 * share a timestamp). Points whose time/equity can't be parsed are skipped.
 */
export function backtestCurve(points: BacktestPoint[] | undefined): EquityPoint[] {
  const out: EquityPoint[] = [];
  let lastTime = 0;
  for (const p of points ?? []) {
    const rawTime =
      typeof p.time === "number" ? p.time : p.t ? Math.floor(new Date(p.t).getTime() / 1000) : NaN;
    const value = Number(p.equity);
    if (!Number.isFinite(rawTime) || !Number.isFinite(value)) continue;
    let time = rawTime;
    if (time <= lastTime) time = lastTime + 1;
    lastTime = time;
    out.push({ time, value });
  }
  return out;
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

/**
 * Pool-level equity curve (Overview §5.1: "Equity curve … pool-level") — the shared capital pool
 * is the SUM of every sleeve's equity (MEMORY.md §5.2: "one pool = the one Alpaca book"), and
 * `portfolio.snapshot` events are recorded per-sleeve, so this sums each sleeve's latest-known
 * equity as of every snapshot timestamp across all sleeves (a running "last value carried
 * forward" per sleeve, summed at each distinct timestamp) rather than one sleeve's series alone.
 */
export function poolEquitySeries(events: StoredEventDto[]): EquityPoint[] {
  const snapshots = events
    .filter((e) => e.event_type === "portfolio.snapshot")
    .map((e) => ({
      sleeveId: str(e.payload.sleeve_id),
      equity: Number(str(e.payload.equity, "0")),
      time: new Date(e.occurred_at).getTime(),
    }))
    .filter((s) => s.sleeveId && Number.isFinite(s.equity) && Number.isFinite(s.time))
    .sort((a, b) => a.time - b.time);

  const latestBySleeve = new Map<string, number>();
  const points: EquityPoint[] = [];
  let lastTime = 0;
  for (const s of snapshots) {
    latestBySleeve.set(s.sleeveId, s.equity);
    const total = [...latestBySleeve.values()].reduce((sum, v) => sum + v, 0);
    let time = Math.floor(s.time / 1000);
    if (time <= lastTime) time = lastTime + 1;
    lastTime = time;
    points.push({ time, value: total });
  }
  return points;
}

/** Largest peak-to-trough decline over an equity series, as a positive percentage — mirrors the
 *  backend's pure `domain/metrics.py::max_drawdown_pct` exactly (same peak-tracking algorithm),
 *  applied client-side to the pool equity curve since there is no live drawdown endpoint (only a
 *  backtest-scoped one) and this needs zero backend change. */
export function maxDrawdownPct(points: EquityPoint[]): number {
  let peak: number | null = null;
  let worst = 0;
  for (const p of points) {
    if (peak === null || p.value > peak) peak = p.value;
    if (peak !== null && peak > 0) {
      const drawdown = ((peak - p.value) / peak) * 100;
      if (drawdown > worst) worst = drawdown;
    }
  }
  return worst;
}

/** Day P&L: latest pool equity minus the last equity value recorded before local midnight today —
 *  or the earliest point of the day if trading started after midnight. Zero when there's under
 *  two points (nothing to diff against yet). */
export function dayPnl(points: EquityPoint[]): number {
  if (points.length < 2) return 0;
  const latest = points[points.length - 1]!;
  const startOfDay = new Date(latest.time * 1000);
  startOfDay.setHours(0, 0, 0, 0);
  const startOfDaySec = Math.floor(startOfDay.getTime() / 1000);

  let baseline = points[0]!;
  for (const p of points) {
    if (p.time <= startOfDaySec) {
      baseline = p;
    } else {
      break;
    }
  }
  return latest.value - baseline.value;
}
