// Pure, client-side analytics for the Algorithms screen, derived entirely from the immutable
// event log (`api.events`) plus current conviction (`api.conviction`). No React, no I/O — just
// typed transforms so the screen stays declarative and these derivations stay unit-testable.
//
// Event payload shapes used here (see backend application/{evaluate_signals,place_order,
// signal_recording}.py and domain/events.py):
//   decision.made    -> { decision_id, sleeve_id, symbol, side, quantity, conviction_score,
//                         contributing_signal_ids, rationale, ... }
//   risk.evaluated   -> { decision_id, verdict, ... }
//   order.filled     -> { sleeve_id, symbol, side, price, quantity, fees, filled_at }
//   signal.received  -> { signal_id, source_type, ticker, sentiment, politician_id, created_at }
//   portfolio.snapshot -> { sleeve_id, equity, ... }
//
// All money/quantity fields are persisted as strings; we parse defensively with Number(...).

import type { ConvictionItem, StoredEventDto } from "../../lib/types";
import { equitySeries } from "../../lib/projections";

// --- small parsing helpers ---------------------------------------------------------------

function str(value: unknown, fallback = ""): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
    return String(value);
  }
  return fallback;
}

function num(value: unknown, fallback = 0): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }
  return fallback;
}

function strArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === "string");
}

// --- performance stats -------------------------------------------------------------------

export interface AlgoStats {
  /** Total return % over the sleeve's recorded equity history (first → last snapshot). */
  totalReturnPct: number;
  /** Worst peak-to-trough drawdown across the equity history, as a positive %. */
  maxDrawdownPct: number;
  /** Number of decisions the strategy produced for this sleeve. */
  decisionCount: number;
  /** Number of recorded fills for this sleeve. */
  fillCount: number;
  /**
   * Win rate over *closing* trades (fills that reduce an open position), i.e. the share of
   * realized round-trips that booked a non-negative realized P&L. Null when nothing has been
   * closed yet so the UI can show "—" rather than a misleading 0%.
   */
  winRatePct: number | null;
  /** Realized P&L from average-cost matching of fills (fees included). */
  realizedPnl: number;
  /** Count of realized (closing) trades that fed the win rate. */
  closedTrades: number;
}

/** Per-(sleeve, symbol) average-cost lot used to realize P&L as positions are reduced. */
interface Lot {
  qty: number; // signed: + long, - short
  avgPrice: number;
}

/**
 * Compute per-algorithm performance stats for one sleeve from the event log. Drawdown and
 * total return come from portfolio snapshots; realized P&L and win rate come from matching
 * fills (average cost), so a profitable close counts as a win.
 */
export function algoStats(events: StoredEventDto[], sleeveId: string): AlgoStats {
  const series = equitySeries(events, sleeveId);
  const first = series[0]?.value ?? 0;
  const last = series[series.length - 1]?.value ?? first;
  const totalReturnPct = first > 0 ? ((last - first) / first) * 100 : 0;

  let peak = first;
  let maxDrawdownPct = 0;
  for (const point of series) {
    if (point.value > peak) peak = point.value;
    if (peak > 0) {
      const dd = ((peak - point.value) / peak) * 100;
      if (dd > maxDrawdownPct) maxDrawdownPct = dd;
    }
  }

  let decisionCount = 0;
  let fillCount = 0;
  let realizedPnl = 0;
  let closedTrades = 0;
  let winningTrades = 0;
  const lots = new Map<string, Lot>();

  for (const e of events) {
    const p = e.payload;
    if (e.event_type === "decision.made" && str(p.sleeve_id) === sleeveId) {
      decisionCount += 1;
      continue;
    }
    if (e.event_type !== "order.filled" || str(p.sleeve_id) !== sleeveId) continue;

    fillCount += 1;
    const symbol = str(p.symbol);
    const price = num(p.price);
    const fees = num(p.fees);
    const signedQty = num(p.quantity) * (str(p.side) === "sell" ? -1 : 1);
    const lot = lots.get(symbol) ?? { qty: 0, avgPrice: 0 };

    const reducing = lot.qty !== 0 && Math.sign(signedQty) !== Math.sign(lot.qty);
    if (reducing) {
      // Realize against the existing average cost for the portion that closes.
      const closeQty = Math.min(Math.abs(signedQty), Math.abs(lot.qty));
      const direction = lot.qty > 0 ? 1 : -1; // long closes on a sell, short on a buy
      const pnl = (price - lot.avgPrice) * closeQty * direction - fees;
      realizedPnl += pnl;
      closedTrades += 1;
      if (pnl >= 0) winningTrades += 1;

      const remaining = lot.qty + signedQty;
      if (Math.sign(remaining) === Math.sign(lot.qty) || remaining === 0) {
        lot.qty = remaining; // partial/full close, same side avg price unchanged
      } else {
        // Flipped through zero: the overshoot opens a new position at this fill price.
        lot.qty = remaining;
        lot.avgPrice = price;
      }
    } else {
      // Opening or adding to the position: blend into the average cost.
      const totalQty = lot.qty + signedQty;
      if (totalQty !== 0) {
        lot.avgPrice =
          (lot.avgPrice * Math.abs(lot.qty) + price * Math.abs(signedQty)) / Math.abs(totalQty);
      }
      lot.qty = totalQty;
      realizedPnl -= fees;
    }
    lots.set(symbol, lot);
  }

  return {
    totalReturnPct,
    maxDrawdownPct,
    decisionCount,
    fillCount,
    winRatePct: closedTrades > 0 ? (winningTrades / closedTrades) * 100 : null,
    realizedPnl,
    closedTrades,
  };
}

// --- conviction breakdown ----------------------------------------------------------------

export interface ConvictionRow {
  ticker: string;
  score: number;
  action: string;
  /** Normalized 0..1 score used to draw the in-cell bar (relative to the strongest score). */
  scorePct: number;
  /** True when this ticker has been acted on by the sleeve (a decision exists for it). */
  traded: boolean;
}

/**
 * Build a conviction breakdown for one algorithm. Conviction is a global per-ticker view, so
 * we surface the tickers this sleeve has actually decided on first (flagged `traded`), then
 * the rest, and normalize scores for an in-grid bar.
 */
export function convictionBreakdown(
  conviction: ConvictionItem[],
  events: StoredEventDto[],
  sleeveId: string,
): ConvictionRow[] {
  const tradedTickers = new Set<string>();
  for (const e of events) {
    if (e.event_type === "decision.made" && str(e.payload.sleeve_id) === sleeveId) {
      const t = str(e.payload.symbol);
      if (t) tradedTickers.add(t);
    }
  }

  const maxScore = conviction.reduce((m, c) => Math.max(m, Math.abs(c.score)), 0);
  const rows: ConvictionRow[] = conviction.map((c) => ({
    ticker: c.ticker,
    score: c.score,
    action: c.action,
    scorePct: maxScore > 0 ? Math.abs(c.score) / maxScore : 0,
    traded: tradedTickers.has(c.ticker),
  }));

  // Traded tickers first, then by descending score.
  rows.sort((a, b) => {
    if (a.traded !== b.traded) return a.traded ? -1 : 1;
    return b.score - a.score;
  });
  return rows;
}

// --- signal attribution ------------------------------------------------------------------

export interface AttributedSignal {
  signal_id: string;
  source_type: string;
  ticker: string;
  sentiment: number;
  politician_id: string | null;
  resolved: boolean; // false when the id couldn't be matched to a recorded signal event
}

export interface DecisionAttribution {
  decision_id: string;
  occurred_at: string;
  symbol: string;
  side: string;
  quantity: number;
  convictionScore: number | null;
  rationale: string;
  signals: AttributedSignal[];
}

/**
 * Resolve each decision's `contributing_signal_ids` against recorded `signal.received` events,
 * scoped to one sleeve. The result is a linked view (decision → the signals that drove it)
 * for the attribution panel. Newest decisions first.
 */
export function signalAttribution(
  events: StoredEventDto[],
  sleeveId: string,
): DecisionAttribution[] {
  // Index signals by id once.
  const signalsById = new Map<string, AttributedSignal>();
  for (const e of events) {
    if (e.event_type !== "signal.received") continue;
    const p = e.payload;
    const id = str(p.signal_id);
    if (!id) continue;
    const politician = p.politician_id;
    signalsById.set(id, {
      signal_id: id,
      source_type: str(p.source_type, "unknown"),
      ticker: str(p.ticker),
      sentiment: num(p.sentiment),
      politician_id: typeof politician === "string" ? politician : null,
      resolved: true,
    });
  }

  const out: DecisionAttribution[] = [];
  for (const e of events) {
    if (e.event_type !== "decision.made" || str(e.payload.sleeve_id) !== sleeveId) continue;
    const p = e.payload;
    const ids = strArray(p.contributing_signal_ids);
    const signals: AttributedSignal[] = ids.map(
      (id) =>
        signalsById.get(id) ?? {
          signal_id: id,
          source_type: "unknown",
          ticker: "",
          sentiment: 0,
          politician_id: null,
          resolved: false,
        },
    );
    const score = p.conviction_score;
    out.push({
      decision_id: str(p.decision_id),
      occurred_at: e.occurred_at,
      symbol: str(p.symbol),
      side: str(p.side),
      quantity: num(p.quantity),
      convictionScore: typeof score === "number" ? score : null,
      rationale: str(p.rationale),
      signals,
    });
  }
  return out.reverse();
}
