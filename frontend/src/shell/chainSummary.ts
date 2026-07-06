// Per-event-type payload summary lines for the Chain Inspector (docs/UI-DESIGN.md §6). Mirrors
// the exact payload shapes each event type is constructed with on the backend (see
// backend/src/trading/application/{signal_recording,evaluate_signals,place_order,sync_orders,
// apply_trade_update,cancel_order}.py) — one line of human-readable context per node, not a raw
// dump (the raw JSONB is still available, expandable, per node).
//
// Also classifies each node's terminal-state color per §6 ("Terminal states colored by outcome")
// and §3.1's binding semantics: --gain/--loss are P&L direction / order side ONLY, never event
// lineage — so a node's summary text can mention "buy"/"sell"/"rejected" colored by that rule,
// while the connecting timeline itself (§6 "vertical timeline in --chain violet") stays violet
// regardless of outcome.

import type { StoredEventDto } from "../lib/types";
import { formatNumber, formatTime } from "../lib/format";

function str(payload: Record<string, unknown>, key: string, fallback = ""): string {
  const v = payload[key];
  if (v === null || v === undefined) return fallback;
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean" || typeof v === "bigint") return String(v);
  return JSON.stringify(v) ?? fallback;
}

function num(payload: Record<string, unknown>, key: string): number {
  return Number(payload[key]);
}

/** One-line human summary of an event's payload, keyed by event_type. Falls back to a generic
 *  key-count line for any event type not explicitly modeled (forward-compatible with new types). */
export function summarizeEvent(event: StoredEventDto): string {
  const p = event.payload;
  switch (event.event_type) {
    case "signal.received": {
      const provenance = str(p, "provenance", "REAL").toUpperCase();
      const source = str(p, "source_type", "—");
      const ticker = str(p, "ticker", "—");
      return `${source} · ${ticker} · ${provenance}`;
    }
    case "decision.made": {
      const score = num(p, "conviction_score");
      const side = str(p, "side", "—");
      return `conviction ${Number.isNaN(score) ? "—" : formatNumber(score, 0)} → ${side || "trade"}`;
    }
    case "risk.evaluated": {
      const verdict = str(p, "verdict", "—");
      const approved = str(p, "approved_quantity");
      const requested = str(p, "requested_quantity");
      const reason = str(p, "reason");
      const qtyNote =
        approved && requested && approved !== requested
          ? ` (${approved}/${requested} approved)`
          : "";
      return `${verdict}${qtyNote}${reason ? ` — ${reason}` : ""}`;
    }
    case "order.intent": {
      const side = str(p, "side", "—").toUpperCase();
      const qty = str(p, "quantity", "—");
      const symbol = str(p, "symbol", "—");
      const type = str(p, "order_type", "market");
      return `${side} ${qty} ${symbol} @ ${type}`;
    }
    case "order.submitted": {
      const coid = str(p, "client_order_id", "—");
      return `coid ${coid}`;
    }
    case "order.submit_failed": {
      return str(p, "error", "submit failed");
    }
    case "order.status_changed": {
      const status = str(p, "status", "—");
      const source = str(p, "source");
      return `${status}${source ? ` (${source})` : ""}`;
    }
    case "order.filled": {
      const qty = str(p, "quantity", "—");
      const price = str(p, "price", "—");
      const side = str(p, "side", "—");
      return `${side ? `${side.toUpperCase()} ` : ""}${qty} @ ${price}`;
    }
    default: {
      const keys = Object.keys(p).length;
      return `${keys} field${keys === 1 ? "" : "s"}`;
    }
  }
}

export type ChainOutcome = "gain" | "loss" | "neutral";

/** Terminal-state coloring per §6/§3.1: green/red mark P&L direction / order side ONLY — never
 *  decorative, never applied to non-terminal nodes. `order.filled` colors by side (buy=gain,
 *  sell=loss, matching the Blotter/Positions convention elsewhere in the app); rejections and
 *  failures color loss; everything else (including the connecting timeline) stays neutral/chain. */
export function chainOutcome(event: StoredEventDto): ChainOutcome {
  const p = event.payload;
  switch (event.event_type) {
    case "order.filled": {
      const side = str(p, "side").toLowerCase();
      return side === "sell" ? "loss" : "gain";
    }
    case "risk.evaluated": {
      const verdict = str(p, "verdict").toLowerCase();
      return verdict.includes("reject") || verdict.includes("deny") || verdict.includes("block")
        ? "loss"
        : "neutral";
    }
    case "order.submit_failed":
      return "loss";
    default:
      return "neutral";
  }
}

/** Local (§3.2) timestamp for a node's timeline label, hover reveals UTC via the title attr
 *  (formatTimeUtc), matching the app-wide timestamp convention exactly. */
export function nodeTime(event: StoredEventDto): string {
  return formatTime(event.occurred_at);
}
