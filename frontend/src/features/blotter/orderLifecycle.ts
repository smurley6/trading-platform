// Blotter order-lifecycle projection + status-chip mapping (UI Phase C §5.4) — pure, unit-tested.
//
// A Phase C sibling of lib/projections.ts's `blotterRows` (which stays untouched as a legacy
// contract): the §5.4 grid needs fields the old projection didn't carry — the correlation id
// (row click → Chain Inspector on the order's full lifecycle), the order type, and the fill
// price/quantity from `order.filled` — plus the lifecycle chip mapping:
//
//   intent  --text-muted · submitted/working  --accent · filled  --gain/--loss by side ·
//   failed/canceled  --loss outline   (§5.4)

import type { StoredEventDto } from "../../lib/types";

export interface OrderRow {
  client_order_id: string;
  correlation_id: string;
  /** Intent timestamp — the §5.4 "submitted ts" column (order.intent is the first lifecycle event). */
  occurred_at: string;
  sleeve_id: string;
  symbol: string;
  side: string;
  quantity: string;
  order_type: string;
  status: string;
  broker_order_id: string;
  fill_price: string | null;
  filled_quantity: string | null;
}

function str(value: unknown, fallback = ""): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
    return String(value);
  }
  return JSON.stringify(value) ?? fallback;
}

/** Latest lifecycle state per client_order_id from the event log, NEWEST intent first (§5.4). */
export function orderRows(events: readonly StoredEventDto[]): OrderRow[] {
  const byOrder = new Map<string, OrderRow>();
  for (const e of events) {
    const p = e.payload;
    const id = str(p.client_order_id);
    if (!id) continue;
    if (e.event_type === "order.intent") {
      byOrder.set(id, {
        client_order_id: id,
        correlation_id: e.correlation_id,
        occurred_at: e.occurred_at,
        sleeve_id: str(p.sleeve_id),
        symbol: str(p.symbol),
        side: str(p.side),
        quantity: str(p.quantity),
        order_type: str(p.order_type, "market"),
        status: "intent",
        broker_order_id: "",
        fill_price: null,
        filled_quantity: null,
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
    } else if (e.event_type === "order.filled") {
      const row = byOrder.get(id);
      if (row) {
        row.status = "filled";
        row.fill_price = str(p.price) || null;
        row.filled_quantity = str(p.quantity) || null;
      }
    } else if (e.event_type === "order.submit_failed") {
      const row = byOrder.get(id);
      if (row) row.status = "submit_failed";
    }
  }
  // Event-log order is oldest-first, so reversing the insertion order gives newest intent first.
  return [...byOrder.values()].reverse();
}

// ---- Status chip (§5.4 lifecycle colors) --------------------------------------------------------

/** Rendered by the Blotter with the matching §3.1 token: muted = --text-muted, accent =
 *  --accent, gain/loss = --gain/--loss (fill colored by ORDER SIDE), lossOutline = --loss
 *  outline (hard failures/cancellations). */
export type StatusTone = "muted" | "accent" | "gain" | "loss" | "lossOutline";

const FAILED_STATUSES = new Set([
  "canceled",
  "cancelled",
  "rejected",
  "expired",
  "submit_failed",
  "done_for_day",
]);

// Terminal = no further lifecycle transitions; not cancelable.
const TERMINAL_STATUSES = new Set(["filled", ...FAILED_STATUSES]);

export function statusChip(status: string, side: string): { label: string; tone: StatusTone } {
  const s = status.toLowerCase();
  if (s === "intent") return { label: status, tone: "muted" };
  if (s === "filled") {
    return { label: status, tone: side.toLowerCase() === "sell" ? "loss" : "gain" };
  }
  if (FAILED_STATUSES.has(s)) return { label: status, tone: "lossOutline" };
  // Everything still working: submitted / new / accepted / pending_* / partially_filled.
  return { label: status, tone: "accent" };
}

/** Open orders can be canceled (`x` hotkey / POST /orders/{id}/cancel). */
export function isCancelable(row: Pick<OrderRow, "status">): boolean {
  return !TERMINAL_STATUSES.has(row.status.toLowerCase());
}
