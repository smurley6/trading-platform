import { describe, expect, it } from "vitest";
import type { StoredEventDto } from "../../lib/types";
import { isCancelable, orderRows, statusChip } from "./orderLifecycle";

let seq = 0;
function evt(
  event_type: string,
  payload: Record<string, unknown>,
  correlation_id = "corr-1",
): StoredEventDto {
  seq += 1;
  return {
    event_id: `e${seq}`,
    stream_id: "order-x",
    seq,
    event_type,
    schema_version: 1,
    occurred_at: `2026-07-01T00:00:${String(seq).padStart(2, "0")}.000Z`,
    recorded_at: `2026-07-01T00:00:${String(seq).padStart(2, "0")}.000Z`,
    correlation_id,
    payload,
  };
}

describe("orderRows", () => {
  it("projects the full §5.4 row: correlation id, type, and fill price/qty from order.filled", () => {
    const events = [
      evt(
        "order.intent",
        {
          client_order_id: "o1",
          sleeve_id: "political",
          symbol: "LMT",
          side: "buy",
          quantity: "9",
          order_type: "market",
        },
        "corr-lmt",
      ),
      evt("order.submitted", { client_order_id: "o1", broker_order_id: "b1" }, "corr-lmt"),
      evt(
        "order.filled",
        { client_order_id: "o1", symbol: "LMT", side: "buy", quantity: "9", price: "266.58" },
        "corr-lmt",
      ),
    ];
    const rows = orderRows(events);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      client_order_id: "o1",
      correlation_id: "corr-lmt",
      sleeve_id: "political",
      symbol: "LMT",
      side: "buy",
      order_type: "market",
      status: "filled",
      broker_order_id: "b1",
      fill_price: "266.58",
      filled_quantity: "9",
    });
  });

  it("orders newest intent first", () => {
    const events = [
      evt("order.intent", { client_order_id: "first", symbol: "A", side: "buy", quantity: "1" }),
      evt("order.intent", { client_order_id: "second", symbol: "B", side: "buy", quantity: "1" }),
    ];
    expect(orderRows(events).map((r) => r.client_order_id)).toEqual(["second", "first"]);
  });

  it("tracks status_changed and submit_failed transitions", () => {
    const events = [
      evt("order.intent", { client_order_id: "o2", symbol: "NOC", side: "sell", quantity: "3" }),
      evt("order.status_changed", { client_order_id: "o2", status: "accepted" }),
      evt("order.submit_failed", { client_order_id: "o2", error: "boom" }),
    ];
    expect(orderRows(events)[0]?.status).toBe("submit_failed");
  });

  it("defaults order_type to market and ignores events without a client_order_id", () => {
    const events = [
      evt("order.intent", { client_order_id: "o3", symbol: "RTX", side: "buy", quantity: "2" }),
      evt("portfolio.snapshot", { sleeve_id: "s", equity: "1" }),
    ];
    const rows = orderRows(events);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.order_type).toBe("market");
  });
});

describe("statusChip (§5.4 lifecycle colors)", () => {
  it("maps intent to muted", () => {
    expect(statusChip("intent", "buy").tone).toBe("muted");
  });
  it("maps working statuses to accent", () => {
    for (const s of ["submitted", "new", "accepted", "partially_filled", "pending_new"]) {
      expect(statusChip(s, "buy").tone).toBe("accent");
    }
  });
  it("colors filled by ORDER SIDE (buy gain, sell loss)", () => {
    expect(statusChip("filled", "buy").tone).toBe("gain");
    expect(statusChip("filled", "sell").tone).toBe("loss");
  });
  it("maps failures and cancellations to the loss outline", () => {
    for (const s of ["canceled", "cancelled", "rejected", "expired", "submit_failed"]) {
      expect(statusChip(s, "buy").tone).toBe("lossOutline");
    }
  });
});

describe("isCancelable", () => {
  it("is true for open lifecycle states and false for terminal ones", () => {
    expect(isCancelable({ status: "intent" })).toBe(true);
    expect(isCancelable({ status: "submitted" })).toBe(true);
    expect(isCancelable({ status: "filled" })).toBe(false);
    expect(isCancelable({ status: "CANCELED" })).toBe(false);
    expect(isCancelable({ status: "submit_failed" })).toBe(false);
  });
});
