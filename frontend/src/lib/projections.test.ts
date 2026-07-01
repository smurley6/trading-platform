import { describe, expect, it } from "vitest";
import {
  backtestCurve,
  blotterRows,
  equitySeries,
  latestSnapshots,
  positionRows,
  riskRows,
} from "./projections";
import type { BacktestPoint, StoredEventDto } from "./types";

let seq = 0;
function evt(
  event_type: string,
  payload: Record<string, unknown>,
  occurred_at = "2026-01-01T00:00:00.000Z",
): StoredEventDto {
  seq += 1;
  return {
    event_id: `e${seq}`,
    stream_id: "s",
    seq,
    event_type,
    schema_version: 1,
    occurred_at,
    recorded_at: occurred_at,
    correlation_id: "c",
    payload,
  };
}

describe("blotterRows", () => {
  it("folds an order's lifecycle into a single newest-first row", () => {
    const events = [
      evt("order.intent", {
        client_order_id: "o1",
        sleeve_id: "a",
        symbol: "AAPL",
        side: "buy",
        quantity: "10",
      }),
      evt("order.submitted", { client_order_id: "o1", broker_order_id: "b-123" }),
      evt("order.status_changed", { client_order_id: "o1", status: "filled" }),
    ];
    const rows = blotterRows(events);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      client_order_id: "o1",
      symbol: "AAPL",
      side: "buy",
      quantity: "10",
      status: "filled",
      broker_order_id: "b-123",
    });
  });

  it("marks a failed submission", () => {
    const events = [
      evt("order.intent", { client_order_id: "o2", symbol: "MSFT", side: "buy", quantity: "5" }),
      evt("order.submit_failed", { client_order_id: "o2" }),
    ];
    expect(blotterRows(events)[0]?.status).toBe("submit_failed");
  });

  it("ignores lifecycle events with no preceding intent and rows without an id", () => {
    const events = [
      evt("order.submitted", { client_order_id: "ghost", broker_order_id: "b" }),
      evt("order.intent", { symbol: "NOID" }), // missing client_order_id
    ];
    expect(blotterRows(events)).toHaveLength(0);
  });

  it("returns rows newest-first", () => {
    const events = [
      evt("order.intent", { client_order_id: "first", symbol: "A", side: "buy", quantity: "1" }),
      evt("order.intent", { client_order_id: "second", symbol: "B", side: "buy", quantity: "1" }),
    ];
    expect(blotterRows(events).map((r) => r.client_order_id)).toEqual(["second", "first"]);
  });
});

describe("positionRows", () => {
  it("nets signed fills per (sleeve, symbol) and drops flat positions", () => {
    const events = [
      evt("order.filled", { sleeve_id: "a", symbol: "AAPL", side: "buy", quantity: "10" }),
      evt("order.filled", { sleeve_id: "a", symbol: "AAPL", side: "sell", quantity: "4" }),
      evt("order.filled", { sleeve_id: "a", symbol: "MSFT", side: "buy", quantity: "3" }),
      evt("order.filled", { sleeve_id: "a", symbol: "MSFT", side: "sell", quantity: "3" }), // flat -> dropped
    ];
    const rows = positionRows(events);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual({ sleeve_id: "a", symbol: "AAPL", quantity: 6 });
  });

  it("keeps positions of different sleeves separate", () => {
    const events = [
      evt("order.filled", { sleeve_id: "a", symbol: "AAPL", side: "buy", quantity: "10" }),
      evt("order.filled", { sleeve_id: "b", symbol: "AAPL", side: "buy", quantity: "2" }),
    ];
    const rows = positionRows(events);
    expect(rows).toHaveLength(2);
    expect(rows.find((r) => r.sleeve_id === "b")?.quantity).toBe(2);
  });
});

describe("riskRows", () => {
  it("projects risk.evaluated events newest-first", () => {
    const events = [
      evt("risk.evaluated", {
        decision_id: "d1",
        verdict: "approved",
        reason: "ok",
        requested_quantity: "10",
        approved_quantity: "10",
      }),
      evt("order.filled", { sleeve_id: "a", symbol: "AAPL", side: "buy", quantity: "1" }),
      evt("risk.evaluated", {
        decision_id: "d2",
        verdict: "rejected",
        reason: "limit",
        requested_quantity: "100",
        approved_quantity: "0",
      }),
    ];
    const rows = riskRows(events);
    expect(rows.map((r) => r.decision_id)).toEqual(["d2", "d1"]);
    expect(rows[0]?.verdict).toBe("rejected");
  });
});

describe("latestSnapshots", () => {
  it("keeps the most recent snapshot per sleeve and counts positions", () => {
    const events = [
      evt("portfolio.snapshot", {
        sleeve_id: "a",
        cash: "100",
        equity: "1000",
        positions: { AAPL: 1 },
      }),
      evt("portfolio.snapshot", {
        sleeve_id: "a",
        cash: "50",
        equity: "1500",
        positions: { AAPL: 1, MSFT: 2 },
      }),
    ];
    const snaps = latestSnapshots(events);
    const a = snaps.get("a");
    expect(a?.equity).toBe(1500);
    expect(a?.cash).toBe(50);
    expect(a?.positionCount).toBe(2);
  });
});

describe("equitySeries", () => {
  it("builds an oldest-first curve and forces strictly increasing time", () => {
    // Two snapshots at the same wall-clock second must yield distinct, increasing times
    // because lightweight-charts rejects duplicate/decreasing timestamps.
    const events = [
      evt("portfolio.snapshot", { sleeve_id: "a", equity: "1000" }, "2026-01-01T00:00:00.000Z"),
      evt("portfolio.snapshot", { sleeve_id: "a", equity: "1100" }, "2026-01-01T00:00:00.000Z"),
      evt("portfolio.snapshot", { sleeve_id: "b", equity: "9" }, "2026-01-01T00:00:05.000Z"),
    ];
    const series = equitySeries(events, "a");
    expect(series.map((p) => p.value)).toEqual([1000, 1100]);
    expect(series[1]!.time).toBe(series[0]!.time + 1);
  });
});

describe("backtestCurve", () => {
  it("returns [] for undefined or empty input", () => {
    expect(backtestCurve(undefined)).toEqual([]);
    expect(backtestCurve([])).toEqual([]);
  });

  it("normalizes the new { t: ISO, equity } shape and coerces string equities", () => {
    const points: BacktestPoint[] = [
      { t: "2026-01-01T00:00:00Z", equity: "1000" },
      { t: "2026-01-02T00:00:00Z", equity: 1100 },
    ];
    const curve = backtestCurve(points);
    expect(curve.map((p) => p.value)).toEqual([1000, 1100]);
    expect(curve[0]!.time).toBe(Math.floor(Date.parse("2026-01-01T00:00:00Z") / 1000));
  });

  it("supports the legacy { time, equity } shape", () => {
    const curve = backtestCurve([{ time: 10, equity: 5 }]);
    expect(curve).toEqual([{ time: 10, value: 5 }]);
  });

  it("forces strictly increasing timestamps for shared/duplicate times", () => {
    const curve = backtestCurve([
      { time: 100, equity: 1 },
      { time: 100, equity: 2 },
      { time: 100, equity: 3 },
    ]);
    expect(curve.map((p) => p.time)).toEqual([100, 101, 102]);
  });

  it("skips points with unparseable time or equity", () => {
    const curve = backtestCurve([
      { t: "not-a-date", equity: 5 },
      { t: "2026-01-01T00:00:00Z", equity: "nope" },
      { t: "2026-01-01T00:00:00Z", equity: 7 },
    ]);
    expect(curve.map((p) => p.value)).toEqual([7]);
  });
});
