import { describe, expect, it } from "vitest";
import type { ConvictionItem, StoredEventDto } from "../../lib/types";
import { algoStats, convictionBreakdown, signalAttribution } from "./analytics";

// Minimal StoredEventDto factory — only `event_type`, `occurred_at` and `payload` matter to
// the analytics transforms, so the rest get harmless placeholders.
let seq = 0;
function evt(
  event_type: string,
  payload: Record<string, unknown>,
  occurred_at = `2026-01-0${(seq % 9) + 1}T00:00:00.000Z`,
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

function snapshot(sleeve_id: string, equity: number, occurred_at: string): StoredEventDto {
  return evt("portfolio.snapshot", { sleeve_id, equity: String(equity) }, occurred_at);
}

describe("algoStats", () => {
  it("computes total return and max drawdown from the equity snapshots", () => {
    const events = [
      snapshot("sleeve-a", 1000, "2026-01-01T00:00:00.000Z"),
      snapshot("sleeve-a", 1200, "2026-01-02T00:00:00.000Z"), // peak
      snapshot("sleeve-a", 900, "2026-01-03T00:00:00.000Z"), // trough: (1200-900)/1200 = 25%
      snapshot("sleeve-a", 1100, "2026-01-04T00:00:00.000Z"), // recovery; last value
    ];
    const stats = algoStats(events, "sleeve-a");
    // (1100 - 1000) / 1000 = 10%
    expect(stats.totalReturnPct).toBeCloseTo(10, 6);
    expect(stats.maxDrawdownPct).toBeCloseTo(25, 6);
  });

  it("ignores snapshots from other sleeves", () => {
    const events = [
      snapshot("sleeve-a", 1000, "2026-01-01T00:00:00.000Z"),
      snapshot("sleeve-b", 5000, "2026-01-02T00:00:00.000Z"),
      snapshot("sleeve-a", 1500, "2026-01-03T00:00:00.000Z"),
    ];
    const stats = algoStats(events, "sleeve-a");
    expect(stats.totalReturnPct).toBeCloseTo(50, 6);
    expect(stats.maxDrawdownPct).toBe(0);
  });

  it("counts decisions and fills for the sleeve only", () => {
    const events = [
      evt("decision.made", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "buy", quantity: "10" }),
      evt("decision.made", { sleeve_id: "sleeve-b", symbol: "MSFT", side: "buy", quantity: "10" }),
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "buy", quantity: "10", price: "100", fees: "0" }),
    ];
    const stats = algoStats(events, "sleeve-a");
    expect(stats.decisionCount).toBe(1);
    expect(stats.fillCount).toBe(1);
  });

  it("realizes P&L and win rate from a profitable round-trip (avg cost, fees included)", () => {
    const events = [
      // Buy 10 @ 100 (fee 1), then sell 10 @ 110 (fee 1).
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "buy", quantity: "10", price: "100", fees: "1" }),
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "sell", quantity: "10", price: "110", fees: "1" }),
    ];
    const stats = algoStats(events, "sleeve-a");
    // Open: realizedPnl -= 1 (fee) => -1
    // Close: (110-100)*10*1 - 1 = 99; total = 98
    expect(stats.realizedPnl).toBeCloseTo(98, 6);
    expect(stats.closedTrades).toBe(1);
    expect(stats.winRatePct).toBe(100);
    expect(stats.fillCount).toBe(2);
  });

  it("books a losing close and reports a sub-100% win rate across two round-trips", () => {
    const events = [
      // Round-trip 1: win. Buy 10 @ 100, sell 10 @ 120 (no fees).
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "buy", quantity: "10", price: "100", fees: "0" }),
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "sell", quantity: "10", price: "120", fees: "0" }),
      // Round-trip 2: loss. Buy 10 @ 100, sell 10 @ 90.
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "MSFT", side: "buy", quantity: "10", price: "100", fees: "0" }),
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "MSFT", side: "sell", quantity: "10", price: "90", fees: "0" }),
    ];
    const stats = algoStats(events, "sleeve-a");
    // +200 (win) and -100 (loss) => +100 realized.
    expect(stats.realizedPnl).toBeCloseTo(100, 6);
    expect(stats.closedTrades).toBe(2);
    expect(stats.winRatePct).toBe(50);
  });

  it("returns null win rate when nothing has been closed yet", () => {
    const events = [
      evt("order.filled", { sleeve_id: "sleeve-a", symbol: "AAPL", side: "buy", quantity: "10", price: "100", fees: "0" }),
    ];
    const stats = algoStats(events, "sleeve-a");
    expect(stats.winRatePct).toBeNull();
    expect(stats.closedTrades).toBe(0);
  });

  it("returns zeroed stats for a sleeve with no events", () => {
    const stats = algoStats([], "sleeve-empty");
    expect(stats).toMatchObject({
      totalReturnPct: 0,
      maxDrawdownPct: 0,
      decisionCount: 0,
      fillCount: 0,
      winRatePct: null,
      realizedPnl: 0,
      closedTrades: 0,
    });
  });
});

describe("convictionBreakdown", () => {
  const conviction: ConvictionItem[] = [
    { ticker: "AAPL", score: 4, action: "trade", contributing_signal_ids: [] },
    { ticker: "MSFT", score: 8, action: "trade", contributing_signal_ids: [] },
    { ticker: "TSLA", score: -2, action: "ignore", contributing_signal_ids: [] },
  ];

  it("flags tickers the sleeve has decided on and orders traded-first", () => {
    const events = [
      evt("decision.made", { sleeve_id: "sleeve-a", symbol: "AAPL" }),
    ];
    const rows = convictionBreakdown(conviction, events, "sleeve-a");
    // AAPL is traded, so it sorts ahead of higher-scoring-but-untraded MSFT.
    expect(rows[0]?.ticker).toBe("AAPL");
    expect(rows[0]?.traded).toBe(true);
    // The untraded remainder is ordered by descending score.
    expect(rows.slice(1).map((r) => r.ticker)).toEqual(["MSFT", "TSLA"]);
    expect(rows.find((r) => r.ticker === "MSFT")?.traded).toBe(false);
  });

  it("normalizes scorePct against the strongest absolute score", () => {
    const rows = convictionBreakdown(conviction, [], "sleeve-a");
    const byTicker = new Map(rows.map((r) => [r.ticker, r]));
    expect(byTicker.get("MSFT")?.scorePct).toBeCloseTo(1, 6); // max |score| = 8
    expect(byTicker.get("AAPL")?.scorePct).toBeCloseTo(0.5, 6); // 4/8
    expect(byTicker.get("TSLA")?.scorePct).toBeCloseTo(0.25, 6); // |-2|/8
  });

  it("does not flag tickers a different sleeve traded", () => {
    const events = [evt("decision.made", { sleeve_id: "sleeve-other", symbol: "AAPL" })];
    const rows = convictionBreakdown(conviction, events, "sleeve-a");
    expect(rows.every((r) => r.traded === false)).toBe(true);
  });
});

describe("signalAttribution", () => {
  it("links a decision to its contributing signals (newest first) and marks unresolved ids", () => {
    const events = [
      evt(
        "signal.received",
        { signal_id: "sig-1", source_type: "statement", ticker: "AAPL", sentiment: 0.8, politician_id: "pol-1" },
        "2026-01-01T00:00:00.000Z",
      ),
      evt(
        "decision.made",
        {
          sleeve_id: "sleeve-a",
          decision_id: "dec-1",
          symbol: "AAPL",
          side: "buy",
          quantity: "10",
          conviction_score: 4.2,
          rationale: "strong corroboration",
          contributing_signal_ids: ["sig-1", "sig-missing"],
        },
        "2026-01-02T00:00:00.000Z",
      ),
      evt(
        "decision.made",
        {
          sleeve_id: "sleeve-a",
          decision_id: "dec-2",
          symbol: "MSFT",
          side: "sell",
          quantity: "5",
          contributing_signal_ids: [],
        },
        "2026-01-03T00:00:00.000Z",
      ),
    ];

    const out = signalAttribution(events, "sleeve-a");
    // Newest decision first.
    expect(out.map((d) => d.decision_id)).toEqual(["dec-2", "dec-1"]);

    const dec1 = out.find((d) => d.decision_id === "dec-1");
    expect(dec1?.convictionScore).toBe(4.2);
    expect(dec1?.signals).toHaveLength(2);
    const resolved = dec1?.signals.find((s) => s.signal_id === "sig-1");
    expect(resolved?.resolved).toBe(true);
    expect(resolved?.ticker).toBe("AAPL");
    expect(resolved?.politician_id).toBe("pol-1");
    const missing = dec1?.signals.find((s) => s.signal_id === "sig-missing");
    expect(missing?.resolved).toBe(false);

    // A decision without a numeric score reports null.
    const dec2 = out.find((d) => d.decision_id === "dec-2");
    expect(dec2?.convictionScore).toBeNull();
  });

  it("excludes decisions from other sleeves", () => {
    const events = [
      evt("decision.made", { sleeve_id: "sleeve-b", decision_id: "x", contributing_signal_ids: [] }),
    ];
    expect(signalAttribution(events, "sleeve-a")).toHaveLength(0);
  });
});
