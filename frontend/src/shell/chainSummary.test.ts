import { describe, expect, it } from "vitest";
import type { StoredEventDto } from "../lib/types";
import { chainOutcome, summarizeEvent } from "./chainSummary";

function evt(event_type: string, payload: Record<string, unknown>): StoredEventDto {
  return {
    event_id: "e1",
    stream_id: "s",
    seq: 1,
    event_type,
    schema_version: 1,
    occurred_at: "2026-07-01T09:31:02.000Z",
    recorded_at: "2026-07-01T09:31:02.000Z",
    correlation_id: "c1",
    payload,
  };
}

describe("summarizeEvent", () => {
  it("summarizes signal.received with source/ticker/provenance", () => {
    const e = evt("signal.received", { source_type: "sec_form4", ticker: "LMT", provenance: "REAL" });
    expect(summarizeEvent(e)).toBe("sec_form4 · LMT · REAL");
  });

  it("defaults provenance to REAL when absent", () => {
    const e = evt("signal.received", { source_type: "sec_form4", ticker: "LMT" });
    expect(summarizeEvent(e)).toContain("REAL");
  });

  it("summarizes decision.made with conviction score and side", () => {
    const e = evt("decision.made", { conviction_score: 84, side: "buy" });
    expect(summarizeEvent(e)).toBe("conviction 84 → buy");
  });

  it("summarizes risk.evaluated with verdict and a quantity note when reduced", () => {
    const e = evt("risk.evaluated", {
      verdict: "approved",
      reason: "size ok",
      requested_quantity: "10",
      approved_quantity: "9",
    });
    expect(summarizeEvent(e)).toBe("approved (9/10 approved) — size ok");
  });

  it("omits the quantity note when approved equals requested", () => {
    const e = evt("risk.evaluated", {
      verdict: "approved",
      reason: "ok",
      requested_quantity: "10",
      approved_quantity: "10",
    });
    expect(summarizeEvent(e)).toBe("approved — ok");
  });

  it("summarizes order.intent as SIDE qty symbol @ type", () => {
    const e = evt("order.intent", { side: "buy", quantity: "9", symbol: "LMT", order_type: "market" });
    expect(summarizeEvent(e)).toBe("BUY 9 LMT @ market");
  });

  it("summarizes order.submitted with the client order id", () => {
    const e = evt("order.submitted", { client_order_id: "pol-7f3a" });
    expect(summarizeEvent(e)).toBe("coid pol-7f3a");
  });

  it("summarizes order.filled as SIDE qty @ price", () => {
    const e = evt("order.filled", { side: "buy", quantity: "9", price: "266.58" });
    expect(summarizeEvent(e)).toBe("BUY 9 @ 266.58");
  });

  it("summarizes order.submit_failed with the error text", () => {
    const e = evt("order.submit_failed", { error: "insufficient buying power" });
    expect(summarizeEvent(e)).toBe("insufficient buying power");
  });

  it("falls back to a field count for unmodeled event types", () => {
    const e = evt("portfolio.snapshot", { sleeve_id: "a", equity: "1", cash: "2" });
    expect(summarizeEvent(e)).toBe("3 fields");
  });
});

describe("chainOutcome", () => {
  it("colors a buy fill as gain", () => {
    expect(chainOutcome(evt("order.filled", { side: "buy" }))).toBe("gain");
  });

  it("colors a sell fill as loss", () => {
    expect(chainOutcome(evt("order.filled", { side: "sell" }))).toBe("loss");
  });

  it("colors a rejected risk verdict as loss", () => {
    expect(chainOutcome(evt("risk.evaluated", { verdict: "rejected" }))).toBe("loss");
  });

  it("colors an approved risk verdict as neutral", () => {
    expect(chainOutcome(evt("risk.evaluated", { verdict: "approved" }))).toBe("neutral");
  });

  it("colors a submit failure as loss", () => {
    expect(chainOutcome(evt("order.submit_failed", { error: "x" }))).toBe("loss");
  });

  it("colors everything else neutral (the connecting timeline stays --chain violet)", () => {
    expect(chainOutcome(evt("signal.received", {}))).toBe("neutral");
    expect(chainOutcome(evt("decision.made", {}))).toBe("neutral");
    expect(chainOutcome(evt("order.intent", {}))).toBe("neutral");
  });
});
