import { describe, expect, it } from "vitest";
import { colors } from "../../lib/tokens";
import type { StoredEventDto } from "../../lib/types";
import { eventTickerColor } from "./eventColors";

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

describe("eventTickerColor", () => {
  it("colors a buy fill --gain and a sell fill --loss (order side, per §3.1)", () => {
    expect(eventTickerColor(evt("order.filled", { side: "buy" }))).toBe(colors.gain);
    expect(eventTickerColor(evt("order.filled", { side: "sell" }))).toBe(colors.loss);
  });

  it("colors a rejected risk verdict --loss", () => {
    expect(eventTickerColor(evt("risk.evaluated", { verdict: "rejected" }))).toBe(colors.loss);
  });

  it("colors reconciliation drift --warn", () => {
    expect(eventTickerColor(evt("reconciliation.drift", {}))).toBe(colors.warn);
  });

  it("colors every other lineage event --chain violet", () => {
    expect(eventTickerColor(evt("signal.received", {}))).toBe(colors.chain);
    expect(eventTickerColor(evt("decision.made", {}))).toBe(colors.chain);
    expect(eventTickerColor(evt("order.intent", {}))).toBe(colors.chain);
    expect(eventTickerColor(evt("order.submitted", {}))).toBe(colors.chain);
  });
});
