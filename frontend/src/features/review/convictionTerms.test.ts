import { describe, expect, it } from "vitest";
import type { SignalFeedItem } from "../../lib/types";
import { convictionTerms, primarySourceUrl } from "./convictionTerms";

function signal(overrides: Partial<SignalFeedItem> & { signal_id: string }): SignalFeedItem {
  return {
    occurred_at: "2026-07-01T10:00:00Z",
    source_type: "disclosed_trade",
    ticker: "LMT",
    politician_id: "pol-1",
    sentiment: 0.8,
    politician_weight: 1.5,
    source_reliability: 0.9,
    ...overrides,
  };
}

describe("convictionTerms (§5.2: real terms only, never invented)", () => {
  it("computes the per-signal s·w·r contribution and the raw weighted sum", () => {
    const byId = new Map([["a", signal({ signal_id: "a" })]]);
    const terms = convictionTerms(["a"], byId);
    expect(terms.resolved).toHaveLength(1);
    expect(terms.resolved[0]!.contribution).toBeCloseTo(0.8 * 1.5 * 0.9);
    expect(terms.rawWeightedSum).toBeCloseTo(0.8 * 1.5 * 0.9);
  });

  it("separates unresolvable ids instead of fabricating terms for them", () => {
    const byId = new Map([["a", signal({ signal_id: "a" })]]);
    const terms = convictionTerms(["a", "ghost"], byId);
    expect(terms.resolved.map((t) => t.signalId)).toEqual(["a"]);
    expect(terms.missingIds).toEqual(["ghost"]);
  });

  it("collects distinct source types (the corroboration input) and sorts newest first", () => {
    const byId = new Map([
      ["a", signal({ signal_id: "a", source_type: "statement", occurred_at: "2026-07-01T10:00:00Z" })],
      ["b", signal({ signal_id: "b", source_type: "contract", occurred_at: "2026-07-03T10:00:00Z" })],
      ["c", signal({ signal_id: "c", source_type: "contract", occurred_at: "2026-07-02T10:00:00Z" })],
    ]);
    const terms = convictionTerms(["a", "b", "c"], byId);
    expect(terms.resolved.map((t) => t.signalId)).toEqual(["b", "c", "a"]);
    expect(terms.distinctSources.sort()).toEqual(["contract", "statement"]);
  });
});

describe("primarySourceUrl", () => {
  it("finds the live adapters' url keys (source_url / link / url)", () => {
    expect(
      primarySourceUrl(signal({ signal_id: "a", raw_payload: { source_url: "https://sec.gov/x" } })),
    ).toBe("https://sec.gov/x");
    expect(
      primarySourceUrl(signal({ signal_id: "b", raw_payload: { link: "http://news/x" } })),
    ).toBe("http://news/x");
  });

  it("rejects non-http payload values and missing payloads", () => {
    expect(
      primarySourceUrl(signal({ signal_id: "a", raw_payload: { url: "javascript:alert(1)" } })),
    ).toBeNull();
    expect(primarySourceUrl(signal({ signal_id: "b" }))).toBeNull();
  });
});
