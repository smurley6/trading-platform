import { describe, expect, it } from "vitest";
import type { SignalMeta } from "../../lib/signalMeta";
import type { SignalFeedItem } from "../../lib/types";
import {
  applySignalFilters,
  buildSignalRows,
  EMPTY_FILTERS,
  hasActiveFilters,
  provenanceFromParam,
  sourcesFromParam,
  sourcesToParam,
} from "./signalFilters";

function feedItem(overrides: Partial<SignalFeedItem> & { signal_id: string }): SignalFeedItem {
  return {
    occurred_at: "2026-07-03T12:00:00Z",
    source_type: "statement",
    ticker: "LMT",
    politician_id: "pol-1",
    sentiment: 0.5,
    politician_weight: 1,
    source_reliability: 1,
    ...overrides,
  };
}

const meta = new Map<string, SignalMeta>([
  ["real", { correlationId: "c-real", provenance: "REAL" }],
  ["backfilled", { correlationId: "c-bf", provenance: "BACKFILLED" }],
  ["synthetic", { correlationId: "c-syn", provenance: "SYNTHETIC" }],
]);

const rows = buildSignalRows(
  [
    feedItem({ signal_id: "real", ticker: "LMT", source_type: "statement" }),
    feedItem({ signal_id: "backfilled", ticker: "NOC", source_type: "disclosed_trade" }),
    feedItem({ signal_id: "synthetic", ticker: null, source_type: "contract" }),
    feedItem({ signal_id: "unknown", ticker: "RTX", occurred_at: "2026-06-01T09:00:00Z" }),
  ],
  meta,
);

describe("buildSignalRows (events-window join)", () => {
  it("joins provenance and correlation id from signal.received events", () => {
    expect(rows[0]).toMatchObject({ resolvedProvenance: "REAL", correlationId: "c-real" });
    expect(rows[1]).toMatchObject({ resolvedProvenance: "BACKFILLED", correlationId: "c-bf" });
  });
  it("leaves rows outside the events window unknown (null), never guessed", () => {
    expect(rows[3]).toMatchObject({ resolvedProvenance: null, correlationId: null });
  });
});

describe("applySignalFilters", () => {
  it("passes everything through with empty filters", () => {
    expect(applySignalFilters(rows, EMPTY_FILTERS)).toHaveLength(4);
  });

  it("filters by source (multi)", () => {
    const out = applySignalFilters(rows, {
      ...EMPTY_FILTERS,
      sources: ["contract", "disclosed_trade"],
    });
    expect(out.map((r) => r.signal_id).sort()).toEqual(["backfilled", "synthetic"]);
  });

  it("filters by provenance; unknown rows match only ALL", () => {
    expect(
      applySignalFilters(rows, { ...EMPTY_FILTERS, provenance: "BACKFILLED" }).map(
        (r) => r.signal_id,
      ),
    ).toEqual(["backfilled"]);
    // The unknown-provenance row is excluded from every explicit provenance filter.
    expect(
      applySignalFilters(rows, { ...EMPTY_FILTERS, provenance: "REAL" }).map((r) => r.signal_id),
    ).toEqual(["real"]);
  });

  it("filters by ticker substring, case-insensitively, excluding unresolved rows", () => {
    const out = applySignalFilters(rows, { ...EMPTY_FILTERS, ticker: "lm" });
    expect(out.map((r) => r.signal_id)).toEqual(["real"]);
  });

  it("finds the unresolved-ticker research set (§5.3 'must be findable')", () => {
    const out = applySignalFilters(rows, { ...EMPTY_FILTERS, unresolvedOnly: true });
    expect(out.map((r) => r.signal_id)).toEqual(["synthetic"]);
  });

  it("filters by inclusive local-day date range", () => {
    const out = applySignalFilters(rows, { ...EMPTY_FILTERS, from: "2026-07-01", to: "2026-07-31" });
    expect(out.map((r) => r.signal_id).sort()).toEqual(["backfilled", "real", "synthetic"]);
    const none = applySignalFilters(rows, { ...EMPTY_FILTERS, to: "2026-05-31" });
    expect(none).toHaveLength(0);
  });
});

describe("URL param round-tripping (§7 /signals?provenance=…&source=…)", () => {
  it("parses and serializes the source multi-filter", () => {
    expect(sourcesFromParam("statement,disclosed_trade")).toEqual([
      "statement",
      "disclosed_trade",
    ]);
    expect(sourcesFromParam("statement,bogus")).toEqual(["statement"]);
    expect(sourcesFromParam(null)).toEqual([]);
    expect(sourcesToParam(["contract"])).toBe("contract");
    expect(sourcesToParam([])).toBeNull();
  });

  it("parses provenance case-insensitively with ALL fallback", () => {
    expect(provenanceFromParam("backfilled")).toBe("BACKFILLED");
    expect(provenanceFromParam("nope")).toBe("ALL");
    expect(provenanceFromParam(null)).toBe("ALL");
  });
});

describe("hasActiveFilters", () => {
  it("is false only for the empty filter set", () => {
    expect(hasActiveFilters(EMPTY_FILTERS)).toBe(false);
    expect(hasActiveFilters({ ...EMPTY_FILTERS, unresolvedOnly: true })).toBe(true);
    expect(hasActiveFilters({ ...EMPTY_FILTERS, ticker: "L" })).toBe(true);
  });
});
