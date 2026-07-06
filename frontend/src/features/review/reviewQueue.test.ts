import { describe, expect, it } from "vitest";
import type { ProposalItem, SignalFeedItem } from "../../lib/types";
import { formatAge, newestSignalAtMs, orderQueue } from "./reviewQueue";

function proposal(symbol: string, conviction: number, signalIds: string[] = []): ProposalItem {
  return {
    symbol,
    side: "buy",
    quantity: "1",
    target_quantity: "1",
    current_quantity: "0",
    conviction_score: conviction,
    action: "trade",
    contributing_signal_ids: signalIds,
    rationale: "",
  };
}

function signal(id: string, occurredAt: string): SignalFeedItem {
  return {
    signal_id: id,
    occurred_at: occurredAt,
    source_type: "statement",
    ticker: "X",
    politician_id: null,
    sentiment: 0.5,
    politician_weight: 1,
    source_reliability: 1,
  };
}

const signalsById = new Map<string, SignalFeedItem>([
  ["s-old", signal("s-old", "2026-07-01T10:00:00Z")],
  ["s-new", signal("s-new", "2026-07-06T10:00:00Z")],
]);

describe("orderQueue (§5.2: conviction desc, then age)", () => {
  it("sorts by conviction descending first", () => {
    const queue = orderQueue(
      [proposal("LOW", 71), proposal("HIGH", 84), proposal("MID", 76)],
      new Map(),
      new Map(),
    );
    expect(queue.map((p) => p.symbol)).toEqual(["HIGH", "MID", "LOW"]);
  });

  it("breaks conviction ties by age — older evidence first", () => {
    const queue = orderQueue(
      [proposal("NEWER", 80, ["s-new"]), proposal("OLDER", 80, ["s-old"])],
      signalsById,
      new Map(),
    );
    expect(queue.map((p) => p.symbol)).toEqual(["OLDER", "NEWER"]);
  });

  it("sorts unknown-age proposals after dated ones at equal conviction, then by symbol", () => {
    const queue = orderQueue(
      [proposal("B", 80), proposal("A", 80), proposal("DATED", 80, ["s-old"])],
      signalsById,
      new Map(),
    );
    expect(queue.map((p) => p.symbol)).toEqual(["DATED", "A", "B"]);
  });

  it("drops session-resolved symbols (approved or rejected)", () => {
    const queue = orderQueue(
      [proposal("KEEP", 90), proposal("GONE", 95)],
      new Map(),
      new Map([["GONE", "rejected" as const]]),
    );
    expect(queue.map((p) => p.symbol)).toEqual(["KEEP"]);
  });
});

describe("newestSignalAtMs", () => {
  it("returns the newest contributing signal's timestamp", () => {
    const at = newestSignalAtMs(proposal("X", 80, ["s-old", "s-new"]), signalsById);
    expect(at).toBe(Date.parse("2026-07-06T10:00:00Z"));
  });

  it("returns null when no contributing signal resolves", () => {
    expect(newestSignalAtMs(proposal("X", 80, ["missing"]), signalsById)).toBeNull();
  });
});

describe("formatAge", () => {
  const now = Date.parse("2026-07-06T12:00:00Z");
  it("formats seconds, minutes, hours, and days", () => {
    expect(formatAge(now - 30_000, now)).toBe("30s");
    expect(formatAge(now - 35 * 60_000, now)).toBe("35m");
    expect(formatAge(now - 2 * 3_600_000, now)).toBe("2h");
    expect(formatAge(now - 3 * 86_400_000, now)).toBe("3d");
  });
  it("renders an em-dash for unknown age and clamps future timestamps", () => {
    expect(formatAge(null, now)).toBe("—");
    expect(formatAge(now + 60_000, now)).toBe("0s");
  });
});
