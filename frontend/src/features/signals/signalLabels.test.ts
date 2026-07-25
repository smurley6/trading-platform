import { describe, expect, it } from "vitest";
import type { OutcomeLabelItem } from "../../lib/types";
import {
  benchmarkKind,
  benchmarkSymbol,
  labelAtHorizon,
  labelsBySignal,
} from "./signalLabels";

function label(partial: Partial<OutcomeLabelItem> & { signal_id: string }): OutcomeLabelItem {
  return {
    ticker: "AAPL",
    horizon_days: 1,
    abnormal_return: 0.01,
    benchmark: "sector:XLK",
    status: "ok",
    reason: null,
    ...partial,
  };
}

describe("benchmark split", () => {
  it("splits sector vs market on the prefix", () => {
    expect(benchmarkKind("sector:XLK")).toBe("sector");
    expect(benchmarkKind("market")).toBe("market");
    expect(benchmarkSymbol("sector:ITA")).toBe("ITA");
    expect(benchmarkSymbol("market")).toBe("SPY");
  });
});

describe("labelsBySignal", () => {
  it("groups horizons under one summary per signal and carries the shared benchmark", () => {
    const map = labelsBySignal([
      label({ signal_id: "s1", horizon_days: 1, abnormal_return: 0.01 }),
      label({ signal_id: "s1", horizon_days: 5, abnormal_return: 0.02 }),
      label({
        signal_id: "s1",
        horizon_days: 20,
        abnormal_return: null,
        status: "unresolvable",
        reason: "insufficient_forward_bars",
      }),
      label({ signal_id: "s2", horizon_days: 1, benchmark: "market", abnormal_return: -0.03 }),
    ]);

    const s1 = map.get("s1");
    expect(s1?.benchmarkKind).toBe("sector");
    expect(s1?.benchmarkSymbol).toBe("XLK");
    expect(s1?.hasUnresolved).toBe(true);
    expect(labelAtHorizon(s1, 5)?.abnormal_return).toBe(0.02);
    expect(labelAtHorizon(s1, 20)?.status).toBe("unresolvable");

    const s2 = map.get("s2");
    expect(s2?.benchmarkKind).toBe("market");
    expect(s2?.benchmarkSymbol).toBe("SPY");
    expect(s2?.hasUnresolved).toBe(false);
  });

  it("returns undefined for an unlabeled signal", () => {
    const map = labelsBySignal([]);
    expect(map.get("nope")).toBeUndefined();
    expect(labelAtHorizon(undefined, 1)).toBeUndefined();
  });
});
