// Research surface on Signals (UI Phase D, docs/UI-DESIGN.md §5.5): pure helpers that fold the
// outcome_labels projection (GET /signals/labels) into a per-signal summary the grid renders as
// forward abnormal-return columns at the 1/5/20-day horizons, with the benchmark's sector-vs-market
// split derived by splitting on the "sector:" / "market" prefix (the spec's data-shape note).
//
// A signal's three horizon rows share one benchmark (the labeler resolves the benchmark per
// ticker, not per horizon), so the summary carries a single benchmark kind + symbol.

import type { OutcomeLabelItem } from "../../lib/types";

/** The three forward horizons (trading days) the labeler computes, in display order. */
export const LABEL_HORIZONS = [1, 5, 20] as const;

export type BenchmarkKind = "sector" | "market";

/** Split the `benchmark` string on its prefix: "sector:XLK" -> "sector", "market" -> "market". */
export function benchmarkKind(benchmark: string): BenchmarkKind {
  return benchmark.startsWith("sector:") ? "sector" : "market";
}

/** The benchmark's proxy symbol: the sector ETF ("sector:XLK" -> "XLK") or SPY for the market. */
export function benchmarkSymbol(benchmark: string): string {
  return benchmark.startsWith("sector:") ? benchmark.slice("sector:".length) : "SPY";
}

export interface SignalLabelSummary {
  /** Label row per horizon (1/5/20). A horizon may be absent (never labeled) or unresolvable. */
  byHorizon: Map<number, OutcomeLabelItem>;
  benchmark: string;
  benchmarkKind: BenchmarkKind;
  benchmarkSymbol: string;
  /** True when any horizon is "unresolvable" (too recent / missing bars) — surfaced as diagnostic. */
  hasUnresolved: boolean;
}

/** Group flat label rows into one summary per signal id (keyed for O(1) grid joins). */
export function labelsBySignal(labels: OutcomeLabelItem[]): Map<string, SignalLabelSummary> {
  const map = new Map<string, SignalLabelSummary>();
  for (const label of labels) {
    let entry = map.get(label.signal_id);
    if (entry === undefined) {
      entry = {
        byHorizon: new Map(),
        benchmark: label.benchmark,
        benchmarkKind: benchmarkKind(label.benchmark),
        benchmarkSymbol: benchmarkSymbol(label.benchmark),
        hasUnresolved: false,
      };
      map.set(label.signal_id, entry);
    }
    entry.byHorizon.set(label.horizon_days, label);
    if (label.status === "unresolvable") entry.hasUnresolved = true;
  }
  return map;
}

/** The label at one horizon for a summary, or undefined if that horizon has no row. */
export function labelAtHorizon(
  summary: SignalLabelSummary | undefined,
  horizon: number,
): OutcomeLabelItem | undefined {
  return summary?.byHorizon.get(horizon);
}
