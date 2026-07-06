// Signals screen filter model (UI Phase C §5.3) — pure, unit-tested.
//
// Filters: source (multi), provenance (REAL/BACKFILLED/SYNTHETIC), ticker text, unresolved-only
// (§5.3: "unresolved = muted em-dash, filterable — the unresolved set is research data, so it
// must be findable"), date range. NO conviction-band filter: conviction is scored per TICKER
// (GET /conviction), not per signal — the feed DTO carries no per-signal conviction, so a band
// filter here would be attributing numbers the surface doesn't have (gap flagged in the phase
// report rather than faked).
//
// Provenance is joined from the events window (lib/signalMeta.ts) because GET /signals/feed's
// DTO omits it; rows whose `signal.received` event falls outside the window have provenance null
// ("unknown") and match only the ALL provenance filter — never guessed to REAL.

import type { SignalMeta } from "../../lib/signalMeta";
import type { Provenance, SignalFeedItem, SignalSourceType } from "../../lib/types";

export interface SignalRow extends SignalFeedItem {
  /** From the signal.received event join; null = outside the loaded events window. */
  resolvedProvenance: Provenance | null;
  /** Correlation id for the Chain Inspector; null = outside the loaded events window. */
  correlationId: string | null;
}

export function buildSignalRows(
  feed: readonly SignalFeedItem[],
  meta: ReadonlyMap<string, SignalMeta>,
): SignalRow[] {
  return feed.map((signal) => {
    const m = meta.get(signal.signal_id);
    return {
      ...signal,
      resolvedProvenance: signal.provenance ?? m?.provenance ?? null,
      correlationId: m?.correlationId ?? null,
    };
  });
}

export interface SignalFilters {
  /** Empty = all sources. */
  sources: readonly SignalSourceType[];
  provenance: Provenance | "ALL";
  /** Case-insensitive ticker substring; empty = no ticker filter. */
  ticker: string;
  /** Show ONLY unresolved-ticker rows (research data). */
  unresolvedOnly: boolean;
  /** Inclusive date bounds as `YYYY-MM-DD` (from <input type="date">), local time; null = open. */
  from: string | null;
  to: string | null;
}

export const EMPTY_FILTERS: SignalFilters = {
  sources: [],
  provenance: "ALL",
  ticker: "",
  unresolvedOnly: false,
  from: null,
  to: null,
};

export function hasActiveFilters(f: SignalFilters): boolean {
  return (
    f.sources.length > 0 ||
    f.provenance !== "ALL" ||
    f.ticker.trim() !== "" ||
    f.unresolvedOnly ||
    f.from !== null ||
    f.to !== null
  );
}

function dayBoundMs(day: string, end: boolean): number {
  const [y, m, d] = day.split("-").map(Number);
  if (!y || !m || !d) return end ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  const date = end ? new Date(y, m - 1, d, 23, 59, 59, 999) : new Date(y, m - 1, d, 0, 0, 0, 0);
  return date.getTime();
}

export function applySignalFilters(rows: readonly SignalRow[], f: SignalFilters): SignalRow[] {
  const tickerNeedle = f.ticker.trim().toUpperCase();
  const fromMs = f.from ? dayBoundMs(f.from, false) : null;
  const toMs = f.to ? dayBoundMs(f.to, true) : null;
  return rows.filter((row) => {
    if (f.sources.length > 0 && !f.sources.includes(row.source_type)) return false;
    if (f.provenance !== "ALL" && row.resolvedProvenance !== f.provenance) return false;
    if (f.unresolvedOnly && row.ticker !== null) return false;
    if (tickerNeedle !== "") {
      if (row.ticker === null || !row.ticker.toUpperCase().includes(tickerNeedle)) return false;
    }
    if (fromMs !== null || toMs !== null) {
      const t = Date.parse(row.occurred_at);
      if (Number.isNaN(t)) return false;
      if (fromMs !== null && t < fromMs) return false;
      if (toMs !== null && t > toMs) return false;
    }
    return true;
  });
}

// ---- `?provenance=` / `?source=` URL round-tripping (§7 addressable filter state) --------------

const SOURCE_VALUES: readonly SignalSourceType[] = ["statement", "contract", "disclosed_trade"];
const PROVENANCE_PARAM_VALUES: ReadonlySet<string> = new Set(["REAL", "BACKFILLED", "SYNTHETIC"]);

export function sourcesFromParam(param: string | null): SignalSourceType[] {
  if (!param) return [];
  return param
    .split(",")
    .map((s) => s.trim())
    .filter((s): s is SignalSourceType => (SOURCE_VALUES as readonly string[]).includes(s));
}

export function sourcesToParam(sources: readonly SignalSourceType[]): string | null {
  return sources.length > 0 ? sources.join(",") : null;
}

export function provenanceFromParam(param: string | null): Provenance | "ALL" {
  if (!param) return "ALL";
  const upper = param.toUpperCase();
  return PROVENANCE_PARAM_VALUES.has(upper) ? (upper as Provenance) : "ALL";
}
