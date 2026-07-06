// Conviction decomposition for the Review evidence pane (UI Phase C §5.2) — pure, unit-tested.
//
// §5.2: "decomposes conviction into its actual scoring terms — the operator should never approve
// a number he can't see the derivation of." The backend's conviction engine
// (domain/conviction.py) computes: score = clamp(Σ decay·sentiment·weight·reliability ·
// corroboration_factor · scale, 0, 100). GET /proposals carries ONLY the final score, the
// rationale sentence, and the contributing signal ids — NOT the engine's component breakdown
// (raw/decayed weighted sums, corroboration factor, per-signal contributions), and no endpoint
// exposes the tunables (scale, decay half-life, corroboration bonus).
//
// So this module derives exactly the terms that ARE real and client-visible, and nothing more:
// per contributing signal, the documented raw contribution term sentiment × politician_weight ×
// source_reliability (the engine's per-signal term with decay weight 1 — decay, when enabled, is
// applied server-side and is NOT reproduced here), plus the distinct source types that feed the
// corroboration factor. The engine-side multipliers we cannot see are surfaced as an explicit
// "not exposed" line in the pane rather than invented. Gap flagged in the Phase C report.

import type { SignalFeedItem } from "../../lib/types";

export interface SignalTerm {
  signalId: string;
  sourceType: string;
  /** Filer/insider — politician_id, or null when the source has none. */
  filer: string | null;
  /** ISO disclosure timestamp (occurred_at). */
  disclosedAt: string;
  sentiment: number;
  weight: number;
  reliability: number;
  /** sentiment × weight × reliability — the engine's raw (undecayed) per-signal term. */
  contribution: number;
  /** Primary source URL from the raw payload (EDGAR/Clerk/news link), when the source carries one. */
  url: string | null;
}

export interface ConvictionTerms {
  /** Contributing signals resolved against the loaded feed, newest first. */
  resolved: SignalTerm[];
  /** Contributing signal ids that fall outside the loaded feed window. */
  missingIds: string[];
  /** Distinct source types among resolved signals — the corroboration factor's input. */
  distinctSources: string[];
  /** Σ contribution over resolved signals (raw weighted sum, undecayed, resolved subset only). */
  rawWeightedSum: number;
}

// Raw-payload keys the live adapters use for their primary source link (sec_form4: source_url;
// news/sec_edgar_atom: link; misc: url).
const URL_KEYS = ["source_url", "url", "link", "filing_url"] as const;

export function primarySourceUrl(signal: SignalFeedItem): string | null {
  const rp = signal.raw_payload;
  if (!rp) return null;
  for (const key of URL_KEYS) {
    const v = rp[key];
    if (typeof v === "string" && /^https?:\/\//i.test(v)) return v;
  }
  return null;
}

export function convictionTerms(
  contributingSignalIds: readonly string[],
  signalsById: ReadonlyMap<string, SignalFeedItem>,
): ConvictionTerms {
  const resolved: SignalTerm[] = [];
  const missingIds: string[] = [];
  for (const id of contributingSignalIds) {
    const sig = signalsById.get(id);
    if (!sig) {
      missingIds.push(id);
      continue;
    }
    resolved.push({
      signalId: sig.signal_id,
      sourceType: sig.source_type,
      filer: sig.politician_id,
      disclosedAt: sig.occurred_at,
      sentiment: sig.sentiment,
      weight: sig.politician_weight,
      reliability: sig.source_reliability,
      contribution: sig.sentiment * sig.politician_weight * sig.source_reliability,
      url: primarySourceUrl(sig),
    });
  }
  resolved.sort((a, b) => b.disclosedAt.localeCompare(a.disclosedAt));
  const distinctSources = [...new Set(resolved.map((t) => t.sourceType))];
  const rawWeightedSum = resolved.reduce((sum, t) => sum + t.contribution, 0);
  return { resolved, missingIds, distinctSources, rawWeightedSum };
}
