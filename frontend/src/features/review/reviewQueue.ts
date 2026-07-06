// Review queue ordering + age display (UI Phase C §5.2) — pure, unit-tested.
//
// §5.2: "Queue ordering: conviction desc, then age." A ProposalItem carries no timestamp of its
// own (GET /proposals is a stateless scan), so a proposal's AGE is derived from its newest
// contributing signal's disclosure timestamp (the moment the actionable evidence appeared) —
// resolved against the loaded signal feed. Proposals whose contributing signals fall outside the
// loaded feed window sort after dated ones at equal conviction (unknown age = oldest evidence
// unknown, so they wait their turn) with symbol as the final deterministic tiebreaker.

import type { ProposalResolution } from "../../lib/proposalResolutions";
import type { ProposalItem, SignalFeedItem } from "../../lib/types";

/** Newest contributing signal's occurred_at (ms epoch), or null when none resolve. */
export function newestSignalAtMs(
  proposal: ProposalItem,
  signalsById: ReadonlyMap<string, SignalFeedItem>,
): number | null {
  let newest: number | null = null;
  for (const id of proposal.contributing_signal_ids) {
    const sig = signalsById.get(id);
    if (!sig) continue;
    const t = Date.parse(sig.occurred_at);
    if (Number.isNaN(t)) continue;
    if (newest === null || t > newest) newest = t;
  }
  return newest;
}

/**
 * §5.2 queue order: conviction desc, then age (older evidence first — it has waited longest),
 * then symbol for a stable, deterministic order. Resolved (approved/rejected this session)
 * symbols are dropped.
 */
export function orderQueue(
  proposals: readonly ProposalItem[],
  signalsById: ReadonlyMap<string, SignalFeedItem>,
  resolutions: ReadonlyMap<string, ProposalResolution>,
): ProposalItem[] {
  return proposals
    .filter((p) => !resolutions.has(p.symbol))
    .map((p) => ({ p, at: newestSignalAtMs(p, signalsById) }))
    .sort((a, b) => {
      if (b.p.conviction_score !== a.p.conviction_score) {
        return b.p.conviction_score - a.p.conviction_score;
      }
      const aAt = a.at ?? Number.POSITIVE_INFINITY; // unknown age sorts last at equal conviction
      const bAt = b.at ?? Number.POSITIVE_INFINITY;
      if (aAt !== bAt) return aAt - bAt;
      return a.p.symbol.localeCompare(b.p.symbol);
    })
    .map((entry) => entry.p);
}

/** Compact age label for the queue ("2h", "35m", "3d"); em-dash when age is unknown. */
export function formatAge(atMs: number | null, nowMs: number): string {
  if (atMs === null) return "—";
  const deltaSec = Math.max(0, Math.floor((nowMs - atMs) / 1000));
  if (deltaSec < 60) return `${deltaSec}s`;
  const minutes = Math.floor(deltaSec / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}
