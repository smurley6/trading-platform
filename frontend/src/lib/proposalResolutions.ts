// Session-local proposal resolutions (UI Phase C §5.2; in lib/ because BOTH the Review screen
// and the shell's StatusBar consume it). The backend has NO reject endpoint —
// GET /proposals is a stateless read-only conviction scan, so a rejected proposal would simply
// come back on the next fetch. This tiny external store records which symbols the operator has
// resolved (approved or rejected) THIS SESSION so that (a) the Review queue drops them, and
// (b) the status bar's unreviewed count decrements live (StatusBar subscribes too — the two
// surfaces must agree without waiting a network round-trip; §3.5 optimistic UI).
//
// Approvals become durable server-side (the buy places orders, and the next scan's target-vs-
// held delta drops the row), so approved entries are pruned once the server stops proposing the
// symbol. Rejections are honestly session-only — a page reload (or backend rescan after new
// signals) resurrects them. Flagged as a known compromise in the Phase C report.
//
// Plain module-level store + subscribe, consumed via useSyncExternalStore (immutable Map
// snapshots so React sees referential changes).

export type ProposalResolution = "approved" | "rejected";

let snapshot: ReadonlyMap<string, ProposalResolution> = new Map();
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
}

export function resolveProposal(symbol: string, resolution: ProposalResolution): void {
  if (snapshot.get(symbol) === resolution) return;
  const next = new Map(snapshot);
  next.set(symbol, resolution);
  snapshot = next;
  emit();
}

/** Roll back an optimistic resolution (approve API call failed — §3.5 rollback). */
export function unresolveProposal(symbol: string): void {
  if (!snapshot.has(symbol)) return;
  const next = new Map(snapshot);
  next.delete(symbol);
  snapshot = next;
  emit();
}

/**
 * Drop resolutions for symbols the server no longer proposes (the approval landed server-side,
 * or the scan moved on) so the store never grows past the live proposal set.
 */
export function pruneResolutions(currentSymbols: ReadonlySet<string>): void {
  let changed = false;
  const next = new Map(snapshot);
  for (const symbol of next.keys()) {
    if (!currentSymbols.has(symbol)) {
      next.delete(symbol);
      changed = true;
    }
  }
  if (changed) {
    snapshot = next;
    emit();
  }
}

export function getResolutions(): ReadonlyMap<string, ProposalResolution> {
  return snapshot;
}

export function subscribeResolutions(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Test-only reset (the store is module-global). */
export function resetResolutionsForTest(): void {
  snapshot = new Map();
  emit();
}
