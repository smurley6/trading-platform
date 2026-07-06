// Event-ticker "type-colored" rows (docs/UI-DESIGN.md §5.1) — colors come ONLY from the §3.1
// reserved palette and its binding rules: --chain is event-lineage color (correlation/audit-trail
// events, which is what every row here fundamentally is), --gain/--loss are P&L direction/order
// side ONLY (so only order.filled — by side — and explicit rejections/failures qualify), --warn is
// degraded/drift, --accent is never used here (interactive-only, and ticker rows aren't itself
// the interactive affordance — the whole row's click handler is, styled neutrally per §3.1).
//
// Reuses chainSummary.ts's chainOutcome so the Chain Inspector and the event ticker agree on
// which events get gain/loss coloring — one rule, not two independently-maintained mappings.

import { colors } from "../../lib/tokens";
import type { StoredEventDto } from "../../lib/types";
import { chainOutcome } from "../../shell/chainSummary";

export function eventTickerColor(event: StoredEventDto): string {
  const outcome = chainOutcome(event);
  if (outcome === "gain") return colors.gain;
  if (outcome === "loss") return colors.loss;
  if (event.event_type === "reconciliation.drift") return colors.warn;
  // Every other lineage stage (signal.received, decision.made, risk.evaluated/approved,
  // order.intent/submitted/status_changed, portfolio.snapshot, …) reads as --chain: it's all
  // "about the event log" per §3.1's rule for the violet hue.
  return colors.chain;
}
