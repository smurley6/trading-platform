// Chain Inspector (docs/UI-DESIGN.md §6) — "the signature element". A right slide-over rendering
// `GET /events?correlation_id=…` as a vertical timeline in --chain violet: one node per event,
// each showing event type (mono), §3.2 timestamp, a payload summary line, and an expandable raw
// JSONB. Terminal states are colored by outcome (§3.1: --gain/--loss are P&L direction/order side
// ONLY, never lineage — see chainSummary.ts's chainOutcome for exactly which nodes qualify).
//
// Invoked with `c` on any focused row that has a correlation id, or by clicking an event-ticker
// row (Overview.tsx wires both). Stacks with the Ticker Pane via SlideOver's shared depth cue.

import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../lib/tokens";
import type { StoredEventDto } from "../lib/types";
import { chainOutcome, nodeTime, summarizeEvent } from "./chainSummary";
import { ShellEmptyState, SkeletonRows } from "./states";
import { SlideOver } from "./SlideOver";
import { useSlideOverDepth } from "./slideOverStack";

interface ChainInspectorProps {
  /** null = closed. A string correlation id opens the inspector and loads its chain. */
  correlationId: string | null;
  onClose: () => void;
  returnFocusRef?: React.RefObject<HTMLElement>;
}

export function ChainInspector({ correlationId, onClose, returnFocusRef }: ChainInspectorProps) {
  const open = correlationId !== null;
  const depth = useSlideOverDepth("chain-inspector", open);

  const [events, setEvents] = useState<StoredEventDto[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!correlationId) {
      setEvents(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setEvents(null);
    setError(null);
    api
      .eventsByCorrelation(correlationId)
      .then((result) => {
        if (!cancelled) setEvents(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [correlationId]);

  return (
    <SlideOver
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
      title={`Chain — ${correlationId ?? ""}`}
      depth={depth}
      returnFocusRef={returnFocusRef}
    >
      <div style={{ flex: 1, overflowY: "auto", padding: layout.x4 }}>
        {error ? (
          <ShellEmptyState message={`Could not load chain: ${error}`} />
        ) : events === null ? (
          <SkeletonRows rows={4} rowHeight={56} />
        ) : events.length === 0 ? (
          <ShellEmptyState message="No events recorded under this correlation id." />
        ) : (
          <ChainTimeline events={events} />
        )}
      </div>
    </SlideOver>
  );
}

function ChainTimeline({ events }: { events: StoredEventDto[] }) {
  // A correlation chain spans MULTIPLE streams (signal-*, decision-*, order-*, sleeve-*), each
  // with its own independent per-stream `seq` counter — sorting by `seq` alone interleaves them
  // wrong (e.g. a sleeve-stream `portfolio.snapshot` at seq=1 would sort before an order-stream
  // event at seq=2 even though it happened later). `recorded_at` is the event store's own
  // insertion-order wall-clock timestamp (monotonic across all streams, unlike `occurred_at`
  // which callers set), so it's the true causal order for a cross-stream timeline; `seq` is only
  // the tiebreaker for same-instant events on the same stream.
  const ordered = [...events].sort((a, b) => {
    const byRecorded = a.recorded_at.localeCompare(b.recorded_at);
    if (byRecorded !== 0) return byRecorded;
    return a.seq - b.seq;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {ordered.map((event, i) => (
        <ChainNode key={event.event_id} event={event} isLast={i === ordered.length - 1} />
      ))}
    </div>
  );
}

function ChainNode({ event, isLast }: { event: StoredEventDto; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const outcome = chainOutcome(event);
  const dotColor =
    outcome === "gain" ? colors.gain : outcome === "loss" ? colors.loss : colors.chain;

  return (
    <div style={{ display: "flex", gap: layout.x3 }}>
      {/* Rail: dot + connecting line, always --chain violet per §6/§3.1 (the LINE is lineage; only
          the dot's fill communicates outcome, on terminal nodes). */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: layout.x4,
          flexShrink: 0,
        }}
      >
        <span
          aria-hidden
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: dotColor,
            boxShadow: `0 0 0 2px ${withAlpha(colors.chain, 0.25)}`,
            marginTop: layout.x1,
            flexShrink: 0,
          }}
        />
        {!isLast && (
          <span
            aria-hidden
            style={{
              flex: 1,
              width: 1,
              background: withAlpha(colors.chain, 0.35),
              minHeight: shape.rowHeightTicker,
            }}
          />
        )}
      </div>

      <div style={{ paddingBottom: layout.x4, minWidth: 0, flex: 1 }}>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          style={{
            display: "block",
            width: "100%",
            textAlign: "left",
            cursor: "pointer",
            border: "none",
            background: "transparent",
            padding: 0,
            color: "inherit",
          }}
        >
          <div style={{ display: "flex", alignItems: "baseline", gap: layout.x2, flexWrap: "wrap" }}>
            <span
              className="tp-numeric"
              style={{ color: colors.chain, fontSize: typeTokens.scale.default, fontWeight: 600 }}
            >
              {event.event_type}
            </span>
            <span
              className="tp-numeric"
              title={event.occurred_at}
              style={{ color: colors.textSecondary, fontSize: typeTokens.scale.meta }}
            >
              {nodeTime(event)}
            </span>
          </div>
          <div
            style={{
              marginTop: 2,
              color: colors.textPrimary,
              fontFamily: "var(--font-sans)",
              fontSize: typeTokens.scale.body,
            }}
          >
            {summarizeEvent(event)}
          </div>
        </button>

        {expanded && (
          <pre
            className="tp-numeric"
            style={{
              marginTop: layout.x2,
              padding: layout.x2,
              borderRadius: shape.radiusPane,
              border: `1px solid ${colors.border}`,
              background: colors.bgSurface,
              color: colors.textSecondary,
              fontSize: typeTokens.scale.meta,
              overflowX: "auto",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {JSON.stringify(event.payload, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
