// Event ticker (docs/UI-DESIGN.md §5.1) — rolling /ws/events tail, 24px rows, type-colored,
// virtualized with @tanstack/react-virtual (§10 — "the event ticker is its intended first
// consumer"), row insert 150ms slide (§3.4), rAF-coalesced WS batching (§3.5 "Coalesced live
// updates": one render per frame, not per message), click row -> Chain Inspector.
//
// Overview.tsx owns the actual WebSocket subscription (one `useEventStream` per screen, matching
// the rest of the app's pattern) and hands this component the growing event list; this component
// owns only rendering + the rAF batching of row-insert animation state.

import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useRef, useState } from "react";
import { formatTime, formatTimeUtc } from "../../lib/format";
import { colors, layout, shape, type as typeTokens } from "../../lib/tokens";
import type { StoredEventDto } from "../../lib/types";
import { eventTickerColor } from "./eventColors";

interface EventTickerProps {
  events: StoredEventDto[];
  onSelect: (event: StoredEventDto, rowEl: HTMLElement) => void;
  focusedEventId: string | null;
  onFocusRow: (event: StoredEventDto) => void;
}

export function EventTicker({ events, onSelect, focusedEventId, onFocusRow }: EventTickerProps) {
  const parentRef = useRef<HTMLDivElement>(null);

  // §3.5 "Coalesced live updates": batch the set of "just arrived" event ids on requestAnimationFrame
  // rather than re-triggering the insert animation once per incoming WS message. `events` itself is
  // already coalesced by App.tsx's refreshKey debounce (one fetch per burst); this second layer
  // coalesces which rows get the tp-ticker-row-enter class within a single rendered batch, so a
  // burst of N new rows plays one frame of inserts, not N staggered restyles.
  const [freshIds, setFreshIds] = useState<ReadonlySet<string>>(new Set());
  const knownIds = useRef<Set<string>>(new Set());
  const rafRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const incoming = events.filter((e) => !knownIds.current.has(e.event_id));
    if (incoming.length === 0) return;
    if (rafRef.current !== undefined) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      for (const e of incoming) knownIds.current.add(e.event_id);
      setFreshIds(new Set(incoming.map((e) => e.event_id)));
    });
    return () => {
      if (rafRef.current !== undefined) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- knownIds/rafRef are refs, not state
  }, [events]);

  // Newest-first per §5.1's "rolling tail" framing (matches Blotter's newest-first convention).
  const rows = [...events].reverse();

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => shape.rowHeightTicker,
    overscan: 12,
  });

  return (
    <div
      ref={parentRef}
      role="table"
      aria-label="Event ticker"
      style={{ height: "100%", overflowY: "auto", overflowX: "hidden" }}
    >
      {rows.length === 0 ? (
        <div
          style={{
            padding: layout.x4,
            color: colors.textMuted,
            fontFamily: "var(--font-sans)",
            fontSize: typeTokens.scale.body,
          }}
        >
          No events yet — they appear here as the platform records activity.
        </div>
      ) : (
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
          {virtualizer.getVirtualItems().map((item) => {
            const event = rows[item.index]!;
            const isFresh = freshIds.has(event.event_id);
            const isFocused = focusedEventId === event.event_id;
            return (
              <div
                key={event.event_id}
                role="row"
                tabIndex={0}
                data-event-id={event.event_id}
                className={isFresh ? "tp-ticker-row-enter" : undefined}
                onClick={(e) => onSelect(event, e.currentTarget)}
                onFocus={() => onFocusRow(event)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onSelect(event, e.currentTarget);
                }}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: item.size,
                  transform: `translateY(${item.start}px)`,
                  display: "flex",
                  alignItems: "center",
                  gap: layout.x2,
                  padding: `0 ${layout.x2}px`,
                  cursor: "pointer",
                  boxSizing: "border-box",
                  borderBottom: `1px solid ${colors.border}`,
                  background: isFocused ? colors.bgRaised : "transparent",
                  outline: isFocused ? `1px solid ${colors.accent}` : "none",
                  outlineOffset: -1,
                }}
              >
                <span
                  aria-hidden
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: shape.radiusBadge,
                    background: eventTickerColor(event),
                    flexShrink: 0,
                  }}
                />
                <span
                  className="tp-numeric"
                  title={formatTimeUtc(event.occurred_at)}
                  style={{
                    color: colors.textSecondary,
                    fontSize: typeTokens.scale.meta,
                    flexShrink: 0,
                    width: 76,
                  }}
                >
                  {formatTime(event.occurred_at).slice(7)}
                </span>
                <span
                  className="tp-numeric"
                  style={{
                    color: colors.textPrimary,
                    fontSize: typeTokens.scale.meta,
                    flexShrink: 0,
                    minWidth: 140,
                  }}
                >
                  {event.event_type}
                </span>
                <span
                  style={{
                    color: colors.textSecondary,
                    fontSize: typeTokens.scale.meta,
                    fontFamily: "var(--font-sans)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    flex: 1,
                  }}
                >
                  {payloadPreview(event)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function payloadPreview(event: StoredEventDto): string {
  const p = event.payload;
  const symbol = p.symbol;
  const ticker = p.ticker;
  const tag = typeof symbol === "string" ? symbol : typeof ticker === "string" ? ticker : null;
  return tag ? String(tag) : event.correlation_id.slice(0, 8);
}
