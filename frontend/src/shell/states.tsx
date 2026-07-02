// §7 state conventions — Skeleton / Empty / Error components. Exported for use across the new
// shell and future (Phase B+) screen rebuilds. Every value below comes from the §3 tokens
// (colors, layout, shape, motion) — zero hardcoded hex/px/ms.

import type { CSSProperties, ReactNode } from "react";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../lib/tokens";

// -------------------------------------------------------------------------------------------
// SkeletonRows — §7 loading convention: skeleton rows in grids (3 rows, shimmer disabled under
// reduced motion), never a full-screen spinner. Reserves the exact row height (§3.5 no layout
// shift) so real data swapping in causes zero reflow.
// -------------------------------------------------------------------------------------------
export function SkeletonRows({
  rows = 3,
  rowHeight = shape.rowHeightDense,
  style,
}: {
  rows?: number;
  rowHeight?: number;
  style?: CSSProperties;
}) {
  return (
    <div
      role="status"
      aria-label="Loading"
      style={{ display: "flex", flexDirection: "column", gap: layout.x1, ...style }}
    >
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={i}
          className="tp-skeleton-row"
          style={{
            height: rowHeight,
            borderRadius: shape.radiusPane,
            background: colors.bgSurface,
            border: `1px solid ${colors.border}`,
          }}
        />
      ))}
    </div>
  );
}

// -------------------------------------------------------------------------------------------
// EmptyState — §7 empty convention: one muted sentence + the next action.
// -------------------------------------------------------------------------------------------
export function ShellEmptyState({
  message,
  action,
  style,
}: {
  message: ReactNode;
  action?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: layout.x2,
        padding: `${layout.x6}px ${layout.x4}px`,
        textAlign: "center",
        ...style,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.default,
          color: colors.textMuted,
        }}
      >
        {message}
      </div>
      {action != null && <div>{action}</div>}
    </div>
  );
}

// -------------------------------------------------------------------------------------------
// FeedWarningBanner — §7 errors/disabled-feeds convention: api.ts already resolves failures to
// disabled shapes (see lib/api.ts — quotes/livePositions/capital `.catch(() => ({enabled:
// false, ...}))`); render that as an inline --warn banner naming the feed and the config flag
// that enables it, never a blank pane.
// -------------------------------------------------------------------------------------------
export function FeedWarningBanner({
  feed,
  configFlag,
  style,
}: {
  /** Human name of the disabled/failing feed, e.g. "Buying power". */
  feed: string;
  /** The env/config flag that enables it, e.g. "ENABLE_ALPACA_PAPER". */
  configFlag: string;
  style?: CSSProperties;
}) {
  return (
    <div
      role="status"
      style={{
        display: "flex",
        alignItems: "center",
        gap: layout.x2,
        padding: `${layout.x1}px ${layout.x3}px`,
        borderRadius: shape.radiusPane,
        border: `1px solid ${colors.warn}`,
        background: withAlpha(colors.warn, 0.08), // --warn at low alpha, banner fill only
        color: colors.warn,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.body,
        ...style,
      }}
    >
      <span aria-hidden style={{ fontWeight: 600 }}>
        ⚠
      </span>
      <span>
        {feed} is disabled — set <code style={{ fontFamily: "var(--font-mono)" }}>{configFlag}</code>{" "}
        to enable it.
      </span>
    </div>
  );
}

