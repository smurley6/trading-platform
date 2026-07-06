// 36px status bar (§4) — the glanceability contract, persistent on every screen:
//   mode chip (paper/live) · cycle mode (MANUAL neutral / AUTO amber) · buying power (live,
//   mono) · unreviewed-proposal count (amber chip, click -> Review) · WS connection dot
//   (green/amber/red) · kill switch (real switch, red when engaged, confirm-on-toggle).
//
// Reuses existing api.killSwitch/setKillSwitch, GET /capital, GET /proposals — no new backend
// endpoints (pre-resolved ambiguity #3). Buying-power fetch failure renders the FeedWarningBanner
// state convention (§7) instead of a blank/zeroed figure.

import { useState, useSyncExternalStore } from "react";
import { api } from "../lib/api";
import { getResolutions, subscribeResolutions } from "../lib/proposalResolutions";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../lib/tokens";
import type { Sleeve } from "../lib/types";
import { useLoader } from "../lib/useLoader";
import { ConfirmChip } from "./ConfirmChip";
import type { ScreenId } from "./useHistoryRoute";

interface StatusBarProps {
  sleeves: Sleeve[];
  refreshKey: number;
  /** "reconnecting" briefly after a drop, before the exponential-backoff reconnect resolves. */
  wsState: "connected" | "reconnecting" | "down";
  killEngaged: boolean;
  killBusy: boolean;
  onToggleKill: (next: boolean) => void;
  onNavigate: (screen: ScreenId) => void;
}

export function StatusBar({
  sleeves,
  refreshKey,
  wsState,
  killEngaged,
  killBusy,
  onToggleKill,
  onNavigate,
}: StatusBarProps) {
  const capital = useLoader(() => api.capital(), [refreshKey]);
  const proposals = useLoader(() => api.proposals(), [refreshKey]);
  const [confirmingKill, setConfirmingKill] = useState(false);

  const live = sleeves.some((s) => s.mode === "live");
  const cycleMode = proposals.data?.mode ?? null;
  // Unreviewed = the server's proposal set minus the symbols the operator already resolved this
  // session (Review's optimistic approve/reject store) — the count decrements in the same frame
  // as the queue row's exit animation (§5.2 "queue count in the status bar decrements live"); the
  // live event stream then makes it authoritative on the next refresh.
  const resolutions = useSyncExternalStore(subscribeResolutions, getResolutions);
  const unreviewedCount =
    proposals.data?.proposals.filter((p) => !resolutions.has(p.symbol)).length ?? 0;
  const buyingPower = capital.data?.enabled
    ? Number(capital.data.buying_power).toLocaleString(undefined, {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
      })
    : null;

  const dotColor =
    wsState === "connected" ? colors.gain : wsState === "reconnecting" ? colors.warn : colors.loss;

  return (
    <div
      className="tp-shell"
      style={{
        height: shape.statusBarHeight,
        display: "flex",
        alignItems: "center",
        gap: layout.x4,
        padding: `0 ${layout.x3}px`,
        borderBottom: `1px solid ${colors.border}`,
        background: colors.bgSurface,
        fontSize: typeTokens.scale.body,
        flexShrink: 0,
      }}
    >
      {/* Mode chip: paper/live — no reserved color (§3.1: --loss/--accent are exclusive-meaning,
          not decorative; a static mode label is neither P&L/order-side nor interactive). */}
      <span
        style={{
          fontFamily: "var(--font-sans)",
          fontWeight: 600,
          letterSpacing: "0.04em",
          color: colors.textPrimary,
        }}
      >
        {live ? "LIVE" : "PAPER"}
      </span>

      <Divider />

      {/* Cycle mode: MANUAL neutral / AUTO amber */}
      <span
        style={{
          fontFamily: "var(--font-sans)",
          fontWeight: 600,
          letterSpacing: "0.04em",
          color: cycleMode === "auto" ? colors.warn : colors.textSecondary,
        }}
      >
        {cycleMode ? cycleMode.toUpperCase() : "—"}
      </span>

      <Divider />

      {/* Buying power — live, mono */}
      {capital.data?.enabled === false ? (
        <span style={{ color: colors.warn, fontSize: typeTokens.scale.meta }} title="ENABLE_ALPACA_PAPER">
          BP disabled
        </span>
      ) : (
        <span className="tp-numeric" style={{ color: colors.textPrimary }}>
          BP {buyingPower ?? "—"}
        </span>
      )}

      <Divider />

      {/* Unreviewed proposal count — amber chip, click -> Review */}
      <button
        type="button"
        onClick={() => onNavigate("proposals")}
        disabled={unreviewedCount === 0}
        style={{
          cursor: unreviewedCount > 0 ? "pointer" : "default",
          border: "none",
          background: "transparent",
          padding: 0,
          fontFamily: "var(--font-sans)",
          fontWeight: 600,
          color: unreviewedCount > 0 ? colors.warn : colors.textMuted,
        }}
        title={unreviewedCount > 0 ? "Go to Review" : "No unreviewed proposals"}
      >
        {unreviewedCount > 0 ? `⚠ ${unreviewedCount} unreviewed` : "0 unreviewed"}
      </button>

      <div style={{ flex: 1 }} />

      {/* WS connection dot */}
      <span
        aria-label={`WebSocket ${wsState}`}
        title={`WebSocket ${wsState}`}
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: dotColor,
          flexShrink: 0,
        }}
      />

      <Divider />

      {/* Kill switch — real switch, red when engaged, confirm-on-toggle */}
      {confirmingKill ? (
        <ConfirmChip
          label={killEngaged ? "Release" : "Engage kill switch"}
          onConfirm={() => {
            setConfirmingKill(false);
            onToggleKill(!killEngaged);
          }}
          onCancel={() => setConfirmingKill(false)}
        />
      ) : (
        <button
          type="button"
          role="switch"
          aria-checked={killEngaged}
          aria-label="Kill switch"
          disabled={killBusy}
          onClick={() => setConfirmingKill(true)}
          style={{
            cursor: killBusy ? "default" : "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            border: `1px solid ${killEngaged ? colors.loss : colors.border}`,
            borderRadius: shape.radiusButton,
            background: killEngaged ? withAlpha(colors.loss, 0.14) : "transparent",
            color: killEngaged ? colors.loss : colors.textSecondary,
            fontFamily: "var(--font-sans)",
            fontSize: typeTokens.scale.meta,
            fontWeight: 600,
            padding: "2px 8px",
          }}
        >
          KILL {killEngaged ? "● ENGAGED" : "○"}
        </button>
      )}
    </div>
  );
}

function Divider() {
  return <span style={{ color: colors.border }}>·</span>;
}
