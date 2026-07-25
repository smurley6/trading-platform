// Live promotion (UI Phase D §5.5, Risk pane) — the per-sleeve readiness preflight + go-live
// control, refit from the legacy RiskDashboard onto the §3 token system. It lives inside the Risk
// pane because promoting to live is gated by exactly the risk caps/ceilings that pane shows: a
// sleeve can only go live when its configured caps sit under the hardened ceilings (the preflight).
// Go-live is a committing, real-money action, so it takes the §7 single-confirm ConfirmChip.

import { useState } from "react";
import { ApiError, api } from "../../lib/api";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../../lib/tokens";
import type { LiveReadiness, Sleeve } from "../../lib/types";
import { useToast } from "../../components/Toast";
import { ConfirmChip } from "../../shell/ConfirmChip";

export function LivePromotion({ sleeve, onChanged }: { sleeve: Sleeve; onChanged: () => void }) {
  const toast = useToast();
  const [readiness, setReadiness] = useState<LiveReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const isLive = sleeve.mode === "live";

  async function check(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      setReadiness(await api.liveReadiness(sleeve.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function setMode(mode: "live" | "paper"): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      const result = await api.setSleeveMode(sleeve.id, mode, mode === "live");
      setReadiness(result.readiness);
      toast.success(`${sleeve.id} → ${mode.toUpperCase()}`);
      onChanged();
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  const passed = readiness ? readiness.checks.filter((c) => c.passed).length : 0;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: layout.x2,
        padding: layout.x2,
        border: `1px solid ${isLive ? colors.warn : colors.border}`,
        borderRadius: shape.radiusPane,
        background: isLive ? withAlpha(colors.warn, 0.06) : "transparent",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: layout.x2 }}>
        <span
          className="tp-numeric"
          style={{ color: colors.textPrimary, fontWeight: 600, fontSize: typeTokens.scale.body }}
        >
          {sleeve.id}
        </span>
        <ModeBadge isLive={isLive} />
        <span style={{ flex: 1 }} />
        <SystemButton disabled={busy} onClick={() => void check()}>
          Check readiness
        </SystemButton>
        {isLive ? (
          <SystemButton disabled={busy} onClick={() => void setMode("paper")}>
            Return to paper
          </SystemButton>
        ) : confirming ? (
          <ConfirmChip
            label="Go live — real money"
            onConfirm={() => {
              setConfirming(false);
              void setMode("live");
            }}
            onCancel={() => setConfirming(false)}
          />
        ) : (
          <SystemButton
            disabled={busy || (readiness !== null && !readiness.ready)}
            onClick={() => setConfirming(true)}
          >
            Go live
          </SystemButton>
        )}
      </div>

      {error && (
        <span style={{ color: colors.loss, fontSize: typeTokens.scale.meta }}>{error}</span>
      )}

      {readiness && (
        <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
          <span
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: typeTokens.scale.meta,
              fontWeight: 600,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
              color: readiness.ready ? colors.textSecondary : colors.warn,
            }}
          >
            Preflight {passed}/{readiness.checks.length} passed
          </span>
          {readiness.checks.map((c) => (
            <div
              key={c.name}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: layout.x2,
                fontSize: typeTokens.scale.meta,
              }}
            >
              <span
                className="tp-numeric"
                aria-hidden
                style={{
                  width: layout.x3,
                  flexShrink: 0,
                  fontWeight: 700,
                  // Pass/fail are gate outcomes, not P&L: neutral for pass, --warn (degraded) for fail.
                  color: c.passed ? colors.textSecondary : colors.warn,
                }}
              >
                {c.passed ? "✓" : "✗"}
              </span>
              <span
                style={{
                  color: colors.textSecondary,
                  fontWeight: 600,
                  flexShrink: 0,
                  minWidth: layout.x8 * 3,
                }}
              >
                {c.name}
              </span>
              <span style={{ color: colors.textMuted }}>{c.detail}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ModeBadge({ isLive }: { isLive: boolean }) {
  // PAPER = neutral (the safe default); LIVE = --warn heightened-alert (never --loss, which §3.1
  // reserves for P&L/side/rejection/kill-switch — a live sleeve is "handle with care", not a loss).
  return (
    <span
      style={{
        border: `1px solid ${isLive ? colors.warn : colors.border}`,
        borderRadius: shape.radiusBadge,
        padding: `0 ${layout.x1}px`,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
        letterSpacing: "0.04em",
        color: isLive ? colors.warn : colors.textSecondary,
      }}
    >
      {isLive ? "LIVE" : "PAPER"}
    </span>
  );
}

export function SystemButton({
  children,
  disabled,
  onClick,
}: {
  children: React.ReactNode;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        cursor: disabled ? "default" : "pointer",
        height: shape.rowHeightTicker,
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusButton,
        background: "transparent",
        color: disabled ? colors.textMuted : colors.accent,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.body,
        fontWeight: 500,
        padding: `0 ${layout.x2}px`,
        transition: "border-color var(--duration-overlay) var(--ease-out)",
      }}
    >
      {children}
    </button>
  );
}
