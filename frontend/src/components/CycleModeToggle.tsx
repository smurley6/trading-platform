// Live arm/disarm of the scheduled cycle's trade mode, mirroring the Paper/Live ModeToggle
// pattern. MANUAL (safe, collect-only: the cycle only surfaces proposals) <-> AUTO (the cycle
// places paper orders automatically each tick). Switching to AUTO is gated behind an explicit
// confirm; switching back to MANUAL is immediate (the safe direction). The backend flips the
// running cycle live (POST /cycle/mode), so the next tick honours it with no redeploy.

import { Button, Text } from "@salt-ds/core";
import { useState } from "react";
import { api } from "../lib/api";
import { color, radius, shadow, space, z } from "../lib/tokens";
import type { CycleMode } from "../lib/types";

interface CycleModeToggleProps {
  mode: CycleMode;
  onChanged: () => void;
}

export function CycleModeToggle({ mode, onChanged }: CycleModeToggleProps) {
  const isAuto = mode === "auto";
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function apply(next: CycleMode): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      await api.setCycleMode(next);
      onChanged();
      setConfirming(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not change cycle mode");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ position: "relative" }}>
      <div
        role="tablist"
        aria-label="Cycle trade mode"
        style={{
          display: "inline-flex",
          padding: 2,
          borderRadius: radius.lg,
          background: color.panelInset,
          border: `1px solid ${color.hairline}`,
        }}
      >
        <Segment
          label="MANUAL"
          active={!isAuto}
          tone="safe"
          disabled={busy}
          onClick={() => {
            if (isAuto) void apply("manual");
          }}
        />
        <Segment
          label="AUTO"
          active={isAuto}
          tone="danger"
          disabled={busy}
          onClick={() => {
            if (!isAuto) setConfirming(true);
          }}
        />
      </div>

      {confirming && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            left: 0,
            zIndex: z.popover,
            width: 320,
            padding: space.lg,
            borderRadius: radius.lg,
            background: color.panelRaised,
            border: `1px solid ${color.faintBorder}`,
            boxShadow: shadow.popover,
          }}
        >
          <strong style={{ fontSize: 14 }}>Arm autonomous trading?</strong>
          <Text
            styleAs="label"
            color="secondary"
            style={{ display: "block", margin: "6px 0 12px" }}
          >
            AUTO places paper orders automatically each cycle when conviction clears the trade
            threshold. You can switch back to MANUAL at any time.
          </Text>
          {error && (
            <Text color="error" styleAs="label" style={{ display: "block", marginBottom: 8 }}>
              {error}
            </Text>
          )}
          <div style={{ display: "flex", gap: 8 }}>
            <Button variant="cta" disabled={busy} onClick={() => void apply("auto")}>
              Arm auto-trading
            </Button>
            <Button variant="secondary" disabled={busy} onClick={() => setConfirming(false)}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function Segment({
  label,
  active,
  tone,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  tone: "safe" | "danger";
  disabled: boolean;
  onClick: () => void;
}) {
  const activeBg = tone === "danger" ? color.dangerSoft : color.accentSoft;
  const activeColor = tone === "danger" ? color.dangerText : color.accentText;
  const activeBorder = tone === "danger" ? color.danger : color.accentBorder;
  return (
    <button
      role="tab"
      aria-selected={active}
      disabled={disabled}
      onClick={onClick}
      style={{
        cursor: disabled ? "default" : "pointer",
        padding: "4px 14px",
        borderRadius: radius.md,
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: "0.04em",
        background: active ? activeBg : "transparent",
        color: active ? activeColor : color.muted,
        border: `1px solid ${active ? activeBorder : "transparent"}`,
      }}
    >
      {label}
    </button>
  );
}
