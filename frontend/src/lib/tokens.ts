// Shared design tokens. The app historically hardcoded the same hexes/rgba inline in every
// component; centralizing them here keeps the look consistent and makes a restyle a one-file
// change. New components import from here; existing ones are migrated incrementally.

import type { CSSProperties } from "react";

export const color = {
  // Brand / interactive
  accent: "#4c9aff",
  accentText: "#7cb3ff",
  accentSoft: "rgba(76,154,255,0.2)",
  // Semantic
  ok: "#4ade80",
  okBorder: "#22c55e",
  okSoft: "rgba(34,197,94,0.16)",
  danger: "#ef4444",
  dangerText: "#f87171",
  dangerSoft: "rgba(239,68,68,0.2)",
  warn: "#fbbf24",
  // Neutrals
  text: "#e2e8f0",
  textDim: "#cbd5e1",
  muted: "#94a3b8",
  faintBorder: "rgba(148,163,184,0.3)",
  hairline: "rgba(255,255,255,0.12)",
  panel: "#0f172a",
  panelSoft: "rgba(148,163,184,0.08)",
} as const;

export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 } as const;

export const radius = { sm: 6, md: 8, lg: 10, pill: 12 } as const;

export const z = { strip: 30, popover: 50, toast: 1000, modal: 1100 } as const;

/** The recurring "status pill" style (PAPER/LIVE, ARMED/HALTED, badges). */
export function pillStyle(tone: "accent" | "ok" | "danger" | "muted"): CSSProperties {
  const map = {
    accent: { bg: color.accentSoft, fg: color.accentText, border: color.accent },
    ok: { bg: color.okSoft, fg: color.ok, border: color.okBorder },
    danger: { bg: color.dangerSoft, fg: color.dangerText, border: color.danger },
    muted: { bg: "rgba(148,163,184,0.15)", fg: color.muted, border: color.faintBorder },
  }[tone];
  return {
    padding: "2px 8px",
    borderRadius: radius.pill,
    fontSize: 11,
    fontWeight: 700,
    whiteSpace: "nowrap",
    background: map.bg,
    color: map.fg,
    border: `1px solid ${map.border}`,
  };
}

/** Green for non-negative, red for negative — the standard P&L coloring. */
export function pnlColor(value: number): string {
  return value >= 0 ? color.ok : color.dangerText;
}
