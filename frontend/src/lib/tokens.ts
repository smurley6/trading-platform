// Shared design tokens — the single source of truth for the platform's visual language.
// The app historically hardcoded the same hexes/rgba inline in every component; centralizing
// them here keeps the look consistent and makes a restyle a one-file change. Feature screens and
// shared components import from here, so changing a VALUE below propagates the new look app-wide.
//
// Design direction: "trading terminal meets premium SaaS" — a confident, deep dark palette with
// one strong electric-blue accent, crisp semantic P&L colors, strong typographic hierarchy,
// tabular/mono numerals for every figure, purposeful density, and subtle depth (hairline borders
// + soft shadows, never heavy chrome).
//
// IMPORTANT (contract): existing exports (color.*, space.*, radius.*, z.*, pillStyle, pnlColor)
// must never be removed or renamed — screens depend on them. New tokens/helpers may be added
// freely, and existing token VALUES may evolve.

import type { CSSProperties } from "react";

export const color = {
  // Brand / interactive — electric azure accent.
  accent: "#3d8bff",
  accentText: "#8fbaff",
  accentSoft: "rgba(61,139,255,0.16)",
  accentStrong: "#5a9dff",
  accentBorder: "rgba(61,139,255,0.55)",

  // Semantic — P&L greens / reds tuned for a dark surface.
  ok: "#34d399",
  okBorder: "#10b981",
  okSoft: "rgba(16,185,129,0.15)",
  danger: "#f2544d",
  dangerText: "#ff6b6b",
  dangerSoft: "rgba(242,84,77,0.16)",
  warn: "#fbbf24",
  warnText: "#fcd34d",
  warnSoft: "rgba(251,191,36,0.15)",

  // Neutrals — text ramp from bright headings to dim captions.
  text: "#eef2f9", // bright heading text
  textDim: "#c6d0e0",
  muted: "#8b97ac",
  faint: "#5f6b80",

  // Borders / dividers.
  faintBorder: "rgba(148,163,184,0.22)",
  hairline: "rgba(255,255,255,0.08)",
  hairlineStrong: "rgba(255,255,255,0.14)",

  // Surfaces — layered background ramp (app -> panel -> raised -> inset).
  bg: "#0a0d14",
  bgGradientTop: "#0d1119",
  panel: "#111725",
  panelRaised: "#161d2e",
  panelSoft: "rgba(148,163,184,0.06)",
  panelInset: "rgba(6,10,18,0.55)",
  overlay: "rgba(4,7,13,0.66)",
} as const;

export const space = { xxs: 2, xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32, xxxl: 48 } as const;

export const radius = { xs: 4, sm: 6, md: 8, lg: 12, xl: 16, pill: 999 } as const;

export const z = { strip: 30, popover: 50, toast: 1000, modal: 1100 } as const;

// Font stacks. `sans` is the UI face; `mono` is used for every numeric figure so columns align
// and prices read like a terminal. Ship a tabular/mono treatment via `numeric`.
export const font = {
  sans: [
    "Inter",
    "system-ui",
    "-apple-system",
    "Segoe UI",
    "Roboto",
    "Helvetica",
    "Arial",
    "sans-serif",
  ].join(", "),
  mono: [
    "ui-monospace",
    "SFMono-Regular",
    "SF Mono",
    "JetBrains Mono",
    "Menlo",
    "Consolas",
    "Liberation Mono",
    "monospace",
  ].join(", "),
} as const;

// Type scale (px) with sensible weights/tracking. Screens should reach for `text.*` rather than
// hardcoding font sizes so hierarchy stays consistent.
export const text = {
  display: { fontSize: 28, fontWeight: 700, lineHeight: 1.15, letterSpacing: "-0.02em" },
  h1: { fontSize: 22, fontWeight: 700, lineHeight: 1.2, letterSpacing: "-0.01em" },
  h2: { fontSize: 18, fontWeight: 700, lineHeight: 1.25, letterSpacing: "-0.01em" },
  h3: { fontSize: 15, fontWeight: 700, lineHeight: 1.3 },
  body: { fontSize: 14, fontWeight: 400, lineHeight: 1.5 },
  label: { fontSize: 13, fontWeight: 500, lineHeight: 1.4 },
  caption: { fontSize: 12, fontWeight: 500, lineHeight: 1.4 },
  micro: { fontSize: 11, fontWeight: 600, lineHeight: 1.3, letterSpacing: "0.04em" },
} as const satisfies Record<string, CSSProperties>;

// Elevation / shadow ramp. `sm` for resting cards, `md` for raised, `lg` for popovers/modals,
// plus a focus ring and an accent glow for LIVE / active emphasis.
export const shadow = {
  sm: "0 1px 2px rgba(0,0,0,0.4), 0 1px 1px rgba(0,0,0,0.3)",
  md: "0 4px 16px rgba(0,0,0,0.4)",
  lg: "0 16px 48px rgba(0,0,0,0.55)",
  popover: "0 12px 34px rgba(0,0,0,0.5)",
  focus: `0 0 0 3px ${color.accentSoft}`,
  glowAccent: "0 0 0 1px rgba(61,139,255,0.35), 0 0 18px rgba(61,139,255,0.28)",
  glowDanger: "0 0 0 1px rgba(242,84,77,0.4), 0 0 18px rgba(242,84,77,0.3)",
} as const;

// Motion timings + easings for restrained micro-interactions.
export const motion = {
  fast: "120ms",
  base: "180ms",
  slow: "280ms",
  ease: "cubic-bezier(0.4, 0, 0.2, 1)",
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
} as const;

/** Standard transition string, e.g. `transition: transitions("background", "border-color")`. */
export function transitions(...props: string[]): string {
  return (props.length ? props : ["all"])
    .map((p) => `${p} ${motion.base} ${motion.ease}`)
    .join(", ");
}

// ---------------------------------------------------------------------------------------------
// Shared style helpers — the recurring "shapes" screens compose. Prefer these over re-deriving
// padding/border/shadow inline so every card, header, and figure reads as one system.
// ---------------------------------------------------------------------------------------------

/** A raised content panel: hairline border, soft shadow, layered surface. */
export function cardStyle(opts?: { padded?: boolean; raised?: boolean }): CSSProperties {
  const padded = opts?.padded ?? true;
  return {
    background: opts?.raised ? color.panelRaised : color.panel,
    border: `1px solid ${color.hairline}`,
    borderRadius: radius.lg,
    boxShadow: opts?.raised ? shadow.md : shadow.sm,
    padding: padded ? space.xl : 0,
  };
}

/** The row wrapping a section title (+ optional actions on the right). */
export function sectionHeaderStyle(): CSSProperties {
  return {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: space.md,
    marginBottom: space.lg,
  };
}

/** Section title typography (use for the H3-level heading inside a card). */
export function sectionTitleStyle(): CSSProperties {
  return { ...text.h3, margin: 0, color: color.text };
}

/** Small uppercase caption/eyebrow above a figure or beside a title. */
export function eyebrowStyle(): CSSProperties {
  return {
    ...text.micro,
    textTransform: "uppercase",
    color: color.muted,
  };
}

/** Monospace + tabular-nums treatment for any numeric figure so columns align. */
export function numeric(extra?: CSSProperties): CSSProperties {
  return {
    fontFamily: font.mono,
    fontVariantNumeric: "tabular-nums",
    fontFeatureSettings: '"tnum" 1, "zero" 1',
    ...extra,
  };
}

/** The recurring "status pill" style (PAPER/LIVE, ARMED/HALTED, badges). */
export function pillStyle(tone: "accent" | "ok" | "danger" | "warn" | "muted"): CSSProperties {
  const map = {
    accent: { bg: color.accentSoft, fg: color.accentText, border: color.accentBorder },
    ok: { bg: color.okSoft, fg: color.ok, border: color.okBorder },
    danger: { bg: color.dangerSoft, fg: color.dangerText, border: color.danger },
    warn: { bg: color.warnSoft, fg: color.warnText, border: color.warn },
    muted: { bg: "rgba(148,163,184,0.12)", fg: color.muted, border: color.faintBorder },
  }[tone];
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "3px 10px",
    borderRadius: radius.pill,
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.04em",
    whiteSpace: "nowrap",
    lineHeight: 1.4,
    background: map.bg,
    color: map.fg,
    border: `1px solid ${map.border}`,
  };
}

/** Green for non-negative, red for negative — the standard P&L coloring. */
export function pnlColor(value: number): string {
  return value >= 0 ? color.ok : color.dangerText;
}
