// Shared design tokens — the single source of truth for the platform's visual language.
//
// UI Phase A (2026-07-02, docs/UI-DESIGN.md): this file now carries TWO token systems side by
// side:
//
//   1. The §3 operator-console system below (`tokens`, `cssVars`) — desaturated blue-slate ink
//      ramp, reserved semantic colors, IBM Plex type scale, 4px space unit, the §3.4/3.5 motion
//      constants. This is what the new app shell (status bar, nav rail, palette, state
//      components) and all future screens are built from. `cssVars` renders it to CSS custom
//      properties (see injectThemeVars in App.tsx) so AG Grid/Perspective/Lightweight Charts
//      adapters and plain CSS can consume the same palette.
//
//   2. The legacy "trading terminal meets premium SaaS" system below that (`color`, `space`,
//      `radius`, `z`, `font`, `text`, `shadow`, `motion`, `transitions`, `cardStyle`,
//      `sectionHeaderStyle`, `sectionTitleStyle`, `eyebrowStyle`, `numeric`, `pillStyle`,
//      `pnlColor`) — still imported by all 7 existing screens and their shared components. Per
//      the Phase A spec, existing screens keep their internals unchanged in this phase
//      (container/chrome/theming only), so every one of these exports is CONTRACT: unchanged in
//      name, shape, and value. Do not remove or rename them. Note the §3 system's own space
//      scale is exported as `layout` (not `space`) specifically to avoid colliding with this
//      legacy `space` export.
//
// Phase B/C will migrate screen internals onto the §3 system; until then both systems coexist.

import type { CSSProperties } from "react";

// =================================================================================================
// §3 token system (docs/UI-DESIGN.md §3) — new app shell + future screens.
// =================================================================================================

/** §3.1 — three-step blue-slate ink ramp + reserved semantic colors. */
export const colors = {
  bgBase: "#0E1116",
  bgSurface: "#151A21",
  bgRaised: "#1C232C",
  border: "#262E38",
  textPrimary: "#E8EAED",
  textSecondary: "#98A2AD",
  textMuted: "#5C6670",

  // Reserved — exclusive meaning, never decorative (§3.1).
  gain: "#2ECC8F",
  loss: "#F0544F",
  warn: "#E5A83B",
  accent: "#4E9CB0",
  chain: "#9D8CFF",
  synthetic: "#D66BA0",
} as const;

/**
 * Derive an `rgba(...)` string from one of the §3.1 hex tokens instead of hand-copying hex
 * digits into a literal (which silently drifts if the token ever changes). `alpha` is 0-1.
 */
export function withAlpha(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  const r = parseInt(normalized.slice(0, 2), 16);
  const g = parseInt(normalized.slice(2, 4), 16);
  const b = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** §3.2 — IBM Plex type scale (px) and font stacks. */
export const type = {
  fontSans:
    '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  fontMono:
    '"IBM Plex Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
  scale: {
    meta: 11, // dense grid meta
    body: 12, // grid body, labels
    default: 13, // default UI
    paneTitle: 15, // pane titles
    screenTitle: 20, // screen title — one per screen, top-left, quiet
    // Not in §3.2's scale — added in the Phase A fix cycle because 15/20 both misuse
    // pane-title/screen-title semantics when applied to a small nav-rail icon glyph. One
    // explicit exception, documented here and in the MEMORY.md changelog.
    icon: 16, // nav-rail icon glyphs only
  },
} as const;

/** §3.3 — 4px base unit, row heights, radii. Named `layout` to avoid clashing with the legacy
 *  `space` export (contracted below) during the Phase A→C transition. */
export const layout = {
  unit: 4,
  x1: 4,
  x2: 8,
  x3: 12,
  x4: 16,
  x5: 20,
  x6: 24,
  x8: 32,
} as const;

export const shape = {
  radiusPane: 4,
  radiusBadge: 2,
  radiusButton: 4,
  radiusMax: 4,
  rowHeightDense: 28,
  rowHeightTicker: 24,
  statusBarHeight: 36,
  navRailWidth: 48,
  overlayShadow: "0 8px 24px rgba(0,0,0,0.5)",
} as const;

/** §3.4 / §3.5 — the one sanctioned easing system + durations. No other curves/durations. */
export const motionTokens = {
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
  easeIn: "cubic-bezier(0.7, 0, 0.84, 0)",
  durationOverlay: 120, // ms — overlays/hover
  durationRow: 150, // ms — row ops (ticker insert)
  durationFlash: 400, // ms — data flash decay
} as const;

/** The full §3 token object, grouped for convenience. */
export const tokens = {
  colors,
  type,
  space: layout,
  shape,
  motion: motionTokens,
} as const;

/**
 * Render the §3 tokens as a flat CSS custom-property map (`--bg-base`, `--gain`, `--ease-out`,
 * etc. — exact names from UI-DESIGN.md §3.1-3.4) for injection into `:root`. One palette shared
 * by Salt, AG Grid, Perspective, and Lightweight Charts adapters plus plain CSS.
 */
export const cssVars = {
  "--bg-base": colors.bgBase,
  "--bg-surface": colors.bgSurface,
  "--bg-raised": colors.bgRaised,
  "--border": colors.border,
  "--text-primary": colors.textPrimary,
  "--text-secondary": colors.textSecondary,
  "--text-muted": colors.textMuted,
  "--gain": colors.gain,
  "--loss": colors.loss,
  "--warn": colors.warn,
  "--accent": colors.accent,
  "--chain": colors.chain,
  "--synthetic": colors.synthetic,

  "--font-sans": type.fontSans,
  "--font-mono": type.fontMono,
  "--font-size-meta": `${type.scale.meta}px`,
  "--font-size-body": `${type.scale.body}px`,
  "--font-size-default": `${type.scale.default}px`,
  "--font-size-pane-title": `${type.scale.paneTitle}px`,
  "--font-size-screen-title": `${type.scale.screenTitle}px`,
  "--font-size-icon": `${type.scale.icon}px`,

  "--space-unit": `${layout.unit}px`,
  "--row-height-dense": `${shape.rowHeightDense}px`,
  "--row-height-ticker": `${shape.rowHeightTicker}px`,
  "--status-bar-height": `${shape.statusBarHeight}px`,
  "--nav-rail-width": `${shape.navRailWidth}px`,
  "--radius-pane": `${shape.radiusPane}px`,
  "--radius-badge": `${shape.radiusBadge}px`,
  "--radius-button": `${shape.radiusButton}px`,
  "--overlay-shadow": shape.overlayShadow,

  "--ease-out": motionTokens.easeOut,
  "--ease-in": motionTokens.easeIn,
  "--duration-overlay": `${motionTokens.durationOverlay}ms`,
  "--duration-row": `${motionTokens.durationRow}ms`,
  "--duration-flash": `${motionTokens.durationFlash}ms`,
};

// =================================================================================================
// Legacy token system — CONTRACT, unchanged. Every screen and shared component under
// components/ and features/ (outside this Phase A change set) still imports from here.
// =================================================================================================

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

