// Perspective theming adapter (docs/UI-DESIGN.md §3 / §10) — maps the §3 palette onto
// Perspective's documented CSS custom properties. Perspective ships named presets (we use
// "Pro Dark" as the closest base, set via the `theme` attribute on <perspective-viewer>, see
// components/StreamingTable.tsx) and lets every visual token be overridden per-instance via
// CSS custom properties on the element or an ancestor. This file is the single place that
// override mapping lives — do not hand-roll Perspective `--*` values elsewhere.
//
// Usage: spread `perspectiveCssVars` onto the <perspective-viewer> element's inline `style`
// (or a wrapping element) to pull it onto the §3 palette; StreamingTable.tsx is unchanged in
// Phase A (container/chrome only), so this adapter is available for Phase B+ call sites.

import { cssVars } from "../tokens";

/** Perspective CSS custom properties, one palette shared with Salt/AG Grid/Lightweight Charts. */
export const perspectiveCssVars: Record<string, string> = {
  "--plugin--background": cssVars["--bg-surface"],
  "--root--background": cssVars["--bg-base"],
  "--icon--color": cssVars["--text-secondary"],
  "--inactive--color": cssVars["--text-muted"],
  "--inactive--border-color": cssVars["--border"],
  "--active--color": cssVars["--accent"],
  "--error--color": cssVars["--loss"],
  "--warning--color": cssVars["--warn"],
  // Perspective's built-in positive/negative cell coloring (used by its own datagrid plugin
  // for numeric deltas) maps onto the reserved gain/loss semantics (§3.1) — never repurposed.
  "--rt-pos-cell--color": cssVars["--gain"],
  "--rt-neg-cell--color": cssVars["--loss"],
  "--interface-monospace--font-family": cssVars["--font-mono"],
  // Plain (non-custom-property) keys must be camelCase: this object is spread into a React
  // `style` prop (see StreamingTable.tsx), and React rejects kebab-case property names.
  fontFamily: cssVars["--font-sans"],
  color: cssVars["--text-primary"],
};

/** The closest built-in Perspective preset to base the palette overrides on. */
export const perspectiveBaseTheme = "Pro Dark";
