// AG Grid theming adapter (docs/UI-DESIGN.md §3 / §10) — maps the §3 palette into AG Grid's
// theming surface. AG Grid 32.x (pinned in package.json) predates v33's JS Theming API, so its
// theming surface is the documented `--ag-*` CSS custom property set, applied via a stylesheet
// class. The `.ag-theme-tp` rule that consumes `agGridCssVars` below does not exist yet — it
// lands in Phase B when the first grid is rebuilt onto the §3 system. This file is the single
// place the mapping lives — do not hand-roll `--ag-*` values elsewhere.
//
// Usage: apply className `ag-theme-tp` (instead of the legacy `ag-theme-quartz-dark`) to any new
// AG Grid instance built in Phase B+; existing screens keep `ag-theme-quartz-dark` unchanged in
// this phase per the Phase A scope (container/chrome only, no screen-internals changes).

import { cssVars } from "../tokens";

/**
 * AG Grid CSS custom properties, one palette shared with Salt/Perspective/Lightweight Charts.
 * Not yet consumed — the `.ag-theme-tp` rule that applies these lands in Phase B. Kept here now
 * as the documented source so a design-tokens change only requires editing tokens.ts.
 */
export const agGridCssVars: Record<string, string> = {
  "--ag-background-color": "transparent",
  "--ag-foreground-color": cssVars["--text-primary"],
  "--ag-header-background-color": cssVars["--bg-surface"],
  "--ag-header-foreground-color": cssVars["--text-secondary"],
  "--ag-odd-row-background-color": cssVars["--bg-surface"],
  "--ag-row-hover-color": cssVars["--bg-raised"],
  "--ag-selected-row-background-color": cssVars["--bg-raised"],
  "--ag-border-color": cssVars["--border"],
  "--ag-row-border-color": cssVars["--border"],
  "--ag-header-column-separator-color": "transparent",
  "--ag-input-focus-border-color": cssVars["--accent"],
  "--ag-range-selection-border-color": cssVars["--accent"],
  "--ag-font-family": cssVars["--font-sans"],
  "--ag-font-size": cssVars["--font-size-default"],
  "--ag-row-height": cssVars["--row-height-dense"],
  "--ag-header-height": cssVars["--row-height-dense"],
  "--ag-borders": "none",
  "--ag-header-column-resize-handle-color": cssVars["--border"],
  "--ag-wrapper-border-radius": cssVars["--radius-pane"],
};

/** Cell-flash colors for live-update pulses (§3.4 grid cell flash), by P&L/side direction. */
export const agGridFlash = {
  gain: cssVars["--gain"],
  loss: cssVars["--loss"],
  durationMs: 400,
} as const;
