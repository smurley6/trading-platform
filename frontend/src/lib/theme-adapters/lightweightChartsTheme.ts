// Lightweight Charts theming adapter (docs/UI-DESIGN.md §3 / §10) — maps the §3 palette into
// Lightweight Charts' `createChart` options object, its native theming surface (no CSS custom
// properties; colors are passed directly at chart-creation time). This file is the single place
// that mapping lives — do not hand-roll chart color options elsewhere.
//
// Usage: spread `lightweightChartsOptions()` into `createChart(container, { ...here })`.
// components/EquityCurve.tsx is unchanged in Phase A (container/chrome only, still reads the
// legacy `color.*` tokens); this adapter is available for Phase B's Overview equity curve.

import type { DeepPartial, LineStyleOptions, TimeChartOptions } from "lightweight-charts";
import { ColorType, LineStyle } from "lightweight-charts";
import { colors, type as typeTokens, withAlpha } from "../tokens";

/** Base chart options (background, grid, axes, font) drawn from the §3 palette. */
export function lightweightChartsOptions(): DeepPartial<TimeChartOptions> {
  return {
    layout: {
      background: { type: ColorType.Solid, color: "transparent" },
      textColor: colors.textSecondary,
      fontFamily: typeTokens.fontMono,
      fontSize: typeTokens.scale.body,
    },
    grid: {
      vertLines: { color: colors.border },
      horzLines: { color: colors.border },
    },
    timeScale: { timeVisible: true, secondsVisible: false, borderColor: colors.border },
    rightPriceScale: { borderColor: colors.border },
    crosshair: {
      vertLine: { color: colors.accent, labelBackgroundColor: colors.bgRaised },
      horzLine: { color: colors.accent, labelBackgroundColor: colors.bgRaised },
    },
  };
}

/** Area-series options for an equity curve, colored by the reserved accent (not gain/loss —
 *  pool-level equity is neutral, per §3.1 gain/loss being reserved for P&L direction/order side). */
export function equitySeriesOptions(): {
  lineColor: string;
  topColor: string;
  bottomColor: string;
  lineWidth: 2;
} {
  return {
    lineColor: colors.accent,
    topColor: withAlpha(colors.accent, 0.32), // --accent at reduced alpha, area fill only
    bottomColor: withAlpha(colors.accent, 0.02),
    lineWidth: 2,
  };
}

/** Benchmark/comparison line series — muted, dashed, never a reserved semantic color. */
export function benchmarkSeriesOptions(): DeepPartial<LineStyleOptions> & {
  color: string;
  lineWidth: 2;
} {
  return {
    color: colors.textMuted,
    lineWidth: 2,
    lineStyle: LineStyle.Dashed,
  };
}

/** Directional colors for candlestick/bar series or line segments, using the reserved semantics. */
export const directionalSeriesColors = {
  up: colors.gain,
  down: colors.loss,
} as const;
