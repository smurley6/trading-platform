// Overview's equity curve (§5.1: "Equity curve (Lightweight Charts, pool-level, 1M default)") —
// the first consumer of lib/theme-adapters/lightweightChartsTheme.ts (Phase A left it unconsumed;
// the legacy components/EquityCurve.tsx keeps reading the old `color.*` tokens for existing
// screens per that phase's container-only scope).

import { type IChartApi, type UTCTimestamp, createChart } from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { EquityPoint } from "../../lib/projections";
import { equitySeriesOptions, lightweightChartsOptions } from "../../lib/theme-adapters/lightweightChartsTheme";
import { ShellEmptyState } from "../../shell/states";

// "1M default" (§5.1): clip the rendered series to the most recent ~30 days of points so the
// chart opens on a one-month window without needing a separate range-picker in this phase.
const ONE_MONTH_SECONDS = 30 * 24 * 60 * 60;

function clipToOneMonth(points: EquityPoint[]): EquityPoint[] {
  if (points.length === 0) return points;
  const latest = points[points.length - 1]!.time;
  const cutoff = latest - ONE_MONTH_SECONDS;
  const clipped = points.filter((p) => p.time >= cutoff);
  return clipped.length > 0 ? clipped : points;
}

export function OverviewEquityCurve({ data }: { data: EquityPoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || data.length === 0) return;

    const chart = createChart(container, {
      ...lightweightChartsOptions(),
      height: container.clientHeight,
      width: container.clientWidth,
    });
    const series = chart.addAreaSeries(equitySeriesOptions());
    const points = clipToOneMonth(data);
    series.setData(points.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })));
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const observer = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data]);

  if (data.length === 0) {
    return <ShellEmptyState message="No equity history yet — submit signals to generate activity." />;
  }

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
