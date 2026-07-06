// Equity curve via TradingView Lightweight Charts (FR-UI-2). Attribution is required and
// rendered below the chart.

import {
  type IChartApi,
  type UTCTimestamp,
  ColorType,
  LineStyle,
  createChart,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { EquityPoint } from "../lib/projections";
import { color, font } from "../lib/tokens";

interface EquityCurveProps {
  data: EquityPoint[];
  height?: number;
  /**
   * Optional benchmark series, drawn as a muted line over the equity area (used by the backtest
   * view to compare strategy equity against a benchmark). Omit it and the chart is unchanged.
   */
  benchmark?: EquityPoint[];
}

export function EquityCurve({ data, height = 280, benchmark }: EquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: color.muted,
        fontFamily: font.mono,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.05)" },
      },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: color.hairline },
    });
    const series = chart.addAreaSeries({
      lineColor: color.accent,
      topColor: "rgba(61,139,255,0.32)",
      bottomColor: "rgba(61,139,255,0.02)",
      lineWidth: 2,
    });
    series.setData(data.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })));
    if (benchmark && benchmark.length > 0) {
      const benchSeries = chart.addLineSeries({
        color: color.muted,
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
      });
      benchSeries.setData(benchmark.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })));
    }
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const observer = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth });
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height, benchmark]);

  return (
    <div>
      <div ref={containerRef} style={{ width: "100%" }} />
      <div style={{ fontSize: 11, color: color.faint, marginTop: 4 }}>
        Charts by{" "}
        <a href="https://www.tradingview.com" target="_blank" rel="noreferrer">
          TradingView
        </a>
      </div>
    </div>
  );
}
