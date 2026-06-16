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
        textColor: "#a3a8b4",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.06)" },
        horzLines: { color: "rgba(255,255,255,0.06)" },
      },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
    });
    const series = chart.addAreaSeries({
      lineColor: "#4c9aff",
      topColor: "rgba(76,154,255,0.35)",
      bottomColor: "rgba(76,154,255,0.02)",
    });
    series.setData(data.map((p) => ({ time: p.time as UTCTimestamp, value: p.value })));
    if (benchmark && benchmark.length > 0) {
      const benchSeries = chart.addLineSeries({
        color: "#94a3b8",
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
      <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
        Charts by{" "}
        <a href="https://www.tradingview.com" target="_blank" rel="noreferrer">
          TradingView
        </a>
      </div>
    </div>
  );
}
