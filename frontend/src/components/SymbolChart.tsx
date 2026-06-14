// Market-data chart via the TradingView embeddable widget (FR-UI-1). Display-only — this
// is a chart, not a data source. The widget injects its own TradingView attribution.

import { useEffect, useRef } from "react";

interface SymbolChartProps {
  symbol: string;
  height?: number;
}

export function SymbolChart({ symbol, height = 360 }: SymbolChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.async = true;
    script.type = "text/javascript";
    script.innerHTML = JSON.stringify({
      symbol,
      autosize: true,
      interval: "D",
      timezone: "Etc/UTC",
      theme: "dark",
      style: "1",
      hide_top_toolbar: false,
      allow_symbol_change: false,
      backgroundColor: "rgba(0,0,0,0)",
    });
    container.appendChild(script);

    return () => {
      container.innerHTML = "";
    };
  }, [symbol]);

  return (
    <div className="tradingview-widget-container" ref={containerRef} style={{ height }}>
      <div className="tradingview-widget-container__widget" style={{ height }} />
    </div>
  );
}
