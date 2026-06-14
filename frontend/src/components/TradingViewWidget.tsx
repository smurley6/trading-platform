// Generic loader for TradingView "external-embedding" widgets (ticker tape, market overview,
// symbol overview, mini chart, …). Each widget is a <script> whose JSON config is its innerHTML;
// TradingView replaces the surrounding container with a live iframe. This wraps that injection
// pattern (mirrors components/SymbolChart.tsx) so every Markets widget cleans up its script on
// unmount and re-injects when its config changes. The widgets are genuinely live and need no
// API keys.

import { useEffect, useRef } from "react";

interface TradingViewWidgetProps {
  /** The widget script filename, e.g. "embed-widget-ticker-tape.js". */
  script: string;
  /** The widget's JSON configuration (serialized as the script body). */
  config: Record<string, unknown>;
  height?: number | string;
}

const SCRIPT_BASE = "https://s3.tradingview.com/external-embedding/";

export function TradingViewWidget({ script, config, height = 400 }: TradingViewWidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  // Re-inject only when the serialized config actually changes (object identity churns on
  // every render otherwise).
  const configKey = JSON.stringify(config);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";

    const widget = document.createElement("div");
    widget.className = "tradingview-widget-container__widget";
    container.appendChild(widget);

    const el = document.createElement("script");
    el.src = `${SCRIPT_BASE}${script}`;
    el.async = true;
    el.type = "text/javascript";
    el.innerHTML = configKey;
    container.appendChild(el);

    return () => {
      container.innerHTML = "";
    };
  }, [script, configKey]);

  return (
    <div
      className="tradingview-widget-container"
      ref={containerRef}
      style={{ height, width: "100%" }}
    />
  );
}
