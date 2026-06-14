// Compact single-symbol sparkline (TradingView embed-widget-mini-symbol-overview), used for
// the mini-chart grid on the Markets screen. Live, no key. Dark + transparent.

import { TradingViewWidget } from "./TradingViewWidget";
import { tvSymbol } from "./tvSymbol";

interface MiniSymbolChartProps {
  symbol: string;
  height?: number;
}

export function MiniSymbolChart({ symbol, height = 180 }: MiniSymbolChartProps) {
  const config = {
    symbol: tvSymbol(symbol),
    width: "100%",
    height: "100%",
    locale: "en",
    dateRange: "3M",
    colorTheme: "dark",
    isTransparent: true,
    autosize: true,
    largeChartUrl: "",
  };
  return (
    <TradingViewWidget
      script="embed-widget-mini-symbol-overview.js"
      config={config}
      height={height}
    />
  );
}
