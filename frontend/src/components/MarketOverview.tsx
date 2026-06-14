// Market overview widget (TradingView embed-widget-market-overview): tabbed mini sparklines
// for the tracked watchlist plus a couple of broad-market reference groups. Live, no key.

import { TradingViewWidget } from "./TradingViewWidget";
import { tvSymbol } from "./tvSymbol";

interface MarketOverviewProps {
  symbols: string[];
  height?: number;
}

export function MarketOverview({ symbols, height = 460 }: MarketOverviewProps) {
  const config = {
    colorTheme: "dark",
    dateRange: "12M",
    showChart: true,
    locale: "en",
    isTransparent: true,
    showSymbolLogo: true,
    width: "100%",
    height: "100%",
    tabs: [
      {
        title: "Watchlist",
        symbols: symbols.map((s) => ({ s: tvSymbol(s), d: s })),
      },
      {
        title: "Indices",
        symbols: [
          { s: "FOREXCOM:SPXUSD", d: "S&P 500" },
          { s: "FOREXCOM:NSXUSD", d: "Nasdaq 100" },
          { s: "FOREXCOM:DJI", d: "Dow 30" },
          { s: "INDEX:VIX", d: "Volatility" },
        ],
      },
    ],
  };
  return (
    <TradingViewWidget script="embed-widget-market-overview.js" config={config} height={height} />
  );
}
