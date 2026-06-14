// Scrolling live ticker tape across the top of the Markets screen (TradingView
// embed-widget-ticker-tape). Live, no API key. Dark + transparent to match the chrome.

import { TradingViewWidget } from "./TradingViewWidget";
import { tvSymbol } from "./tvSymbol";

interface TickerTapeProps {
  symbols: string[];
}

export function TickerTape({ symbols }: TickerTapeProps) {
  const config = {
    symbols: symbols.map((s) => ({ proName: tvSymbol(s), title: s })),
    showSymbolLogo: true,
    isTransparent: true,
    displayMode: "adaptive",
    colorTheme: "dark",
    locale: "en",
  };
  return <TradingViewWidget script="embed-widget-ticker-tape.js" config={config} height={78} />;
}
