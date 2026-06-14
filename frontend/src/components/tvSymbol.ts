// Qualify a bare ticker for TradingView widgets; pass through anything already exchange-
// qualified (e.g. "NASDAQ:AAPL", "FOREXCOM:SPXUSD"). Shared by the Markets widgets and screen.
export function tvSymbol(symbol: string): string {
  return symbol.includes(":") ? symbol : `NASDAQ:${symbol}`;
}
