// Qualify a bare ticker for TradingView widgets; pass through anything already exchange-
// qualified (e.g. "NASDAQ:AAPL", "FOREXCOM:SPXUSD"). Shared by the Markets widgets and screen.
//
// TradingView widgets need the right exchange prefix or they render "Invalid symbol".
// Most mega-cap tech (the default watchlist) is NASDAQ, but the political-signal strategy
// leans on NYSE-listed defense/industrial/financial names, so we keep an explicit NYSE set
// and default everything else to NASDAQ.
const NYSE_LISTED = new Set([
  // Defense / aerospace / government contractors (the political-signal universe)
  "LMT",
  "RTX",
  "NOC",
  "GD",
  "BA",
  "LHX",
  "HII",
  "LDOS",
  "GE",
  "HON",
  "CAT",
  "DE",
  // Energy
  "XOM",
  "CVX",
  "COP",
  "SLB",
  // Financials
  "JPM",
  "BAC",
  "WFC",
  "GS",
  "MS",
  "C",
  "BRK.B",
  "V",
  "MA",
  // Healthcare / pharma
  "UNH",
  "JNJ",
  "PFE",
  "MRK",
  "LLY",
  "ABBV",
  "TMO",
  // Consumer / industrial / other large NYSE names
  "WMT",
  "KO",
  "PG",
  "DIS",
  "MCD",
  "NKE",
  "HD",
  "T",
  "VZ",
  "ORCL",
  "CRM",
  "IBM",
  "F",
  "GM",
]);

export function tvSymbol(symbol: string): string {
  if (symbol.includes(":")) return symbol;
  const ticker = symbol.toUpperCase();
  return NYSE_LISTED.has(ticker) ? `NYSE:${ticker}` : `NASDAQ:${ticker}`;
}
