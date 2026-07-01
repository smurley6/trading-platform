// Qualify a bare ticker for TradingView widgets; pass through anything already exchange-
// qualified (e.g. "NASDAQ:AAPL", "FOREXCOM:SPXUSD"). Shared by the Markets widgets and screen.
//
// TradingView needs the RIGHT exchange or it renders "Invalid symbol". We can't enumerate every
// ticker the political-signal cycle may propose (gov/defense contractors are mostly NYSE), so:
//   1. known names get an explicit, correct exchange prefix (below), and
//   2. anything unknown is passed BARE — TradingView resolves a bare US ticker to its primary
//      listing on its own, which is far safer than guessing the wrong exchange and hard-failing.
// Previously the fallback forced "NASDAQ:" on everything unlisted, so NYSE names like ACN/BAH/CACI
// rendered as invalid symbols.

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
  // Government IT / services contractors (frequent proposal symbols — all NYSE)
  "ACN",
  "BAH",
  "CACI",
  "SAIC",
  "KBR",
  "JCI",
  "EME",
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

// Known NASDAQ names (the default tech watchlist + common mega-caps) get an explicit prefix so
// those widgets that prefer a qualified symbol stay rock-solid; everything else falls through to
// the bare-ticker resolver above.
const NASDAQ_LISTED = new Set([
  "AAPL",
  "MSFT",
  "NVDA",
  "TSLA",
  "AMZN",
  "GOOGL",
  "GOOG",
  "META",
  "AVGO",
  "AMD",
  "INTC",
  "CSCO",
  "ADBE",
  "NFLX",
  "PEP",
  "COST",
  "QCOM",
  "TXN",
  "AMAT",
  "MU",
  "PLTR",
  "SPCX", // SpaceX — IPO'd 2026-06-12 on NASDAQ
]);

export function tvSymbol(symbol: string): string {
  if (symbol.includes(":")) return symbol;
  const ticker = symbol.toUpperCase();
  if (NYSE_LISTED.has(ticker)) return `NYSE:${ticker}`;
  if (NASDAQ_LISTED.has(ticker)) return `NASDAQ:${ticker}`;
  return ticker; // unknown ticker: let TradingView resolve it to its primary listing
}
