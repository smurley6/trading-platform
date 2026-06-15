import { describe, expect, it } from "vitest";
import { tvSymbol } from "./tvSymbol";

describe("tvSymbol", () => {
  it("prefixes known NYSE-listed tickers with NYSE:", () => {
    expect(tvSymbol("LMT")).toBe("NYSE:LMT");
    expect(tvSymbol("JPM")).toBe("NYSE:JPM");
  });

  it("defaults other tickers to NASDAQ:", () => {
    expect(tvSymbol("AAPL")).toBe("NASDAQ:AAPL");
    expect(tvSymbol("MSFT")).toBe("NASDAQ:MSFT");
  });

  it("uppercases bare tickers before classifying", () => {
    expect(tvSymbol("lmt")).toBe("NYSE:LMT");
    expect(tvSymbol("aapl")).toBe("NASDAQ:AAPL");
  });

  it("passes through already exchange-qualified symbols unchanged", () => {
    expect(tvSymbol("NASDAQ:AAPL")).toBe("NASDAQ:AAPL");
    expect(tvSymbol("FOREXCOM:SPXUSD")).toBe("FOREXCOM:SPXUSD");
  });
});
