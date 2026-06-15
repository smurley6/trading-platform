import { describe, expect, it } from "vitest";
import { formatMoney, formatNumber, formatTime } from "./format";

describe("formatMoney", () => {
  it("formats a positive number as USD currency", () => {
    expect(formatMoney(1234.5)).toBe("$1,234.50");
  });

  it("formats zero", () => {
    expect(formatMoney(0)).toBe("$0.00");
  });

  it("formats a negative number with a leading minus sign", () => {
    expect(formatMoney(-42.25)).toBe("-$42.25");
  });

  it("rounds to at most two fraction digits", () => {
    expect(formatMoney(2.119)).toBe("$2.12");
    expect(formatMoney(2.114)).toBe("$2.11");
  });

  it("parses numeric strings (backend sends money as strings)", () => {
    expect(formatMoney("9999.99")).toBe("$9,999.99");
  });

  it("returns an em dash for non-numeric input", () => {
    expect(formatMoney("not-a-number")).toBe("—");
    expect(formatMoney(Number.NaN)).toBe("—");
  });
});

describe("formatNumber", () => {
  it("defaults to two fraction digits", () => {
    expect(formatNumber(3.14159)).toBe("3.14");
  });

  it("honours an explicit digit count", () => {
    expect(formatNumber(3.14159, 4)).toBe("3.1416");
    expect(formatNumber(3.14159, 0)).toBe("3");
  });

  it("adds grouping separators for large numbers", () => {
    expect(formatNumber(1234567, 0)).toBe("1,234,567");
  });

  it("formats negatives and zero", () => {
    expect(formatNumber(-1.5)).toBe("-1.5");
    expect(formatNumber(0)).toBe("0");
  });

  it("parses numeric strings", () => {
    expect(formatNumber("2.5")).toBe("2.5");
  });

  it("returns an em dash for non-numeric input", () => {
    expect(formatNumber("oops")).toBe("—");
  });
});

describe("formatTime", () => {
  it("renders a valid ISO timestamp via toLocaleString (non-empty, not the raw input)", () => {
    const iso = "2026-01-02T03:04:05.000Z";
    const out = formatTime(iso);
    expect(out).not.toBe("");
    // A locale string is produced, so it should not equal the raw ISO string.
    expect(out).not.toBe(iso);
  });

  it("returns the original string for an unparseable timestamp", () => {
    expect(formatTime("definitely-not-a-date")).toBe("definitely-not-a-date");
  });
});
