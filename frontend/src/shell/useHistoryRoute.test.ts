import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useTickerQueryParam } from "./useHistoryRoute";

describe("useTickerQueryParam", () => {
  const originalUrl = window.location.href;

  beforeEach(() => {
    window.history.replaceState(null, "", "/");
  });

  afterEach(() => {
    window.history.replaceState(null, "", originalUrl);
  });

  it("reads the initial ticker from ?ticker= on mount (deep link survives reload)", () => {
    window.history.replaceState(null, "", "/?ticker=LMT");
    const { result } = renderHook(() => useTickerQueryParam());
    expect(result.current[0]).toBe("LMT");
  });

  it("starts null when there is no ticker param", () => {
    const { result } = renderHook(() => useTickerQueryParam());
    expect(result.current[0]).toBeNull();
  });

  it("pushes ?ticker=SYMBOL onto the URL when opened", () => {
    const { result } = renderHook(() => useTickerQueryParam());
    act(() => {
      result.current[1]("NOC");
    });
    expect(result.current[0]).toBe("NOC");
    expect(window.location.search).toBe("?ticker=NOC");
  });

  it("removes the ticker param when closed with null", () => {
    window.history.replaceState(null, "", "/?ticker=LMT");
    const { result } = renderHook(() => useTickerQueryParam());
    act(() => {
      result.current[1](null);
    });
    expect(result.current[0]).toBeNull();
    expect(window.location.search).toBe("");
  });

  it("syncs from external popstate navigation", () => {
    const { result } = renderHook(() => useTickerQueryParam());
    act(() => {
      window.history.pushState(null, "", "/?ticker=RTX");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(result.current[0]).toBe("RTX");
  });
});
