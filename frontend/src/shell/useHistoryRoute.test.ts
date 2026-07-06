import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useScreenQueryParam } from "./useHistoryRoute";

describe("useScreenQueryParam", () => {
  const originalUrl = window.location.href;

  beforeEach(() => {
    window.history.replaceState(null, "", "/");
  });

  afterEach(() => {
    window.history.replaceState(null, "", originalUrl);
  });

  it("reads the initial value from the URL on mount (deep link survives reload)", () => {
    window.history.replaceState(null, "", "/?ticker=LMT");
    const { result } = renderHook(() => useScreenQueryParam("ticker", true));
    expect(result.current[0]).toBe("LMT");
  });

  it("starts null when the param is absent", () => {
    const { result } = renderHook(() => useScreenQueryParam("ticker", true));
    expect(result.current[0]).toBeNull();
  });

  it("pushes the param onto the URL when set in push mode", () => {
    const { result } = renderHook(() => useScreenQueryParam("ticker", true, "push"));
    act(() => {
      result.current[1]("NOC");
    });
    expect(result.current[0]).toBe("NOC");
    expect(window.location.search).toBe("?ticker=NOC");
  });

  it("removes the param when cleared with null", () => {
    window.history.replaceState(null, "", "/?ticker=LMT");
    const { result } = renderHook(() => useScreenQueryParam("ticker", true));
    act(() => {
      result.current[1](null);
    });
    expect(result.current[0]).toBeNull();
    expect(window.location.search).toBe("");
  });

  it("rewrites in place in replace mode (no history spam for j/k selection state)", () => {
    const historyLength = window.history.length;
    const { result } = renderHook(() => useScreenQueryParam("proposal", true, "replace"));
    act(() => {
      result.current[1]("LMT");
    });
    act(() => {
      result.current[1]("NOC");
    });
    expect(window.location.search).toBe("?proposal=NOC");
    expect(window.history.length).toBe(historyLength);
  });

  it("syncs from external popstate navigation while active", () => {
    const { result } = renderHook(() => useScreenQueryParam("ticker", true));
    act(() => {
      window.history.pushState(null, "", "/?ticker=RTX");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(result.current[0]).toBe("RTX");
  });

  it("ignores popstate while inactive (hidden keep-alive screens must not react)", () => {
    const { result } = renderHook(() => useScreenQueryParam("ticker", false));
    act(() => {
      window.history.pushState(null, "", "/?ticker=RTX");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(result.current[0]).toBeNull();
  });
});
