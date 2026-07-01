import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";

// A typed stand-in for the global fetch mock so we don't reach for `any`.
type FetchMock = ReturnType<typeof vi.fn<typeof fetch>>;

function okResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(typeof body === "string" ? body : JSON.stringify(body)),
  } as unknown as Response;
}

function errorResponse(status: number, message: string): Response {
  return {
    ok: false,
    status,
    json: () => Promise.reject(new Error("should not be called")),
    text: () => Promise.resolve(message),
  } as unknown as Response;
}

let fetchMock: FetchMock;

beforeEach(() => {
  fetchMock = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("api request building", () => {
  it("GETs the right /api path and parses the JSON body", async () => {
    const sleeves = [{ id: "s1" }];
    fetchMock.mockResolvedValue(okResponse(sleeves));

    const result = await api.listSleeves();

    expect(result).toEqual(sleeves);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/sleeves");
    expect(init?.headers).toMatchObject({ "content-type": "application/json" });
  });

  it("appends query parameters for the signal feed", async () => {
    fetchMock.mockResolvedValue(okResponse([]));
    await api.signalFeed(50);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/signals/feed?limit=50");
  });

  it("POSTs a JSON body for the kill switch", async () => {
    fetchMock.mockResolvedValue(okResponse({ engaged: true }));

    const result = await api.setKillSwitch(true);

    expect(result).toEqual({ engaged: true });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/kill-switch");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ engaged: true }));
  });

  it("POSTs to the cancel route with the encoded client order id", async () => {
    fetchMock.mockResolvedValue(okResponse({ client_order_id: "o/1", status: "canceled" }));
    const result = await api.cancelOrder("o/1");
    expect(result).toEqual({ client_order_id: "o/1", status: "canceled" });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/orders/o%2F1/cancel");
    expect(init?.method).toBe("POST");
  });

  it("POSTs to /backtest", async () => {
    fetchMock.mockResolvedValue(okResponse({ trades: [] }));
    await api.backtest();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/backtest");
    expect(init?.method).toBe("POST");
  });

  it("encodes path segments for sleeve-scoped routes", async () => {
    fetchMock.mockResolvedValue(okResponse({ sleeve_id: "a/b", ready: true, checks: [] }));
    await api.liveReadiness("a/b");
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/sleeves/a%2Fb/readiness");
    expect(fetchMock.mock.calls[0]![1]?.method).toBe("POST");
  });

  it("includes the sleeve_id query when submitting signals with a sleeve", async () => {
    fetchMock.mockResolvedValue(okResponse({ correlation_id: "c", orders: [] }));
    await api.submitSignals(
      [{ source_type: "statement", ticker: "AAPL", sentiment: 0.5 }],
      "sleeve-x",
    );
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/signals?sleeve_id=sleeve-x");
    expect(init?.method).toBe("POST");
  });
});

describe("api error handling", () => {
  it("throws an ApiError carrying the status and response body on non-200", async () => {
    fetchMock.mockResolvedValue(errorResponse(500, "boom"));

    await expect(api.listSleeves()).rejects.toBeInstanceOf(ApiError);
    await expect(api.listSleeves()).rejects.toMatchObject({ status: 500 });
    await expect(api.listSleeves()).rejects.toThrow(/boom/);
  });

  it("throws a 409 ApiError for an unconfigured backtest so the UI can show an empty state", async () => {
    fetchMock.mockResolvedValue(errorResponse(409, "backtesting is not configured"));
    await expect(api.backtest()).rejects.toMatchObject({ status: 409 });
  });
});

describe("graceful market-data fallbacks", () => {
  it("returns a disabled shape for quotes without hitting the network when there are no symbols", async () => {
    const result = await api.quotes([]);
    expect(result).toEqual({ enabled: false, as_of: null, quotes: [] });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("requests the joined symbol list and parses an enabled quotes response", async () => {
    fetchMock.mockResolvedValue(okResponse({ enabled: true, as_of: "t", quotes: [] }));
    const result = await api.quotes(["AAPL", "MSFT"]);
    expect(result.enabled).toBe(true);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/quotes?symbols=AAPL%2CMSFT");
  });

  it("falls back to a disabled shape when the quotes request errors", async () => {
    fetchMock.mockResolvedValue(errorResponse(404, "no feed"));
    const result = await api.quotes(["AAPL"]);
    expect(result).toEqual({ enabled: false, as_of: null, quotes: [] });
  });

  it("falls back to disabled live positions when the request errors", async () => {
    fetchMock.mockResolvedValue(errorResponse(503, "down"));
    const result = await api.livePositions();
    expect(result).toEqual({ enabled: false, positions: [] });
  });
});
