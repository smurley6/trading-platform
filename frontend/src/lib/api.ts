// REST client. All calls go through the Vite proxy at /api -> backend (see vite.config.ts).

import type {
  BacktestResult,
  CapitalResult,
  ConvictionItem,
  CycleMode,
  CycleModeResult,
  EvaluateResult,
  IngestResult,
  KillSwitchState,
  LivePositionsResult,
  LiveReadiness,
  OrderCancelResult,
  ProposalsExecuteResult,
  ProposalsResult,
  QuotesResult,
  ReconcileReport,
  SignalFeedItem,
  SignalInput,
  Sleeve,
  SleeveModeResult,
  StoredEventDto,
  TrackedSymbols,
  TradingMode,
} from "./types";

// In prod the SPA is served from a different origin than the API (a static host), so it calls the
// backend directly via VITE_API_URL (baked in at build time). In dev VITE_API_URL is unset, so it
// falls back to same-origin "/api", which the Vite dev proxy forwards to the backend.
const BASE = import.meta.env.VITE_API_URL ?? "/api";

class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, `${init?.method ?? "GET"} ${path} failed: ${body}`);
  }
  return (await response.json()) as T;
}

export const api = {
  listSleeves: () => request<Sleeve[]>("/sleeves"),
  conviction: () => request<ConvictionItem[]>("/conviction"),
  signalFeed: (limit = 200) => request<SignalFeedItem[]>(`/signals/feed?limit=${limit}`),
  events: (limit = 500) => request<StoredEventDto[]>(`/events?limit=${limit}`),
  eventsByStream: (streamId: string) =>
    request<StoredEventDto[]>(`/events?stream_id=${encodeURIComponent(streamId)}`),

  killSwitch: () => request<KillSwitchState>("/kill-switch"),
  setKillSwitch: (engaged: boolean) =>
    request<KillSwitchState>("/kill-switch", {
      method: "POST",
      body: JSON.stringify({ engaged }),
    }),

  submitSignals: (signals: SignalInput[], sleeveId?: string) =>
    request<EvaluateResult>(`/signals${sleeveId ? `?sleeve_id=${sleeveId}` : ""}`, {
      method: "POST",
      body: JSON.stringify(signals),
    }),

  ingest: () => request<IngestResult>("/ingest", { method: "POST" }),
  reconcile: () => request<ReconcileReport[]>("/reconcile", { method: "POST" }),

  // Daily-cycle proposals: the actionable trades the cycle found, the watch-tier candidates it
  // expects to act on next, and the current cycle mode (auto/manual).
  proposals: () => request<ProposalsResult>("/proposals"),
  // Approve & place proposals. Omit `symbols` (or pass []) to execute ALL current proposals.
  // The live event stream then drives the cross-screen refresh, same as other writes.
  executeProposals: (symbols: string[] = []) =>
    request<ProposalsExecuteResult>("/proposals/execute", {
      method: "POST",
      body: JSON.stringify({ symbols }),
    }),

  // Live arm/disarm of the scheduled cycle's trade mode. GET reads the current mode; POST flips
  // it and the running cycle honours it on its next tick (no redeploy). NOT persisted — reverts
  // to the CYCLE_TRADE_MODE env default if the backend restarts.
  cycleMode: () => request<CycleModeResult>("/cycle/mode"),
  setCycleMode: (mode: CycleMode) =>
    request<CycleModeResult>("/cycle/mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
  // POST /backtest replays recorded signals through the strategy + risk + execution path.
  // Returns 409 ("backtesting is not configured") when no historical market-data adapter is
  // wired; callers surface that as an explanatory empty state (see Backtest.tsx).
  backtest: () => request<BacktestResult>("/backtest", { method: "POST" }),

  // Cancel a still-open order by its client order id. Backend records it as an order event,
  // so the live event stream drives the blotter refresh.
  cancelOrder: (clientOrderId: string) =>
    request<OrderCancelResult>(`/orders/${encodeURIComponent(clientOrderId)}/cancel`, {
      method: "POST",
    }),

  liveReadiness: (sleeveId: string) =>
    request<LiveReadiness>(`/sleeves/${encodeURIComponent(sleeveId)}/readiness`, {
      method: "POST",
    }),
  setSleeveMode: (sleeveId: string, mode: TradingMode, confirm = false) =>
    request<SleeveModeResult>(`/sleeves/${encodeURIComponent(sleeveId)}/mode`, {
      method: "POST",
      body: JSON.stringify({ mode, confirm }),
    }),

  // --- Live market-data feeds (Markets screen + Dashboard) -------------------------------
  // These feeds default to OFF (no credentials) and may be absent on older backends, so each
  // resolves to a disabled-but-valid shape on any non-200 / network error. Callers can render
  // unconditionally without try/catch and simply branch on `.enabled`.

  quotes: (symbols: string[]): Promise<QuotesResult> => {
    if (symbols.length === 0) return Promise.resolve({ enabled: false, as_of: null, quotes: [] });
    return request<QuotesResult>(`/quotes?symbols=${encodeURIComponent(symbols.join(","))}`).catch(
      () => ({ enabled: false, as_of: null, quotes: [] }),
    );
  },

  trackedSymbols: (): Promise<TrackedSymbols> =>
    request<TrackedSymbols>("/quotes/tracked").catch(() => ({ symbols: [] })),

  livePositions: (): Promise<LivePositionsResult> =>
    request<LivePositionsResult>("/positions/live").catch(() => ({
      enabled: false,
      positions: [],
    })),

  // The single shared capital pool both algos draw from (live Alpaca buying power/equity + each
  // algo's slice). Resolves to a disabled shape on any non-200/network error so the Dashboard
  // can render unconditionally and simply branch on `.enabled`.
  capital: (): Promise<CapitalResult> =>
    request<CapitalResult>("/capital").catch(() => ({
      enabled: false,
      cash: "0",
      equity: "0",
      buying_power: "0",
      allocations: [],
    })),
};

export { ApiError };
