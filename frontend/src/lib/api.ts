// REST client. All calls go through the Vite proxy at /api -> backend (see vite.config.ts).

import type {
  BacktestResult,
  ConvictionItem,
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

const BASE = "/api";

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
};

export { ApiError };
