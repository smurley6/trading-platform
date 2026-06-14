// REST client. All calls go through the Vite proxy at /api -> backend (see vite.config.ts).

import type {
  BacktestResult,
  ConvictionItem,
  EvaluateResult,
  IngestResult,
  KillSwitchState,
  LiveReadiness,
  ReconcileReport,
  SignalFeedItem,
  SignalInput,
  Sleeve,
  SleeveModeResult,
  StoredEventDto,
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
  backtest: () => request<BacktestResult>("/backtest", { method: "POST" }),

  liveReadiness: (sleeveId: string) =>
    request<LiveReadiness>(`/sleeves/${encodeURIComponent(sleeveId)}/readiness`, {
      method: "POST",
    }),
  setSleeveMode: (sleeveId: string, mode: TradingMode, confirm = false) =>
    request<SleeveModeResult>(`/sleeves/${encodeURIComponent(sleeveId)}/mode`, {
      method: "POST",
      body: JSON.stringify({ mode, confirm }),
    }),
};

export { ApiError };
