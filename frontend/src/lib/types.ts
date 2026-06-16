// API DTOs — mirror the backend Pydantic schemas (trading/interface/schemas.py).
// Kept in one place so a backend contract change surfaces as a TypeScript error.

export type SignalSourceType = "statement" | "contract" | "disclosed_trade";
export type ConvictionAction = "trade" | "watch" | "ignore";
export type TradingMode = "paper" | "live";
export type SleeveStatus = "running" | "paused";

export interface Sleeve {
  id: string;
  strategy_id: string;
  allocated_capital: string;
  mode: TradingMode;
  status: SleeveStatus;
}

export interface SignalFeedItem {
  signal_id: string;
  occurred_at: string;
  source_type: SignalSourceType;
  ticker: string;
  politician_id: string | null;
  sentiment: number;
  politician_weight: number;
  source_reliability: number;
}

export interface ConvictionItem {
  ticker: string;
  score: number;
  action: ConvictionAction;
  contributing_signal_ids: string[];
}

export interface StoredEventDto {
  event_id: string;
  stream_id: string;
  seq: number;
  event_type: string;
  schema_version: number;
  occurred_at: string;
  recorded_at: string;
  correlation_id: string;
  payload: Record<string, unknown>;
}

export interface OrderDto {
  client_order_id: string;
  broker_order_id: string;
  symbol: string;
  side: string;
  quantity: string;
  status: string;
}

export interface EvaluateResult {
  correlation_id: string;
  orders: OrderDto[];
}

export interface IngestResult {
  ingestion_correlation_id: string;
  signals_ingested: number;
  failed_sources: string[];
  evaluation_correlation_id: string;
  orders: OrderDto[];
}

export interface Drift {
  kind: string;
  key: string;
  local: string;
  broker: string;
}

export interface ReconcileReport {
  correlation_id: string;
  sleeve_id: string | null;
  clean: boolean;
  drifts: Drift[];
  checked_positions: number;
  checked_orders: number;
}

export interface KillSwitchState {
  engaged: boolean;
}

export interface LiveCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface LiveReadiness {
  sleeve_id: string;
  ready: boolean;
  checks: LiveCheck[];
}

export interface SleeveModeResult {
  sleeve: Sleeve;
  readiness: LiveReadiness | null;
}

// Backtest result (POST /backtest). Numeric fields may arrive as JSON strings from the backend's
// Decimal serialization, so every consumer coerces with Number(...) (see the Markets watchlist).
//
// The curve points support BOTH shapes so the type survives a backend contract change:
//   - the legacy shape: { time: <unix seconds>, equity }
//   - the new shape:     { t: <ISO timestamp>, equity }
// `backtestCurve()` (lib/projections) normalizes either form into EquityPoint[] for EquityCurve.
export type Numeric = number | string;

export interface BacktestPoint {
  /** New shape: ISO-8601 timestamp. */
  t?: string;
  /** Legacy shape: unix seconds. */
  time?: number;
  equity: Numeric;
}

export interface BacktestTrade {
  symbol: string;
  side: string;
  price: Numeric;
  quantity: Numeric;
  filled_at: string;
}

export interface BacktestResult {
  starting_equity: Numeric;
  ending_equity: Numeric;
  total_return_pct: Numeric;
  max_drawdown_pct: Numeric;
  sharpe: Numeric | null;
  sortino: Numeric | null;
  annualized_return_pct: Numeric | null;
  volatility: Numeric | null;
  win_rate: Numeric | null;
  profit_factor: Numeric | null;
  avg_win: Numeric | null;
  avg_loss: Numeric | null;
  num_trades: number;
  decision_count: number;
  benchmark_total_return_pct: Numeric | null;
  equity_curve: BacktestPoint[];
  /** Optional benchmark equity series, plotted alongside the strategy curve. */
  benchmark_curve?: BacktestPoint[];
  trades: BacktestTrade[];
}

export interface OrderCancelResult {
  client_order_id: string;
  status: string;
}

export interface SignalInput {
  source_type: SignalSourceType;
  ticker: string;
  politician_id?: string | null;
  sentiment: number;
  politician_weight?: number;
  source_reliability?: number;
  raw_payload?: Record<string, unknown>;
}

// Live market-data feeds (Markets screen + Dashboard live P&L). Numeric fields arrive as
// strings from the backend (parse with Number(...)). All three feeds default to OFF when no
// data-provider credentials are configured, so every consumer must treat `enabled: false`
// (and any network error) as a graceful no-op.

export interface Quote {
  symbol: string;
  bid: string;
  ask: string;
  last: string;
  as_of: string;
}

export interface QuotesResult {
  enabled: boolean;
  as_of: string | null;
  quotes: Quote[];
}

export interface TrackedSymbols {
  symbols: string[];
}

export interface LivePosition {
  symbol: string;
  sleeve_id: string;
  quantity: string;
  avg_price: string;
  last: string;
  market_value: string;
  unrealized_pnl: string;
}

export interface LivePositionsResult {
  enabled: boolean;
  positions: LivePosition[];
}
