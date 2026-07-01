// Backtest results view. A "Run backtest" action POSTs to /backtest; the result is rendered as
// an equity curve (strategy + benchmark), a row of headline stat cards, and a trades table.
//
// The endpoint replays recorded signals through the strategy + risk + execution path. It returns
// 409 ("backtesting is not configured") when no historical market-data adapter is wired — that's
// surfaced as an explanatory empty state rather than an error. All numeric fields may arrive as
// JSON strings (Decimal serialization), so every value is coerced with Number(...).

import type { ColDef } from "ag-grid-community";
import { Button, Card, H2, H3, StackLayout, Text } from "@salt-ds/core";
import { useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { EquityCurve } from "../../components/EquityCurve";
import { ApiError, api } from "../../lib/api";
import { formatMoney, formatNumber, formatTime } from "../../lib/format";
import { InfoButton } from "../../components/InfoButton";
import { backtestCurve } from "../../lib/projections";
import type { BacktestResult, BacktestTrade } from "../../lib/types";

type Tone = "success" | "danger" | "info" | "muted";

function toneColor(tone: Tone): string {
  switch (tone) {
    case "success":
      return "#4ade80";
    case "danger":
      return "#f87171";
    case "info":
      return "#7cb3ff";
    case "muted":
      return "#94a3b8";
  }
}

function signTone(value: number): Tone {
  if (value > 0) return "success";
  if (value < 0) return "danger";
  return "muted";
}

// Format a possibly-null, possibly-string numeric as a percentage, "—" when absent/unparseable.
function pct(value: number | string | null, digits = 2): string {
  if (value === null) return "—";
  const n = Number(value);
  return Number.isNaN(n) ? "—" : `${formatNumber(n, digits)}%`;
}

// Format a possibly-null numeric ratio (Sharpe/Sortino/profit factor), "—" when absent.
function ratio(value: number | string | null, digits = 2): string {
  if (value === null) return "—";
  const n = Number(value);
  return Number.isNaN(n) ? "—" : formatNumber(n, digits);
}

// Format a possibly-null money value, "—" when absent.
function money(value: number | string | null): string {
  if (value === null) return "—";
  const n = Number(value);
  return Number.isNaN(n) ? "—" : formatMoney(n);
}

const tradeColumns: ColDef<BacktestTrade>[] = [
  { headerName: "Symbol", field: "symbol", maxWidth: 120 },
  { headerName: "Side", field: "side", maxWidth: 100 },
  {
    headerName: "Qty",
    field: "quantity",
    maxWidth: 120,
    valueFormatter: (p) => formatNumber(Number(p.value), 0),
  },
  {
    headerName: "Price",
    field: "price",
    valueFormatter: (p) => formatMoney(Number(p.value)),
  },
  {
    headerName: "Filled",
    field: "filled_at",
    valueFormatter: (p) => formatTime(p.value as string),
  },
];

interface BacktestProps {
  // Present for nav consistency with the other screens; the backtest is run on demand, not on
  // the live refresh tick, so the key is intentionally unused.
  refreshKey: number;
}

export function Backtest(_props: BacktestProps) {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // True once the backend reports backtesting is unavailable (HTTP 409). Distinct from a generic
  // error so we can show an explanatory empty state instead of a red error line.
  const [notConfigured, setNotConfigured] = useState(false);

  async function run(): Promise<void> {
    setBusy(true);
    setError(null);
    setNotConfigured(false);
    try {
      setResult(await api.backtest());
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setNotConfigured(true);
        setResult(null);
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setBusy(false);
    }
  }

  const strategyCurve = backtestCurve(result?.equity_curve);
  const benchmark = backtestCurve(result?.benchmark_curve);

  const totalReturn = result ? Number(result.total_return_pct) : 0;
  const maxDrawdown = result ? Number(result.max_drawdown_pct) : 0;
  const benchReturn = result?.benchmark_total_return_pct ?? null;

  return (
    <StackLayout gap={2}>
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div>
            <H2 style={{ margin: 0 }}>Backtest</H2>
            <Text color="secondary" styleAs="label">
              Replays recorded signals through the strategy and the same risk + execution path used
              live, then compares strategy equity against a benchmark.
            </Text>
          </div>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
            <Button disabled={busy} onClick={() => void run()}>
              {busy ? "Running…" : "Run backtest"}
            </Button>
            <InfoButton title="Run backtest" align="right">
              Replays your recorded political signals through the strategy, the risk checks, and the
              same execution engine used live — over historical prices — then charts strategy equity
              vs. a buy-and-hold benchmark and reports Sharpe, max drawdown, win rate, and every
              simulated trade. It's a SIMULATION: no real or paper orders are placed. Needs a
              historical market-data adapter on the backend (otherwise you'll see "not configured").
            </InfoButton>
          </div>
        </div>
        {error && (
          <Text color="error" style={{ marginTop: 8 }}>
            {error}
          </Text>
        )}
      </Card>

      {notConfigured && (
        <Card>
          <H3 style={{ marginTop: 0 }}>Backtesting is not configured</H3>
          <Text color="secondary">
            No historical market-data adapter is wired on the backend, so backtests can't be run
            yet. Once a data provider is configured, use “Run backtest” to replay recorded signals
            and see strategy vs. benchmark performance here.
          </Text>
        </Card>
      )}

      {!result && !notConfigured && !error && (
        <Card>
          <Text color="secondary">
            Run a backtest to see the equity curve, headline metrics, and the simulated trades.
          </Text>
        </Card>
      )}

      {result && (
        <>
          <Card>
            <div
              style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}
            >
              <H3 style={{ marginTop: 0 }}>Equity curve</H3>
              <div style={{ display: "flex", gap: 16, fontSize: 12, color: "#94a3b8" }}>
                <LegendDot color="#4c9aff" label="Strategy" />
                {benchmark.length > 0 && <LegendDot color="#94a3b8" label="Benchmark" />}
              </div>
            </div>
            {strategyCurve.length > 0 ? (
              <EquityCurve
                data={strategyCurve}
                benchmark={benchmark.length > 0 ? benchmark : undefined}
              />
            ) : (
              <Text color="secondary">No equity history was produced for this backtest.</Text>
            )}
          </Card>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
              gap: 12,
            }}
          >
            <StatCard
              label="Total return"
              value={pct(result.total_return_pct)}
              tone={signTone(totalReturn)}
            />
            <StatCard
              label="Benchmark return"
              value={pct(benchReturn)}
              tone={benchReturn === null ? "muted" : signTone(Number(benchReturn))}
            />
            <StatCard
              label="Max drawdown"
              value={pct(result.max_drawdown_pct)}
              tone={maxDrawdown > 0 ? "danger" : "muted"}
            />
            <StatCard label="Sharpe" value={ratio(result.sharpe)} />
            <StatCard label="Sortino" value={ratio(result.sortino)} />
            <StatCard label="Win rate" value={pct(result.win_rate, 0)} />
            <StatCard label="Profit factor" value={ratio(result.profit_factor)} />
            <StatCard
              label="Annualized return"
              value={pct(result.annualized_return_pct)}
              tone={
                result.annualized_return_pct === null
                  ? "muted"
                  : signTone(Number(result.annualized_return_pct))
              }
            />
            <StatCard label="Volatility" value={pct(result.volatility)} />
            <StatCard label="Avg win" value={money(result.avg_win)} tone="success" />
            <StatCard label="Avg loss" value={money(result.avg_loss)} tone="danger" />
            <StatCard label="Trades" value={formatNumber(result.num_trades, 0)} />
            <StatCard label="Decisions" value={formatNumber(result.decision_count, 0)} />
            <StatCard label="Start equity" value={money(result.starting_equity)} />
            <StatCard label="End equity" value={money(result.ending_equity)} />
          </div>

          <Card>
            <H3 style={{ marginTop: 0 }}>Trades ({result.trades.length})</H3>
            {result.trades.length > 0 ? (
              <DataGrid<BacktestTrade>
                rows={result.trades}
                columns={tradeColumns}
                getRowId={(r) => `${r.symbol}-${r.filled_at}-${r.side}-${r.quantity}`}
                height={Math.min(420, Math.max(160, 60 + result.trades.length * 44))}
              />
            ) : (
              <Text color="secondary">No trades were generated in this backtest.</Text>
            )}
          </Card>
        </>
      )}
    </StackLayout>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 2,
          background: color,
          display: "inline-block",
        }}
      />
      {label}
    </span>
  );
}

function StatCard({ label, value, tone }: { label: string; value: string; tone?: Tone }) {
  return (
    <Card style={{ minWidth: 0 }}>
      <Text color="secondary" styleAs="label">
        {label}
      </Text>
      <div
        style={{
          fontSize: 20,
          fontWeight: 600,
          marginTop: 4,
          color: tone ? toneColor(tone) : undefined,
        }}
      >
        {value}
      </div>
    </Card>
  );
}
