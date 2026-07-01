// Backtest results view. A "Run backtest" action POSTs to /backtest; the result is rendered as
// an equity curve (strategy + benchmark), a row of headline stat cards, and a trades table.
//
// The endpoint replays recorded signals through the strategy + risk + execution path. It returns
// 409 ("backtesting is not configured") when no historical market-data adapter is wired — that's
// surfaced as an explanatory empty state rather than an error. All numeric fields may arrive as
// JSON strings (Decimal serialization), so every value is coerced with Number(...).

import type { ColDef } from "ag-grid-community";
import { Button, StackLayout } from "@salt-ds/core";
import { useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { EquityCurve } from "../../components/EquityCurve";
import {
  Badge,
  EmptyState,
  Figure,
  LoadingState,
  MetricTile,
  Panel,
  SectionHeader,
} from "../../components/ui";
import { ApiError, api } from "../../lib/api";
import { formatMoney, formatNumber, formatTime } from "../../lib/format";
import { InfoButton } from "../../components/InfoButton";
import { backtestCurve } from "../../lib/projections";
import { color, radius, space, text, transitions } from "../../lib/tokens";
import type { BacktestResult, BacktestTrade } from "../../lib/types";

type Tone = "neutral" | "ok" | "danger" | "accent" | "warn";

function signTone(value: number): Tone {
  if (value > 0) return "ok";
  if (value < 0) return "danger";
  return "neutral";
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

// Color a trade's side chip: buys read as accent, sells as danger.
function sideTone(side: string): "accent" | "danger" | "muted" {
  const s = side.toLowerCase();
  if (s === "buy" || s === "long") return "accent";
  if (s === "sell" || s === "short") return "danger";
  return "muted";
}

const tradeColumns: ColDef<BacktestTrade>[] = [
  { headerName: "Symbol", field: "symbol", maxWidth: 120 },
  {
    headerName: "Side",
    field: "side",
    maxWidth: 110,
    cellRenderer: (p: { value: string | null | undefined }) => {
      const side = p.value ?? "";
      return <Badge tone={sideTone(side)}>{side.toUpperCase()}</Badge>;
    },
  },
  {
    headerName: "Qty",
    field: "quantity",
    maxWidth: 120,
    type: "rightAlignedColumn",
    valueFormatter: (p) => formatNumber(Number(p.value), 0),
  },
  {
    headerName: "Price",
    field: "price",
    type: "rightAlignedColumn",
    valueFormatter: (p) => formatMoney(Number(p.value)),
  },
  {
    headerName: "Filled",
    field: "filled_at",
    type: "rightAlignedColumn",
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
      <Panel>
        <SectionHeader
          eyebrow="Simulation"
          title="Backtest"
          caption="Replays recorded signals through the strategy and the same risk + execution path used live, then compares strategy equity against a benchmark."
          actions={
            <>
              <Button disabled={busy} onClick={() => void run()}>
                {busy ? "Running…" : "Run backtest"}
              </Button>
              <InfoButton title="Run backtest" align="right">
                Replays your recorded political signals through the strategy, the risk checks, and
                the same execution engine used live — over historical prices — then charts strategy
                equity vs. a buy-and-hold benchmark and reports Sharpe, max drawdown, win rate, and
                every simulated trade. It's a SIMULATION: no real or paper orders are placed. Needs
                a historical market-data adapter on the backend (otherwise you'll see "not
                configured").
              </InfoButton>
            </>
          }
        />
        {error && (
          <div
            style={{
              ...text.body,
              color: color.dangerText,
              background: color.dangerSoft,
              border: `1px solid ${color.danger}`,
              borderRadius: radius.md,
              padding: `${space.sm}px ${space.md}px`,
            }}
          >
            {error}
          </div>
        )}
      </Panel>

      {busy && !result && (
        <Panel>
          <LoadingState label="Running backtest — replaying signals over historical prices…" />
        </Panel>
      )}

      {notConfigured && (
        <Panel>
          <EmptyState
            icon="⚙"
            title="Backtesting is not configured"
            message="No historical market-data adapter is wired on the backend, so backtests can't be run yet. Once a data provider is configured, use “Run backtest” to replay recorded signals and see strategy vs. benchmark performance here."
          />
        </Panel>
      )}

      {!result && !notConfigured && !error && !busy && (
        <Panel>
          <EmptyState
            icon="▲"
            title="No backtest yet"
            message="Run a backtest to see the equity curve, headline metrics, and the simulated trades."
            action={
              <Button disabled={busy} onClick={() => void run()}>
                Run your first backtest
              </Button>
            }
          />
        </Panel>
      )}

      {result && (
        <>
          <Panel>
            <SectionHeader
              eyebrow="Performance"
              title="Equity curve"
              caption="Strategy equity vs. buy-and-hold benchmark over the replay window."
              actions={
                <div style={{ display: "flex", gap: space.md }}>
                  <LegendDot color={color.accent} label="Strategy" />
                  {benchmark.length > 0 && <LegendDot color={color.muted} label="Benchmark" />}
                </div>
              }
            />
            {strategyCurve.length > 0 ? (
              <EquityCurve
                data={strategyCurve}
                benchmark={benchmark.length > 0 ? benchmark : undefined}
              />
            ) : (
              <EmptyState message="No equity history was produced for this backtest." />
            )}
          </Panel>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: space.lg,
            }}
          >
            <MetricTile
              label="Total return"
              value={pct(result.total_return_pct)}
              tone={signTone(totalReturn)}
            />
            <MetricTile
              label="Benchmark return"
              value={pct(benchReturn)}
              tone={benchReturn === null ? "neutral" : signTone(Number(benchReturn))}
            />
            <MetricTile
              label="Max drawdown"
              value={pct(result.max_drawdown_pct)}
              tone={maxDrawdown > 0 ? "danger" : "neutral"}
            />
            <MetricTile label="Sharpe" value={ratio(result.sharpe)} tone="accent" />
            <MetricTile label="Sortino" value={ratio(result.sortino)} tone="accent" />
            <MetricTile label="Win rate" value={pct(result.win_rate, 0)} />
            <MetricTile label="Profit factor" value={ratio(result.profit_factor)} />
            <MetricTile
              label="Annualized return"
              value={pct(result.annualized_return_pct)}
              tone={
                result.annualized_return_pct === null
                  ? "neutral"
                  : signTone(Number(result.annualized_return_pct))
              }
            />
            <MetricTile label="Volatility" value={pct(result.volatility)} />
            <MetricTile label="Avg win" value={money(result.avg_win)} tone="ok" />
            <MetricTile label="Avg loss" value={money(result.avg_loss)} tone="danger" />
            <MetricTile label="Trades" value={formatNumber(result.num_trades, 0)} />
            <MetricTile label="Decisions" value={formatNumber(result.decision_count, 0)} />
            <MetricTile label="Start equity" value={money(result.starting_equity)} />
            <MetricTile label="End equity" value={money(result.ending_equity)} />
          </div>

          <Panel>
            <SectionHeader
              eyebrow="Executions"
              title={`Trades (${result.trades.length})`}
              caption="Every order the strategy would have filled during the replay."
              actions={
                result.trades.length > 0 ? (
                  <Badge tone="muted">
                    <Figure weight={700}>{result.trades.length}</Figure> fills
                  </Badge>
                ) : undefined
              }
            />
            {result.trades.length > 0 ? (
              <DataGrid<BacktestTrade>
                rows={result.trades}
                columns={tradeColumns}
                getRowId={(r) => `${r.symbol}-${r.filled_at}-${r.side}-${r.quantity}`}
                height={Math.min(420, Math.max(160, 60 + result.trades.length * 44))}
              />
            ) : (
              <EmptyState message="No trades were generated in this backtest." />
            )}
          </Panel>
        </>
      )}
    </StackLayout>
  );
}

function LegendDot({ color: dotColor, label }: { color: string; label: string }) {
  return (
    <span
      style={{
        ...text.caption,
        color: color.muted,
        display: "inline-flex",
        alignItems: "center",
        gap: space.xs,
      }}
    >
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: radius.xs,
          background: dotColor,
          display: "inline-block",
          transition: transitions("background"),
        }}
      />
      {label}
    </span>
  );
}
