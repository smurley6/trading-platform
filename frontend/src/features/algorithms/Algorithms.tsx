// Algorithm selector + per-algorithm detail (FR-UI-6): config, equity curve, signals,
// trades, and a backtest panel (M4).

import type { ColDef } from "ag-grid-community";
import { Button, Card, H2, H3, StackLayout, Text } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { EquityCurve } from "../../components/EquityCurve";
import { SymbolChart } from "../../components/SymbolChart";
import { api } from "../../lib/api";
import { formatMoney, formatNumber } from "../../lib/format";
import { blotterRows, equitySeries } from "../../lib/projections";
import type { BlotterRow, EquityPoint } from "../../lib/projections";
import type { BacktestResult, BacktestTrade, Sleeve } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

const sleeveColumns: ColDef<Sleeve>[] = [
  { headerName: "Sleeve", field: "id" },
  { headerName: "Strategy", field: "strategy_id" },
  { headerName: "Mode", field: "mode", maxWidth: 110 },
  { headerName: "Status", field: "status", maxWidth: 120 },
  {
    headerName: "Capital",
    field: "allocated_capital",
    valueFormatter: (p) => formatMoney(p.value as string),
  },
];

const tradeColumns: ColDef<BlotterRow>[] = [
  { headerName: "Symbol", field: "symbol" },
  { headerName: "Side", field: "side" },
  { headerName: "Qty", field: "quantity" },
  { headerName: "Status", field: "status" },
];

interface AlgorithmsProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

export function Algorithms({ refreshKey, sleeves }: AlgorithmsProps) {
  const events = useLoader(() => api.events(1000), [refreshKey]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selected = sleeves.find((s) => s.id === (selectedId ?? sleeves[0]?.id)) ?? null;
  const all = useMemo(() => events.data ?? [], [events.data]);
  const series = selected ? equitySeries(all, selected.id) : [];
  const trades = useMemo(
    () => (selected ? blotterRows(all).filter((r) => r.sleeve_id === selected.id) : []),
    [all, selected],
  );

  return (
    <StackLayout gap={2}>
      <Card>
        <H3>Algorithms</H3>
        <DataGrid<Sleeve>
          rows={sleeves}
          columns={sleeveColumns}
          getRowId={(r) => r.id}
          onRowClick={(r) => setSelectedId(r.id)}
          height={Math.max(140, 60 + sleeves.length * 44)}
        />
      </Card>

      {selected && (
        <>
          <Card>
            <H2 style={{ marginTop: 0 }}>{selected.strategy_id}</H2>
            <Text color="secondary">
              Sleeve {selected.id} · {selected.mode} · status {selected.status} ·{" "}
              {formatMoney(selected.allocated_capital)} allocated
            </Text>
            {series.length > 0 ? (
              <EquityCurve data={series} />
            ) : (
              <Text color="secondary">No equity history yet.</Text>
            )}
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <Card>
              <H3>Trades</H3>
              <DataGrid<BlotterRow>
                rows={trades}
                columns={tradeColumns}
                getRowId={(r) => r.client_order_id}
                height={300}
              />
            </Card>
            <Card>
              <H3>Market context</H3>
              <SymbolChart symbol={`NASDAQ:${trades[0]?.symbol ?? "AAPL"}`} height={300} />
            </Card>
          </div>

          <BacktestPanel />
        </>
      )}
    </StackLayout>
  );
}

const backtestTradeColumns: ColDef<BacktestTrade>[] = [
  { headerName: "Symbol", field: "symbol" },
  { headerName: "Side", field: "side" },
  { headerName: "Qty", field: "quantity" },
  { headerName: "Price", field: "price", valueFormatter: (p) => formatMoney(p.value as number) },
];

function BacktestPanel() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.backtest());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  // Backtest snapshots can share a timestamp; lightweight-charts needs strictly increasing.
  const curve: EquityPoint[] = [];
  let last = 0;
  for (const point of result?.equity_curve ?? []) {
    const time = point.time <= last ? last + 1 : point.time;
    last = time;
    curve.push({ time, value: point.equity });
  }

  return (
    <Card>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <H3>Backtest</H3>
        <Button disabled={busy} onClick={() => void run()}>
          {busy ? "Running…" : "Run backtest"}
        </Button>
      </div>
      <Text color="secondary" styleAs="label">
        Replays recorded signals through the political-signal strategy and the same risk + execution
        path used live. Prices are illustrative until a historical market-data adapter is added.
      </Text>
      {error && <Text color="error">{error}</Text>}
      {result && (
        <>
          <div style={{ display: "flex", gap: 24, margin: "12px 0", flexWrap: "wrap" }}>
            <Metric label="Return" value={`${formatNumber(result.total_return_pct, 2)}%`} />
            <Metric label="Max drawdown" value={`${formatNumber(result.max_drawdown_pct, 2)}%`} />
            <Metric label="Start equity" value={formatMoney(result.starting_equity)} />
            <Metric label="End equity" value={formatMoney(result.ending_equity)} />
            <Metric label="Decisions" value={formatNumber(result.decision_count, 0)} />
          </div>
          {curve.length > 0 && <EquityCurve data={curve} />}
          <div style={{ marginTop: 12 }}>
            <Text styleAs="label">Trades ({result.trades.length})</Text>
            <DataGrid<BacktestTrade>
              rows={result.trades}
              columns={backtestTradeColumns}
              getRowId={(r) => `${r.symbol}-${r.filled_at}-${r.side}`}
              height={220}
            />
          </div>
        </>
      )}
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Text color="secondary" styleAs="label">
        {label}
      </Text>
      <div style={{ fontSize: 20, fontWeight: 600 }}>{value}</div>
    </div>
  );
}
