// Algorithms screen — algorithm selector (cards) + per-algorithm detail: a header with name /
// mode / status / capital and a performance stats row, a prominent equity curve, then supporting
// panels (conviction breakdown, signal attribution, trades, market context) and the backtest
// panel. All analytics are derived client-side from the event log + conviction (see analytics.ts).

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { Button, Card, H2, H3, StackLayout, Text } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { EquityCurve } from "../../components/EquityCurve";
import { SymbolChart } from "../../components/SymbolChart";
import { api } from "../../lib/api";
import { formatMoney, formatNumber, formatTime } from "../../lib/format";
import { blotterRows, equitySeries } from "../../lib/projections";
import type { BlotterRow, EquityPoint } from "../../lib/projections";
import type { BacktestResult, BacktestTrade, Sleeve } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";
import {
  algoStats,
  convictionBreakdown,
  signalAttribution,
  type AlgoStats,
  type ConvictionRow,
  type DecisionAttribution,
} from "./analytics";

interface AlgorithmsProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

export function Algorithms({ refreshKey, sleeves }: AlgorithmsProps) {
  const events = useLoader(() => api.events(1000), [refreshKey]);
  const conviction = useLoader(() => api.conviction(), [refreshKey]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const all = useMemo(() => events.data ?? [], [events.data]);
  const selected = sleeves.find((s) => s.id === (selectedId ?? sleeves[0]?.id)) ?? null;

  const stats = useMemo<AlgoStats | null>(
    () => (selected ? algoStats(all, selected.id) : null),
    [all, selected],
  );
  const series = useMemo(() => (selected ? equitySeries(all, selected.id) : []), [all, selected]);
  const trades = useMemo(
    () => (selected ? blotterRows(all).filter((r) => r.sleeve_id === selected.id) : []),
    [all, selected],
  );
  const convictionRows = useMemo<ConvictionRow[]>(
    () => (selected ? convictionBreakdown(conviction.data ?? [], all, selected.id) : []),
    [conviction.data, all, selected],
  );
  const attribution = useMemo<DecisionAttribution[]>(
    () => (selected ? signalAttribution(all, selected.id) : []),
    [all, selected],
  );

  // Per-sleeve quick stats for the selector cards.
  const cardStats = useMemo(() => {
    const map = new Map<string, AlgoStats>();
    for (const s of sleeves) map.set(s.id, algoStats(all, s.id));
    return map;
  }, [sleeves, all]);

  return (
    <StackLayout gap={2}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
          gap: 12,
        }}
      >
        {sleeves.map((s) => (
          <AlgorithmCard
            key={s.id}
            sleeve={s}
            stats={cardStats.get(s.id)}
            active={s.id === selected?.id}
            onSelect={() => setSelectedId(s.id)}
          />
        ))}
      </div>

      {selected && stats && (
        <>
          <Card>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                flexWrap: "wrap",
                gap: 8,
              }}
            >
              <div>
                <H2 style={{ margin: 0 }}>{selected.strategy_id}</H2>
                <Text color="secondary">
                  Sleeve {selected.id} · {formatMoney(selected.allocated_capital)} allocated
                </Text>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <Pill
                  text={selected.mode.toUpperCase()}
                  tone={selected.mode === "live" ? "danger" : "info"}
                />
                <Pill
                  text={selected.status}
                  tone={selected.status === "running" ? "success" : "muted"}
                />
              </div>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
                gap: 16,
                margin: "16px 0",
              }}
            >
              <Stat
                label="Total return"
                value={`${formatNumber(stats.totalReturnPct, 2)}%`}
                tone={signTone(stats.totalReturnPct)}
              />
              <Stat
                label="Max drawdown"
                value={`${formatNumber(stats.maxDrawdownPct, 2)}%`}
                tone={stats.maxDrawdownPct > 0 ? "danger" : "muted"}
              />
              <Stat
                label="Realized P&L"
                value={formatMoney(stats.realizedPnl)}
                tone={signTone(stats.realizedPnl)}
              />
              <Stat
                label="Win rate"
                value={stats.winRatePct === null ? "—" : `${formatNumber(stats.winRatePct, 0)}%`}
              />
              <Stat label="Decisions" value={formatNumber(stats.decisionCount, 0)} />
              <Stat label="Fills" value={formatNumber(stats.fillCount, 0)} />
            </div>

            {series.length > 0 ? (
              <EquityCurve data={series} />
            ) : (
              <Text color="secondary">
                No equity history yet — submit signals to generate activity.
              </Text>
            )}
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <Card>
              <H3 style={{ marginTop: 0 }}>Conviction by ticker</H3>
              <Text color="secondary" styleAs="label">
                Current per-ticker conviction; tickers this algo has acted on are marked.
              </Text>
              {convictionRows.length > 0 ? (
                <DataGrid<ConvictionRow>
                  rows={convictionRows}
                  columns={convictionColumns}
                  getRowId={(r) => r.ticker}
                  height={300}
                />
              ) : (
                <Text color="secondary">No conviction within the corroboration window.</Text>
              )}
            </Card>
            <Card>
              <H3 style={{ marginTop: 0 }}>Market context</H3>
              <SymbolChart symbol={`NASDAQ:${trades[0]?.symbol ?? "AAPL"}`} height={300} />
            </Card>
          </div>

          <Card>
            <H3 style={{ marginTop: 0 }}>Signal attribution</H3>
            <Text color="secondary" styleAs="label">
              Each decision and the recorded signals that drove it (source · ticker · sentiment).
            </Text>
            {attribution.length > 0 ? (
              <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 10 }}>
                {attribution.slice(0, 25).map((d) => (
                  <DecisionRow key={d.decision_id} decision={d} />
                ))}
              </div>
            ) : (
              <Text color="secondary">No decisions recorded for this algorithm yet.</Text>
            )}
          </Card>

          <Card>
            <H3 style={{ marginTop: 0 }}>Trades</H3>
            <DataGrid<BlotterRow>
              rows={trades}
              columns={tradeColumns}
              getRowId={(r) => r.client_order_id}
              height={280}
            />
          </Card>

          <BacktestPanel />
        </>
      )}
    </StackLayout>
  );
}

// --- selector cards ----------------------------------------------------------------------

function AlgorithmCard({
  sleeve,
  stats,
  active,
  onSelect,
}: {
  sleeve: Sleeve;
  stats: AlgoStats | undefined;
  active: boolean;
  onSelect: () => void;
}) {
  const ret = stats?.totalReturnPct ?? 0;
  return (
    <button
      onClick={onSelect}
      style={{
        cursor: "pointer",
        textAlign: "left",
        color: "inherit",
        borderRadius: 10,
        padding: 16,
        border: `1px solid ${active ? "#4c9aff" : "rgba(255,255,255,0.12)"}`,
        background: active ? "rgba(76,154,255,0.14)" : "rgba(255,255,255,0.02)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ fontSize: 15 }}>{sleeve.strategy_id}</strong>
        <Pill text={sleeve.mode.toUpperCase()} tone={sleeve.mode === "live" ? "danger" : "info"} />
      </div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 2 }}>
        {sleeve.id} · {sleeve.status}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12 }}>
        <div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>Capital</div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>
            {formatMoney(sleeve.allocated_capital)}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>Return</div>
          <div style={{ fontSize: 15, fontWeight: 600, color: toneColor(signTone(ret)) }}>
            {formatNumber(ret, 2)}%
          </div>
        </div>
      </div>
    </button>
  );
}

// --- conviction grid: in-cell bar --------------------------------------------------------

function ConvictionBarCell(params: ICellRendererParams<ConvictionRow, number>) {
  const row = params.data;
  if (!row) return null;
  const pct = Math.round(row.scorePct * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, height: "100%" }}>
      <div
        style={{
          flex: 1,
          height: 8,
          borderRadius: 4,
          background: "rgba(255,255,255,0.08)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: row.traded ? "#4c9aff" : "rgba(148,163,184,0.6)",
          }}
        />
      </div>
      <span style={{ fontVariantNumeric: "tabular-nums", minWidth: 44, textAlign: "right" }}>
        {formatNumber(row.score, 1)}
      </span>
    </div>
  );
}

const convictionColumns: ColDef<ConvictionRow>[] = [
  { headerName: "Ticker", field: "ticker", maxWidth: 110 },
  { headerName: "Action", field: "action", maxWidth: 110 },
  {
    headerName: "Acted",
    field: "traded",
    maxWidth: 90,
    valueFormatter: (p) => (p.value ? "yes" : ""),
  },
  {
    headerName: "Conviction",
    field: "score",
    flex: 2,
    sortable: true,
    cellRenderer: ConvictionBarCell,
  },
];

// --- signal attribution row --------------------------------------------------------------

function DecisionRow({ decision }: { decision: DecisionAttribution }) {
  const score = decision.convictionScore;
  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8,
        padding: "10px 12px",
        background: "rgba(255,255,255,0.02)",
      }}
    >
      <div
        style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}
      >
        <div>
          <strong style={{ fontSize: 14 }}>
            {decision.side.toUpperCase()} {formatNumber(decision.quantity, 0)} {decision.symbol}
          </strong>{" "}
          <span style={{ fontSize: 12, color: "#94a3b8" }}>{formatTime(decision.occurred_at)}</span>
        </div>
        <span style={{ fontSize: 12, color: "#94a3b8" }}>
          conviction {score === null ? "—" : formatNumber(score, 1)}
        </span>
      </div>
      {decision.rationale && (
        <div style={{ fontSize: 12, color: "#a3a8b4", marginTop: 4 }}>{decision.rationale}</div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
        {decision.signals.length === 0 ? (
          <span style={{ fontSize: 12, color: "#6b7280" }}>No contributing signals recorded.</span>
        ) : (
          decision.signals.map((sig) => (
            <span
              key={sig.signal_id}
              title={
                sig.resolved ? `signal ${sig.signal_id}` : `unresolved signal ${sig.signal_id}`
              }
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 10,
                border: "1px solid rgba(255,255,255,0.15)",
                background: sig.resolved ? "rgba(76,154,255,0.12)" : "rgba(148,163,184,0.1)",
                color: sig.resolved ? "#cbd5e1" : "#94a3b8",
              }}
            >
              {sig.resolved ? (
                <>
                  {sig.source_type} · {sig.ticker} ·{" "}
                  <span style={{ color: toneColor(signTone(sig.sentiment)) }}>
                    {formatNumber(sig.sentiment, 2)}
                  </span>
                </>
              ) : (
                "unresolved signal"
              )}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

// --- shared presentational bits ----------------------------------------------------------

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

function Pill({ text, tone }: { text: string; tone: Tone }) {
  const color = toneColor(tone);
  return (
    <span
      style={{
        padding: "2px 10px",
        borderRadius: 10,
        fontSize: 11,
        fontWeight: 700,
        color,
        border: `1px solid ${color}`,
        background: `${color}22`,
      }}
    >
      {text}
    </span>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: Tone }) {
  return (
    <div>
      <Text color="secondary" styleAs="label">
        {label}
      </Text>
      <div style={{ fontSize: 20, fontWeight: 600, color: tone ? toneColor(tone) : undefined }}>
        {value}
      </div>
    </div>
  );
}

const tradeColumns: ColDef<BlotterRow>[] = [
  { headerName: "Symbol", field: "symbol" },
  { headerName: "Side", field: "side" },
  { headerName: "Qty", field: "quantity" },
  { headerName: "Status", field: "status" },
];

// --- backtest panel (M4) -----------------------------------------------------------------

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
        <H3 style={{ margin: 0 }}>Backtest</H3>
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
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
              gap: 16,
              margin: "16px 0",
            }}
          >
            <Stat
              label="Return"
              value={`${formatNumber(result.total_return_pct, 2)}%`}
              tone={signTone(result.total_return_pct)}
            />
            <Stat
              label="Max drawdown"
              value={`${formatNumber(result.max_drawdown_pct, 2)}%`}
              tone={result.max_drawdown_pct > 0 ? "danger" : "muted"}
            />
            <Stat label="Start equity" value={formatMoney(result.starting_equity)} />
            <Stat label="End equity" value={formatMoney(result.ending_equity)} />
            <Stat label="Decisions" value={formatNumber(result.decision_count, 0)} />
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
