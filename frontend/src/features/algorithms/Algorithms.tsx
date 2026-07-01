// Algorithms screen — algorithm selector (cards) + per-algorithm detail: a header with name /
// mode / status / capital and a performance stats row, a prominent equity curve, then supporting
// panels (conviction breakdown, signal attribution, trades, market context). All analytics are
// derived client-side from the event log + conviction (see analytics.ts). The dedicated Backtest
// screen (features/backtest) runs and renders historical backtests.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { StackLayout } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { EquityCurve } from "../../components/EquityCurve";
import { SymbolChart } from "../../components/SymbolChart";
import {
  Badge,
  EmptyState,
  Figure,
  LoadingState,
  MetricTile,
  Panel,
  SectionHeader,
} from "../../components/ui";
import { api } from "../../lib/api";
import { formatMoney, formatNumber, formatTime } from "../../lib/format";
import { blotterRows, equitySeries } from "../../lib/projections";
import type { BlotterRow } from "../../lib/projections";
import { color, eyebrowStyle, numeric, radius, space, text, transitions } from "../../lib/tokens";
import type { Sleeve } from "../../lib/types";
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

type MetricTone = "neutral" | "ok" | "danger" | "accent" | "warn";
type BadgeTone = "accent" | "ok" | "danger" | "warn" | "muted";

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
      <Panel>
        <SectionHeader
          eyebrow="Strategy sleeves"
          title="Algorithms"
          caption="Select a sleeve to inspect its performance, conviction, and signal attribution."
        />
        {sleeves.length === 0 ? (
          <EmptyState message="No algorithm sleeves are configured yet." />
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
              gap: space.md,
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
        )}
      </Panel>

      {events.loading && !selected && <LoadingState label="Loading algorithm activity…" />}

      {selected && stats && (
        <>
          <Panel>
            <SectionHeader
              eyebrow={`Sleeve ${selected.id} · ${formatMoney(selected.allocated_capital)} allocated`}
              title={selected.strategy_id}
              actions={
                <>
                  <Badge tone={selected.mode === "live" ? "danger" : "accent"} dot>
                    {selected.mode.toUpperCase()}
                  </Badge>
                  <Badge tone={selected.status === "running" ? "ok" : "muted"} dot>
                    {selected.status.toUpperCase()}
                  </Badge>
                </>
              }
            />

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
                gap: space.lg,
                marginBottom: space.xl,
              }}
            >
              <MetricTile
                label="Total return"
                value={`${formatNumber(stats.totalReturnPct, 2)}%`}
                tone={signMetricTone(stats.totalReturnPct)}
              />
              <MetricTile
                label="Max drawdown"
                value={`${formatNumber(stats.maxDrawdownPct, 2)}%`}
                tone={stats.maxDrawdownPct > 0 ? "danger" : "neutral"}
              />
              <MetricTile
                label="Realized P&L"
                value={formatMoney(stats.realizedPnl)}
                tone={signMetricTone(stats.realizedPnl)}
              />
              <MetricTile
                label="Win rate"
                value={stats.winRatePct === null ? "—" : `${formatNumber(stats.winRatePct, 0)}%`}
                hint={stats.closedTrades > 0 ? `${stats.closedTrades} closed` : undefined}
              />
              <MetricTile
                label="Decisions"
                value={formatNumber(stats.decisionCount, 0)}
                tone="accent"
              />
              <MetricTile label="Fills" value={formatNumber(stats.fillCount, 0)} />
            </div>

            {series.length > 0 ? (
              <EquityCurve data={series} />
            ) : (
              <EmptyState message="No equity history yet — submit signals to generate activity." />
            )}
          </Panel>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: space.lg }}>
            <Panel>
              <SectionHeader
                title="Conviction by ticker"
                caption="Current per-ticker conviction; tickers this algo has acted on are marked."
              />
              {convictionRows.length > 0 ? (
                <DataGrid<ConvictionRow>
                  rows={convictionRows}
                  columns={convictionColumns}
                  getRowId={(r) => r.ticker}
                  height={300}
                />
              ) : (
                <EmptyState message="No conviction within the corroboration window." />
              )}
            </Panel>
            <Panel>
              <SectionHeader
                title="Market context"
                caption={`Live chart for ${trades[0]?.symbol ?? "AAPL"}.`}
              />
              <SymbolChart symbol={`NASDAQ:${trades[0]?.symbol ?? "AAPL"}`} height={300} />
            </Panel>
          </div>

          <Panel>
            <SectionHeader
              title="Signal attribution"
              caption="Each decision and the recorded signals that drove it (source · ticker · sentiment)."
            />
            {attribution.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: space.sm }}>
                {attribution.slice(0, 25).map((d) => (
                  <DecisionRow key={d.decision_id} decision={d} />
                ))}
              </div>
            ) : (
              <EmptyState message="No decisions recorded for this algorithm yet." />
            )}
          </Panel>

          <Panel>
            <SectionHeader title="Trades" caption={`${trades.length} recorded orders.`} />
            {trades.length > 0 ? (
              <DataGrid<BlotterRow>
                rows={trades}
                columns={tradeColumns}
                getRowId={(r) => r.client_order_id}
                height={280}
              />
            ) : (
              <EmptyState message="No trades recorded for this algorithm yet." />
            )}
          </Panel>
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
      aria-pressed={active}
      style={{
        cursor: "pointer",
        textAlign: "left",
        color: color.text,
        borderRadius: radius.lg,
        padding: space.lg,
        border: `1px solid ${active ? color.accentBorder : color.hairline}`,
        background: active ? color.accentSoft : color.panelSoft,
        boxShadow: active ? `inset 0 0 0 1px ${color.accentBorder}` : "none",
        transition: transitions("background", "border-color", "box-shadow"),
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ ...text.h3, color: color.text }}>{sleeve.strategy_id}</strong>
        <Badge tone={sleeve.mode === "live" ? "danger" : "accent"}>
          {sleeve.mode.toUpperCase()}
        </Badge>
      </div>
      <div style={{ ...text.caption, color: color.muted, marginTop: space.xxs }}>
        {sleeve.id} · {sleeve.status}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: space.md }}>
        <div>
          <div style={eyebrowStyle()}>Capital</div>
          <Figure color={color.text} style={{ ...text.h3, marginTop: 2, display: "block" }}>
            {formatMoney(sleeve.allocated_capital)}
          </Figure>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={eyebrowStyle()}>Return</div>
          <Figure pnl={ret} style={{ ...text.h3, marginTop: 2, display: "block" }}>
            {formatNumber(ret, 2)}%
          </Figure>
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
    <div style={{ display: "flex", alignItems: "center", gap: space.sm, height: "100%" }}>
      <div
        style={{
          flex: 1,
          height: 8,
          borderRadius: radius.xs,
          background: color.panelInset,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            borderRadius: radius.xs,
            background: row.traded ? color.accent : color.faint,
          }}
        />
      </div>
      <span
        style={{
          ...numeric(),
          minWidth: 44,
          textAlign: "right",
          color: row.traded ? color.text : color.textDim,
        }}
      >
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
  const sideTone: BadgeTone = decision.side === "sell" ? "danger" : "ok";
  return (
    <div
      style={{
        border: `1px solid ${color.hairline}`,
        borderRadius: radius.md,
        padding: `${space.md}px ${space.md}px`,
        background: color.panelSoft,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: space.sm,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: space.sm, flexWrap: "wrap" }}>
          <Badge tone={sideTone}>{decision.side.toUpperCase()}</Badge>
          <span style={{ ...text.label, color: color.text }}>
            <Figure color={color.text}>{formatNumber(decision.quantity, 0)}</Figure>{" "}
            {decision.symbol}
          </span>
          <span style={{ ...text.caption, color: color.muted }}>
            {formatTime(decision.occurred_at)}
          </span>
        </div>
        <span style={{ ...text.caption, color: color.muted }}>
          conviction{" "}
          <Figure color={color.textDim}>{score === null ? "—" : formatNumber(score, 1)}</Figure>
        </span>
      </div>
      {decision.rationale && (
        <div style={{ ...text.caption, color: color.textDim, marginTop: space.xs }}>
          {decision.rationale}
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: space.xs, marginTop: space.sm }}>
        {decision.signals.length === 0 ? (
          <span style={{ ...text.caption, color: color.faint }}>
            No contributing signals recorded.
          </span>
        ) : (
          decision.signals.map((sig, i) =>
            sig.resolved ? (
              <Badge key={`${sig.signal_id}-${i}`} tone="muted" title={`signal ${sig.signal_id}`}>
                {sig.source_type} · {sig.ticker} ·{" "}
                <Figure pnl={sig.sentiment}>{formatNumber(sig.sentiment, 2)}</Figure>
              </Badge>
            ) : (
              <Badge
                key={`${sig.signal_id}-${i}`}
                tone="muted"
                title={`unresolved signal ${sig.signal_id}`}
                style={{ opacity: 0.7 }}
              >
                unresolved signal
              </Badge>
            ),
          )
        )}
      </div>
    </div>
  );
}

// --- shared presentational bits ----------------------------------------------------------

function signMetricTone(value: number): MetricTone {
  if (value > 0) return "ok";
  if (value < 0) return "danger";
  return "neutral";
}

function StatusCell(params: ICellRendererParams<BlotterRow, string>) {
  const value = params.value ?? "";
  const tone: BadgeTone =
    value === "filled" ? "ok" : value === "rejected" || value === "cancelled" ? "danger" : "muted";
  return <Badge tone={tone}>{value.toUpperCase()}</Badge>;
}

function SideCell(params: ICellRendererParams<BlotterRow, string>) {
  const value = params.value ?? "";
  return (
    <span style={{ color: value === "sell" ? color.dangerText : color.ok, fontWeight: 600 }}>
      {value.toUpperCase()}
    </span>
  );
}

const tradeColumns: ColDef<BlotterRow>[] = [
  { headerName: "Symbol", field: "symbol" },
  { headerName: "Side", field: "side", cellRenderer: SideCell, maxWidth: 120 },
  { headerName: "Qty", field: "quantity", type: "rightAlignedColumn" },
  { headerName: "Status", field: "status", cellRenderer: StatusCell },
];
