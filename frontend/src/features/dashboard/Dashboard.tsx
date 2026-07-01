// Portfolio dashboard (FR-UI-6): aggregate + per-sleeve, equity curve, paper/live indicator.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { StackLayout } from "@salt-ds/core";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
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
  Segmented,
} from "../../components/ui";
import { api } from "../../lib/api";
import { formatMoney, formatNumber } from "../../lib/format";
import { equitySeries, latestSnapshots, positionRows } from "../../lib/projections";
import type { SleeveSnapshot } from "../../lib/projections";
import { color, pillStyle, space } from "../../lib/tokens";
import type { LivePosition, LivePositionsResult, Sleeve } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

const LIVE_POLL_INTERVAL_MS = 5_000;

interface SleeveRow {
  id: string;
  strategy_id: string;
  mode: string;
  allocated: string;
  equity: number;
  positions: number;
}

// Right-aligned numeric cell: mono/tabular figure so the design's numeral rule applies in-grid.
function moneyCell<T>(params: ICellRendererParams<T>): ReactNode {
  return <Figure color={color.textDim}>{formatMoney(params.value as number | string)}</Figure>;
}

// P&L cell: mono figure auto-colored green/red by sign.
function pnlCell<T>(params: ICellRendererParams<T>): ReactNode {
  const n = Number(params.value);
  return (
    <Figure pnl={Number.isNaN(n) ? 0 : n}>{formatMoney(params.value as number | string)}</Figure>
  );
}

// Count cell: mono figure in the dim body tone.
function countCell<T>(params: ICellRendererParams<T>): ReactNode {
  return <Figure color={color.textDim}>{formatNumber(params.value as number, 0)}</Figure>;
}

// Mode pill (paper/live) rendered as an on-theme badge inside the grid.
function modeCell<T>(params: ICellRendererParams<T>): ReactNode {
  const mode = String(params.value ?? "").toLowerCase();
  const live = mode === "live";
  return (
    <span style={{ ...pillStyle(live ? "danger" : "muted"), fontSize: 10 }}>
      {mode ? mode.toUpperCase() : "—"}
    </span>
  );
}

const columns: ColDef<SleeveRow>[] = [
  { headerName: "Sleeve", field: "id", minWidth: 140 },
  { headerName: "Strategy", field: "strategy_id", minWidth: 140 },
  { headerName: "Mode", field: "mode", maxWidth: 110, cellRenderer: modeCell },
  {
    headerName: "Allocated",
    field: "allocated",
    type: "rightAlignedColumn",
    cellRenderer: moneyCell,
  },
  {
    headerName: "Equity",
    field: "equity",
    type: "rightAlignedColumn",
    cellRenderer: moneyCell,
  },
  {
    headerName: "Positions",
    field: "positions",
    maxWidth: 130,
    type: "rightAlignedColumn",
    cellRenderer: countCell,
  },
];

const livePositionColumns: ColDef<LivePosition>[] = [
  { headerName: "Symbol", field: "symbol", maxWidth: 120 },
  { headerName: "Sleeve", field: "sleeve_id" },
  {
    headerName: "Qty",
    field: "quantity",
    maxWidth: 110,
    type: "rightAlignedColumn",
    cellRenderer: countCell,
  },
  {
    headerName: "Avg price",
    field: "avg_price",
    type: "rightAlignedColumn",
    cellRenderer: moneyCell,
  },
  { headerName: "Last", field: "last", type: "rightAlignedColumn", cellRenderer: moneyCell },
  {
    headerName: "Market value",
    field: "market_value",
    type: "rightAlignedColumn",
    cellRenderer: moneyCell,
  },
  {
    headerName: "Unrealized P&L",
    field: "unrealized_pnl",
    type: "rightAlignedColumn",
    cellRenderer: pnlCell,
  },
];

interface DashboardProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

export function Dashboard({ refreshKey, sleeves }: DashboardProps) {
  const events = useLoader(() => api.events(1000), [refreshKey]);
  // Command-center signals: pending proposals + cycle mode share one loader; kill-switch state.
  const proposals = useLoader(() => api.proposals(), [refreshKey]);
  const killSwitch = useLoader(() => api.killSwitch(), [refreshKey]);
  // Optional live P&L overlay. Defaults to OFF (no credentials) and resolves to a disabled
  // shape on any error, so the rest of the dashboard is unaffected when it's not available.
  // Once enabled, it refreshes on a light interval so market value / unrealized P&L track the
  // moving feed in near-real-time; while disabled it stays dormant (no polling) — like today.
  const live = useLivePositions(refreshKey);
  const [selected, setSelected] = useState<string | null>(null);

  const allEvents = events.data ?? [];
  const snapshots = latestSnapshots(allEvents);
  const openPositions = positionRows(allEvents).length;

  const rows: SleeveRow[] = sleeves.map((s) => {
    const snap: SleeveSnapshot | undefined = snapshots.get(s.id);
    return {
      id: s.id,
      strategy_id: s.strategy_id,
      mode: s.mode,
      allocated: s.allocated_capital,
      equity: snap ? snap.equity : Number(s.allocated_capital),
      positions: snap ? snap.positionCount : 0,
    };
  });
  const totalEquity = rows.reduce((sum, r) => sum + r.equity, 0);
  const activeSleeve = selected ?? sleeves[0]?.id ?? null;
  const series = activeSleeve ? equitySeries(allEvents, activeSleeve) : [];

  const livePositions: LivePosition[] = live.enabled ? live.positions : [];
  const liveOn = live.enabled;
  const liveUnrealized = livePositions.reduce((sum, p) => sum + Number(p.unrealized_pnl), 0);
  const liveMarketValue = livePositions.reduce((sum, p) => sum + Number(p.market_value), 0);

  // Command-center derived values.
  const pendingProposals = proposals.data?.proposals.length ?? 0;
  const cycleMode = proposals.data?.mode ?? "—";
  const tradingLive = sleeves.some((s) => s.mode === "live");
  const killEngaged = killSwitch.data?.engaged ?? false;

  return (
    <StackLayout gap={2}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: space.lg,
        }}
      >
        <MetricTile label="Total equity" value={formatMoney(totalEquity)} tone="accent" />
        <MetricTile label="Sleeves" value={formatNumber(sleeves.length, 0)} />
        <MetricTile label="Open positions" value={formatNumber(openPositions, 0)} />
        <MetricTile label="Pending proposals" value={formatNumber(pendingProposals, 0)} />
        <MetricTile label="Cycle mode" value={cycleMode} />
        <MetricTile
          label="Trading mode"
          value={tradingLive ? "LIVE" : "PAPER"}
          tone={tradingLive ? "danger" : "ok"}
        />
        <MetricTile
          label="Kill switch"
          value={killEngaged ? "HALTED" : "ARMED"}
          tone={killEngaged ? "danger" : "ok"}
        />
        {liveOn && (
          <>
            <MetricTile label="Live market value" value={formatMoney(liveMarketValue)} />
            <MetricTile
              label="Unrealized P&L"
              value={formatMoney(liveUnrealized)}
              tone={liveUnrealized >= 0 ? "ok" : "danger"}
            />
          </>
        )}
      </div>

      {liveOn && livePositions.length > 0 && (
        <Panel padded={false}>
          <SectionHeader
            style={{ padding: `${space.xl}px ${space.xl}px 0` }}
            eyebrow="Real-time feed"
            title="Live positions"
            caption="Marks tracking the live market data feed"
            actions={
              <Badge tone="ok" dot>
                LIVE QUOTES
              </Badge>
            }
          />
          <div style={{ padding: `0 ${space.xl}px ${space.xl}px` }}>
            <DataGrid<LivePosition>
              rows={livePositions}
              columns={livePositionColumns}
              getRowId={(r) => `${r.sleeve_id}|${r.symbol}`}
              height={Math.max(140, 60 + livePositions.length * 44)}
            />
          </div>
        </Panel>
      )}

      <Panel padded={false}>
        <SectionHeader
          style={{ padding: `${space.xl}px ${space.xl}px 0` }}
          eyebrow="Portfolio"
          title="Sleeves"
          caption="Allocation and equity across strategy sleeves"
          actions={
            <Badge tone="muted">
              {sleeves.length} {sleeves.length === 1 ? "sleeve" : "sleeves"}
            </Badge>
          }
        />
        <div style={{ padding: `0 ${space.xl}px ${space.xl}px` }}>
          {events.loading && !events.data ? (
            <LoadingState label="Loading portfolio activity…" />
          ) : events.error ? (
            <EmptyState title="Could not load activity" message={events.error} />
          ) : rows.length > 0 ? (
            <DataGrid<SleeveRow>
              rows={rows}
              columns={columns}
              getRowId={(r) => r.id}
              height={Math.max(140, 60 + rows.length * 44)}
            />
          ) : (
            <EmptyState
              title="No sleeves configured"
              message="Configure a strategy sleeve to begin allocating capital."
            />
          )}
        </div>
      </Panel>

      <Panel>
        <SectionHeader
          eyebrow="Performance"
          title="Equity curve"
          caption={activeSleeve ? `Snapshots for ${activeSleeve}` : undefined}
          actions={
            sleeves.length > 0 && activeSleeve != null ? (
              <Segmented
                size="sm"
                options={sleeves.map((s) => ({ value: s.id, label: s.id }))}
                value={activeSleeve}
                onChange={setSelected}
              />
            ) : undefined
          }
        />
        {series.length > 0 ? (
          <EquityCurve data={series} />
        ) : (
          <EmptyState
            title="No snapshots yet"
            message="Submit signals to generate portfolio activity and plot the equity curve."
          />
        )}
      </Panel>
    </StackLayout>
  );
}

// Live positions overlay: fetch once on mount / refreshKey change, then — only while the feed
// reports enabled — keep refreshing on a light interval so the P&L marks track the moving feed.
// When the feed is disabled (today's default) it never schedules an interval, so the Dashboard
// behaves exactly as before: a single resolved-to-disabled probe and nothing more.
function useLivePositions(refreshKey: number): LivePositionsResult {
  const [state, setState] = useState<LivePositionsResult>({ enabled: false, positions: [] });

  useEffect(() => {
    let cancelled = false;
    let pollTimer: number | undefined;

    const load = (): void => {
      void api.livePositions().then((result) => {
        if (cancelled) return;
        setState(result);
        // Schedule the next poll only while the feed is on; otherwise stay dormant.
        if (result.enabled && pollTimer === undefined) {
          pollTimer = window.setInterval(load, LIVE_POLL_INTERVAL_MS);
        } else if (!result.enabled && pollTimer !== undefined) {
          window.clearInterval(pollTimer);
          pollTimer = undefined;
        }
      });
    };

    load();

    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearInterval(pollTimer);
    };
  }, [refreshKey]);

  return state;
}
