// Portfolio dashboard (FR-UI-6): aggregate + per-sleeve, equity curve, paper/live indicator.

import type { ColDef } from "ag-grid-community";
import { Card, H2, H3, StackLayout, Text } from "@salt-ds/core";
import { useEffect, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { EquityCurve } from "../../components/EquityCurve";
import { api } from "../../lib/api";
import { formatMoney, formatNumber } from "../../lib/format";
import { equitySeries, latestSnapshots, positionRows } from "../../lib/projections";
import type { SleeveSnapshot } from "../../lib/projections";
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

const columns: ColDef<SleeveRow>[] = [
  { headerName: "Sleeve", field: "id" },
  { headerName: "Strategy", field: "strategy_id" },
  { headerName: "Mode", field: "mode", maxWidth: 110 },
  {
    headerName: "Allocated",
    field: "allocated",
    valueFormatter: (p) => formatMoney(p.value as string),
  },
  { headerName: "Equity", field: "equity", valueFormatter: (p) => formatMoney(p.value as number) },
  { headerName: "Positions", field: "positions", maxWidth: 130 },
];

const livePositionColumns: ColDef<LivePosition>[] = [
  { headerName: "Symbol", field: "symbol", maxWidth: 120 },
  { headerName: "Sleeve", field: "sleeve_id" },
  { headerName: "Qty", field: "quantity", maxWidth: 110 },
  {
    headerName: "Avg price",
    field: "avg_price",
    valueFormatter: (p) => formatMoney(p.value as string),
  },
  { headerName: "Last", field: "last", valueFormatter: (p) => formatMoney(p.value as string) },
  {
    headerName: "Market value",
    field: "market_value",
    valueFormatter: (p) => formatMoney(p.value as string),
  },
  {
    headerName: "Unrealized P&L",
    field: "unrealized_pnl",
    valueFormatter: (p) => formatMoney(p.value as string),
  },
];

interface DashboardProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

export function Dashboard({ refreshKey, sleeves }: DashboardProps) {
  const events = useLoader(() => api.events(1000), [refreshKey]);
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

  return (
    <StackLayout gap={2}>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <SummaryCard label="Total equity" value={formatMoney(totalEquity)} />
        <SummaryCard label="Sleeves" value={formatNumber(sleeves.length, 0)} />
        <SummaryCard label="Open positions" value={formatNumber(openPositions, 0)} />
        {liveOn && (
          <>
            <SummaryCard label="Live market value" value={formatMoney(liveMarketValue)} />
            <SummaryCard label="Unrealized P&L" value={formatMoney(liveUnrealized)} />
          </>
        )}
      </div>

      {liveOn && livePositions.length > 0 && (
        <Card>
          <H3>Live positions</H3>
          <DataGrid<LivePosition>
            rows={livePositions}
            columns={livePositionColumns}
            getRowId={(r) => `${r.sleeve_id}|${r.symbol}`}
            height={Math.max(140, 60 + livePositions.length * 44)}
          />
        </Card>
      )}

      <Card>
        <H3>Sleeves</H3>
        <DataGrid<SleeveRow>
          rows={rows}
          columns={columns}
          getRowId={(r) => r.id}
          height={Math.max(140, 60 + rows.length * 44)}
        />
      </Card>

      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <H3>Equity curve</H3>
          <div style={{ display: "flex", gap: 8 }}>
            {sleeves.map((s) => (
              <button
                key={s.id}
                onClick={() => setSelected(s.id)}
                style={{
                  cursor: "pointer",
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: s.id === activeSleeve ? "rgba(76,154,255,0.2)" : "transparent",
                  color: "inherit",
                  borderRadius: 6,
                  padding: "2px 10px",
                  fontSize: 12,
                }}
              >
                {s.id}
              </button>
            ))}
          </div>
        </div>
        {series.length > 0 ? (
          <EquityCurve data={series} />
        ) : (
          <Text color="secondary">
            No portfolio snapshots yet — submit signals to generate activity.
          </Text>
        )}
      </Card>
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

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <Card style={{ minWidth: 200, flex: 1 }}>
      <Text color="secondary" styleAs="label">
        {label}
      </Text>
      <H2 style={{ margin: "4px 0 0" }}>{value}</H2>
    </Card>
  );
}
