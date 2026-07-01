// Order blotter and positions (FR-UI-3), both AG Grid surfaces projected from the event log.
// Open orders carry an inline Cancel action wired to POST /orders/{client_order_id}/cancel; the
// cancel is optimistic (the row flips to "canceling…" immediately) and the live event stream
// drives the authoritative refresh once the backend records the cancellation.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { Button, StackLayout } from "@salt-ds/core";
import { useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import {
  Badge,
  EmptyState,
  LoadingState,
  MetricTile,
  Panel,
  SectionHeader,
} from "../../components/ui";
import { api } from "../../lib/api";
import { formatNumber, formatTime } from "../../lib/format";
import { blotterRows, positionRows } from "../../lib/projections";
import type { BlotterRow, PositionRow } from "../../lib/projections";
import { color, numeric, pillStyle, radius, space, text, transitions } from "../../lib/tokens";
import { useLoader } from "../../lib/useLoader";

// Statuses that mean the order is done — no Cancel action and not part of "open orders".
const TERMINAL_STATUSES = new Set([
  "filled",
  "canceled",
  "cancelled",
  "rejected",
  "expired",
  "submit_failed",
  "done_for_day",
]);

function isOpen(row: BlotterRow): boolean {
  return !TERMINAL_STATUSES.has(row.status.toLowerCase());
}

// Map a raw order status to a pill tone so the grid reads at a glance. Fills are "ok", hard
// failures are "danger", cancellations/expiries are muted, and everything still working is accent.
function statusTone(status: string): "accent" | "ok" | "danger" | "warn" | "muted" {
  const s = status.toLowerCase();
  if (s === "filled") return "ok";
  if (s === "rejected" || s === "submit_failed") return "danger";
  if (s === "partially_filled") return "warn";
  if (TERMINAL_STATUSES.has(s)) return "muted";
  return "accent";
}

// Side rendered as a tinted pill — buy green / sell red — matching the semantic P&L palette.
const SideCell = (params: ICellRendererParams<BlotterRow>) => {
  const side = (params.value as string | undefined) ?? params.data?.side;
  if (!side) return null;
  const tone =
    side.toLowerCase() === "buy" ? "ok" : side.toLowerCase() === "sell" ? "danger" : "muted";
  return <span style={{ ...pillStyle(tone), textTransform: "uppercase" }}>{side}</span>;
};

const positionColumns: ColDef<PositionRow>[] = [
  { headerName: "Sleeve", field: "sleeve_id" },
  { headerName: "Symbol", field: "symbol", maxWidth: 140 },
  {
    headerName: "Quantity",
    field: "quantity",
    type: "rightAlignedColumn",
    maxWidth: 180,
    valueFormatter: (p) => formatNumber(p.value as number, 4),
    cellRenderer: (p: ICellRendererParams<PositionRow>) => {
      const qty = p.value as number;
      return (
        <span style={{ ...numeric(), color: qty >= 0 ? color.ok : color.dangerText }}>
          {formatNumber(qty, 4)}
        </span>
      );
    },
  },
];

interface BlotterProps {
  refreshKey: number;
}

export function Blotter({ refreshKey }: BlotterProps) {
  const events = useLoader(() => api.events(1000), [refreshKey]);
  const all = events.data ?? [];

  // Optimistic local state for in-flight cancels, keyed by client_order_id. Cleared whenever the
  // event log reloads (refreshKey changes) so the authoritative projection wins.
  const [canceling, setCanceling] = useState<Set<string>>(new Set());
  const [cancelError, setCancelError] = useState<string | null>(null);

  async function cancel(row: BlotterRow): Promise<void> {
    setCancelError(null);
    setCanceling((prev) => new Set(prev).add(row.client_order_id));
    try {
      await api.cancelOrder(row.client_order_id);
      // The live event stream bumps refreshKey, which reloads the blotter and clears this row's
      // optimistic flag below.
    } catch (err) {
      setCanceling((prev) => {
        const next = new Set(prev);
        next.delete(row.client_order_id);
        return next;
      });
      setCancelError(err instanceof Error ? err.message : String(err));
    }
  }

  const rows = blotterRows(all);
  // Effective optimistic set: an order is only still "canceling" if the latest projection also
  // shows it open. Once the cancel lands in the event log the order leaves the open set, so we
  // drop its optimistic flag (derived during render — never mutate the state Set in place).
  const stillOpen = new Set(rows.filter(isOpen).map((r) => r.client_order_id));
  const effectiveCanceling = new Set([...canceling].filter((id) => stillOpen.has(id)));

  const StatusCell = (params: ICellRendererParams<BlotterRow>) => {
    const row = params.data;
    if (!row) return null;
    const inFlight = effectiveCanceling.has(row.client_order_id);
    const label = inFlight ? "canceling…" : row.status;
    const tone = inFlight ? "warn" : statusTone(row.status);
    return <Badge tone={tone}>{label}</Badge>;
  };

  const CancelCell = (params: ICellRendererParams<BlotterRow>) => {
    const row = params.data;
    if (!row || !isOpen(row)) return null;
    const inFlight = effectiveCanceling.has(row.client_order_id);
    return (
      <Button
        disabled={inFlight}
        onClick={() => void cancel(row)}
        style={{
          height: 28,
          padding: "0 14px",
          borderRadius: radius.md,
          transition: transitions("background", "border-color"),
        }}
      >
        {inFlight ? "Canceling…" : "Cancel"}
      </Button>
    );
  };

  const orderColumns: ColDef<BlotterRow>[] = [
    {
      headerName: "Time",
      field: "occurred_at",
      maxWidth: 130,
      valueFormatter: (p) => formatTime(p.value as string),
    },
    { headerName: "Sleeve", field: "sleeve_id" },
    { headerName: "Symbol", field: "symbol", maxWidth: 110 },
    { headerName: "Side", field: "side", maxWidth: 110, cellRenderer: SideCell },
    { headerName: "Qty", field: "quantity", maxWidth: 110, type: "rightAlignedColumn" },
    {
      headerName: "Status",
      field: "status",
      maxWidth: 170,
      cellRenderer: StatusCell,
      valueFormatter: (p) =>
        p.data && effectiveCanceling.has(p.data.client_order_id)
          ? "canceling…"
          : (p.value as string),
    },
    { headerName: "Broker ID", field: "broker_order_id" },
    {
      headerName: "",
      colId: "actions",
      maxWidth: 130,
      sortable: false,
      filter: false,
      cellRenderer: CancelCell,
    },
  ];

  const positions = positionRows(all);
  // First-load only: show a spinner while the very first events request is in flight. On
  // refresh (data already present) the grids stay put rather than flashing the spinner.
  const initialLoading = events.loading && !events.data;
  const loadError = events.error;

  // Summary metrics — derived from the current projection for the KPI strip.
  const openCount = rows.filter(isOpen).length;
  const filledCount = rows.filter((r) => r.status.toLowerCase() === "filled").length;
  const rejectedCount = rows.filter((r) =>
    ["rejected", "submit_failed"].includes(r.status.toLowerCase()),
  ).length;

  return (
    <StackLayout gap={2}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: space.lg,
        }}
      >
        <MetricTile
          label="Open Orders"
          value={openCount}
          tone="accent"
          hint="Working / cancelable"
        />
        <MetricTile label="Filled" value={filledCount} tone="ok" hint="Completed today" />
        <MetricTile
          label="Rejected"
          value={rejectedCount}
          tone={rejectedCount > 0 ? "danger" : "neutral"}
          hint="Rejected / failed"
        />
        <MetricTile label="Positions" value={positions.length} hint="Sleeve × symbol" />
      </div>

      <Panel>
        <SectionHeader
          eyebrow="Live"
          title="Positions"
          caption="Net open quantity per sleeve and symbol"
          actions={
            positions.length > 0 ? (
              <Badge tone="muted">
                {positions.length} {positions.length === 1 ? "row" : "rows"}
              </Badge>
            ) : undefined
          }
        />
        {initialLoading ? (
          <LoadingState label="Loading positions…" />
        ) : loadError ? (
          <EmptyState title="Couldn't load positions" message={loadError} icon="⚠" />
        ) : positions.length > 0 ? (
          <DataGrid<PositionRow>
            rows={positions}
            columns={positionColumns}
            getRowId={(r) => `${r.sleeve_id}|${r.symbol}`}
            height={240}
          />
        ) : (
          <EmptyState
            title="No open positions"
            message="Fills will appear here as orders execute."
            icon="◇"
          />
        )}
      </Panel>

      <Panel>
        <SectionHeader
          eyebrow="Order flow"
          title="Order blotter"
          caption="Every order from the event log — open orders can be canceled inline"
          actions={
            <div style={{ display: "flex", alignItems: "center", gap: space.sm }}>
              {cancelError && (
                <span style={{ ...text.caption, color: color.dangerText }}>{cancelError}</span>
              )}
              {rows.length > 0 && (
                <Badge tone={openCount > 0 ? "accent" : "muted"} dot={openCount > 0}>
                  {openCount} open
                </Badge>
              )}
            </div>
          }
        />
        {initialLoading ? (
          <LoadingState label="Loading orders…" />
        ) : loadError ? (
          <EmptyState title="Couldn't load orders" message={loadError} icon="⚠" />
        ) : rows.length > 0 ? (
          <DataGrid<BlotterRow>
            rows={rows}
            columns={orderColumns}
            getRowId={(r) => r.client_order_id}
            height={420}
          />
        ) : (
          <EmptyState
            title="No orders yet"
            message="Submit signals to generate order activity."
            icon="◎"
          />
        )}
      </Panel>
    </StackLayout>
  );
}
