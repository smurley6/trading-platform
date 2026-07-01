// Order blotter and positions (FR-UI-3), both AG Grid surfaces projected from the event log.
// Open orders carry an inline Cancel action wired to POST /orders/{client_order_id}/cancel; the
// cancel is optimistic (the row flips to "canceling…" immediately) and the live event stream
// drives the authoritative refresh once the backend records the cancellation.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { Button, Card, H3, Spinner, StackLayout, Text } from "@salt-ds/core";
import { useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { api } from "../../lib/api";
import { formatNumber, formatTime } from "../../lib/format";
import { blotterRows, positionRows } from "../../lib/projections";
import type { BlotterRow, PositionRow } from "../../lib/projections";
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

// Color-code the side as a presentational overlay only — buy green / sell red, matching the
// tone palette used across the dashboard. Falls back to plain inherited text for any other value.
const SIDE_COLOR: Record<string, string> = { buy: "#4ade80", sell: "#f87171" };

const SideCell = (params: ICellRendererParams<BlotterRow>) => {
  const side = (params.value as string | undefined) ?? params.data?.side;
  if (!side) return null;
  const color = SIDE_COLOR[side.toLowerCase()];
  return (
    <span style={{ color, fontWeight: color ? 600 : undefined, textTransform: "uppercase" }}>
      {side}
    </span>
  );
};

const positionColumns: ColDef<PositionRow>[] = [
  { headerName: "Sleeve", field: "sleeve_id" },
  { headerName: "Symbol", field: "symbol" },
  {
    headerName: "Quantity",
    field: "quantity",
    valueFormatter: (p) => formatNumber(p.value as number, 4),
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

  const CancelCell = (params: ICellRendererParams<BlotterRow>) => {
    const row = params.data;
    if (!row || !isOpen(row)) return null;
    const inFlight = effectiveCanceling.has(row.client_order_id);
    return (
      <Button
        disabled={inFlight}
        onClick={() => void cancel(row)}
        style={{ height: 28, padding: "0 12px" }}
      >
        {inFlight ? "Canceling…" : "Cancel"}
      </Button>
    );
  };

  const orderColumns: ColDef<BlotterRow>[] = [
    {
      headerName: "Time",
      field: "occurred_at",
      valueFormatter: (p) => formatTime(p.value as string),
    },
    { headerName: "Sleeve", field: "sleeve_id" },
    { headerName: "Symbol", field: "symbol", maxWidth: 110 },
    { headerName: "Side", field: "side", maxWidth: 100, cellRenderer: SideCell },
    { headerName: "Qty", field: "quantity", maxWidth: 110 },
    {
      headerName: "Status",
      field: "status",
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

  return (
    <StackLayout gap={2}>
      <Card>
        <H3 style={{ marginTop: 0 }}>Positions</H3>
        <Text color="secondary" styleAs="label">
          Net open quantity per sleeve and symbol
        </Text>
        <div style={{ marginTop: 12 }}>
          {initialLoading ? (
            <Spinner />
          ) : loadError ? (
            <Text color="error">Could not load positions: {loadError}</Text>
          ) : positions.length > 0 ? (
            <DataGrid<PositionRow>
              rows={positions}
              columns={positionColumns}
              getRowId={(r) => `${r.sleeve_id}|${r.symbol}`}
              height={240}
            />
          ) : (
            <Text color="secondary">No open positions yet — fills will appear here.</Text>
          )}
        </div>
      </Card>
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <H3 style={{ marginTop: 0 }}>Order blotter</H3>
          {cancelError && <Text color="error">{cancelError}</Text>}
        </div>
        <Text color="secondary" styleAs="label">
          Every order from the event log — open orders can be canceled inline
        </Text>
        <div style={{ marginTop: 12 }}>
          {initialLoading ? (
            <Spinner />
          ) : loadError ? (
            <Text color="error">Could not load orders: {loadError}</Text>
          ) : rows.length > 0 ? (
            <DataGrid<BlotterRow>
              rows={rows}
              columns={orderColumns}
              getRowId={(r) => r.client_order_id}
              height={420}
            />
          ) : (
            <Text color="secondary">No orders yet — submit signals to generate activity.</Text>
          )}
        </div>
      </Card>
    </StackLayout>
  );
}
