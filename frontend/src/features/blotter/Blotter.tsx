// Order blotter and positions (FR-UI-3), both AG Grid surfaces projected from the event log.

import type { ColDef } from "ag-grid-community";
import { Card, H3, StackLayout } from "@salt-ds/core";
import { DataGrid } from "../../components/DataGrid";
import { api } from "../../lib/api";
import { formatNumber, formatTime } from "../../lib/format";
import { blotterRows, positionRows } from "../../lib/projections";
import type { BlotterRow, PositionRow } from "../../lib/projections";
import { useLoader } from "../../lib/useLoader";

const orderColumns: ColDef<BlotterRow>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    valueFormatter: (p) => formatTime(p.value as string),
  },
  { headerName: "Sleeve", field: "sleeve_id" },
  { headerName: "Symbol", field: "symbol", maxWidth: 110 },
  { headerName: "Side", field: "side", maxWidth: 100 },
  { headerName: "Qty", field: "quantity", maxWidth: 110 },
  { headerName: "Status", field: "status" },
  { headerName: "Broker ID", field: "broker_order_id" },
];

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

  return (
    <StackLayout gap={2}>
      <Card>
        <H3>Positions</H3>
        <DataGrid<PositionRow>
          rows={positionRows(all)}
          columns={positionColumns}
          getRowId={(r) => `${r.sleeve_id}|${r.symbol}`}
          height={240}
        />
      </Card>
      <Card>
        <H3>Order blotter</H3>
        <DataGrid<BlotterRow>
          rows={blotterRows(all)}
          columns={orderColumns}
          getRowId={(r) => r.client_order_id}
          height={420}
        />
      </Card>
    </StackLayout>
  );
}
