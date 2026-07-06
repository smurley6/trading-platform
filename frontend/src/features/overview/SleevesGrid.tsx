// Sleeves grid (§5.1: "id · strategy · mode · status · held value") — Algorithms folded into
// Overview as a grid section per §2 ("two sleeves, one inert — a grid section, not a page"). The
// full Algorithms screen/route is NOT deleted this phase (still reachable via the nav rail
// trailing entry from Phase A) — see the Phase B report for why.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import { formatMoney } from "../../lib/format";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../../lib/tokens";
import type { Sleeve } from "../../lib/types";
import { ShellEmptyState } from "../../shell/states";

export interface SleeveRowView {
  id: string;
  strategy_id: string;
  mode: Sleeve["mode"];
  status: Sleeve["status"];
  held_value: number;
}

function ModeCell(params: ICellRendererParams<SleeveRowView, string>) {
  const live = params.value === "live";
  // Mode isn't P&L/order-side (§3.1 reserves gain/loss for that), so this stays neutral text
  // rather than reaching for green/red — same rule the status bar's mode chip already follows.
  return (
    <span style={{ color: live ? colors.textPrimary : colors.textSecondary, fontWeight: 600 }}>
      {(params.value ?? "").toUpperCase()}
    </span>
  );
}

function StatusCell(params: ICellRendererParams<SleeveRowView, string>) {
  const running = params.value === "running";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: layout.x1,
        padding: `1px ${layout.x2}px`,
        borderRadius: shape.radiusBadge,
        border: `1px solid ${running ? colors.border : colors.warn}`,
        background: running ? "transparent" : withAlpha(colors.warn, 0.1),
        color: running ? colors.textSecondary : colors.warn,
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
      }}
    >
      {(params.value ?? "").toUpperCase()}
    </span>
  );
}

function moneyCell(params: ICellRendererParams<SleeveRowView, number>) {
  return <span className="tp-numeric">{formatMoney(params.value ?? 0)}</span>;
}

const columns: ColDef<SleeveRowView>[] = [
  { headerName: "ID", field: "id", maxWidth: 140 },
  { headerName: "Strategy", field: "strategy_id" },
  { headerName: "Mode", field: "mode", maxWidth: 90, cellRenderer: ModeCell },
  { headerName: "Status", field: "status", maxWidth: 110, cellRenderer: StatusCell },
  {
    // §5.1 numeric alignment: right-align via the built-in `ag-right-aligned-cell` class directly
    // (see PositionsGrid.tsx's docblock — `type: "rightAlignedColumn"` isn't a registered column
    // type in this AG Grid Community version without explicit gridOptions.columnTypes).
    headerName: "Held value",
    field: "held_value",
    cellClass: "ag-right-aligned-cell",
    headerClass: "ag-right-aligned-cell",
    cellRenderer: moneyCell,
  },
];

export function SleevesGrid({ rows }: { rows: SleeveRowView[] }) {
  if (rows.length === 0) {
    return <ShellEmptyState message="No algorithm sleeves are configured yet." />;
  }
  return (
    <div className="ag-theme-tp" style={{ height: "100%", width: "100%" }}>
      <AgGridReact<SleeveRowView>
        rowData={rows}
        columnDefs={columns}
        defaultColDef={{ sortable: true, resizable: true, flex: 1 }}
        getRowId={(p) => p.data.id}
        rowHeight={shape.rowHeightDense}
        headerHeight={shape.rowHeightDense}
        suppressCellFocus
        animateRows
      />
    </div>
  );
}
