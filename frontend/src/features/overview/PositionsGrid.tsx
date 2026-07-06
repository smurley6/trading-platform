// Positions grid (§5.1: "Positions grid (live marks, P&L flash)") — the first grid rebuilt onto
// the §3 AG Grid adapter (`.ag-theme-tp`, lib/theme-adapters/agGridTheme.ts). Falls back to
// event-derived net positions (no live marks) when the live-quotes feed is disabled, same
// graceful-degradation shape as the legacy Dashboard.
//
// Numeric columns right-align via the built-in `ag-right-aligned-cell` class directly (not the
// `type: "rightAlignedColumn"` shortcut every legacy screen uses — that shortcut isn't a
// registered column type in AG Grid Community 32.x without an explicit `gridOptions.columnTypes`
// entry, so it silently no-ops with a console warning; pre-existing across every legacy grid,
// out of scope to fix there this phase, but new grids shouldn't repeat it).

import type { ColDef, ICellRendererParams, ValueFormatterParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import { formatMoney, formatNumber } from "../../lib/format";
import { colors, shape } from "../../lib/tokens";
import { ShellEmptyState } from "../../shell/states";
import { FlashFigure } from "./FlashCell";
import type { PositionRowView } from "./positionRows";

function moneyCell(params: ICellRendererParams<PositionRowView, number>) {
  const v = params.value;
  if (v == null || Number.isNaN(v)) return <span style={{ color: colors.textMuted }}>—</span>;
  const field = params.colDef?.field ?? "value";
  return (
    <FlashFigure value={v} flashKey={`${params.data?.key}|${field}`} color={colors.textPrimary}>
      {formatMoney(v)}
    </FlashFigure>
  );
}

function pnlCell(params: ICellRendererParams<PositionRowView, number>) {
  const v = params.value;
  if (v == null || Number.isNaN(v)) return <span style={{ color: colors.textMuted }}>—</span>;
  const field = params.colDef?.field ?? "pnl";
  return (
    <FlashFigure value={v} flashKey={`${params.data?.key}|${field}`} color={v >= 0 ? colors.gain : colors.loss}>
      {formatMoney(v)}
    </FlashFigure>
  );
}

const RIGHT_ALIGNED = "ag-right-aligned-cell";

const columns: ColDef<PositionRowView>[] = [
  { headerName: "Symbol", field: "symbol", maxWidth: 100 },
  { headerName: "Sleeve", field: "sleeve_id", maxWidth: 110 },
  {
    headerName: "Qty",
    field: "quantity",
    maxWidth: 90,
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    valueFormatter: (p: ValueFormatterParams<PositionRowView, number>) =>
      formatNumber(p.value ?? 0, 0),
  },
  {
    headerName: "Avg",
    field: "avg_price",
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    cellRenderer: moneyCell,
  },
  {
    headerName: "Last",
    field: "last",
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    cellRenderer: moneyCell,
  },
  {
    headerName: "Mkt value",
    field: "market_value",
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    cellRenderer: moneyCell,
  },
  {
    headerName: "P&L",
    field: "unrealized_pnl",
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    cellRenderer: pnlCell,
  },
];

export function PositionsGrid({ rows }: { rows: PositionRowView[] }) {
  if (rows.length === 0) {
    return <ShellEmptyState message="No open positions." />;
  }
  return (
    <div className="ag-theme-tp" style={{ height: "100%", width: "100%" }}>
      <AgGridReact<PositionRowView>
        rowData={rows}
        columnDefs={columns}
        defaultColDef={{ sortable: true, resizable: true, flex: 1 }}
        getRowId={(p) => p.data.key}
        rowHeight={shape.rowHeightDense}
        headerHeight={shape.rowHeightDense}
        suppressCellFocus
        animateRows
      />
    </div>
  );
}
