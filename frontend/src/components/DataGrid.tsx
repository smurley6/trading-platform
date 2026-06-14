// AG Grid wrapper (FR-UI-3): the single place tabular surfaces are built. Blotter,
// positions, algorithm list, and the signal feed all use this.

import type { ColDef } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";

interface DataGridProps<T> {
  rows: T[];
  columns: ColDef<T>[];
  height?: number;
  getRowId?: (row: T) => string;
  onRowClick?: (row: T) => void;
}

export function DataGrid<T>({
  rows,
  columns,
  height = 360,
  getRowId,
  onRowClick,
}: DataGridProps<T>) {
  return (
    <div className="ag-theme-quartz-dark" style={{ height, width: "100%" }}>
      <AgGridReact<T>
        rowData={rows}
        columnDefs={columns}
        defaultColDef={{ sortable: true, filter: true, resizable: true, flex: 1 }}
        getRowId={getRowId ? (params) => getRowId(params.data) : undefined}
        onRowClicked={onRowClick ? (event) => event.data && onRowClick(event.data) : undefined}
        animateRows
        suppressCellFocus
      />
    </div>
  );
}
