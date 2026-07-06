// Blotter `/blotter` (UI Phase C, docs/UI-DESIGN.md §5.4): a single newest-first AG Grid on the
// §3 `.ag-theme-tp` adapter — submitted ts · sleeve · ticker · side (order-side colored) · qty ·
// type · lifecycle-colored status chip · fill price · client order id (mono, click-copies with a
// toast). Row click → Chain Inspector on the order's correlation id (the full §6 lifecycle).
// `x` on the focused cancelable row → cancel with the §7 single-confirm keystroke, optimistic
// (the chip flips to canceling immediately; the live event stream is authoritative).
//
// Projection + chip mapping live in orderLifecycle.ts (pure, tested). All visual values are §3
// tokens; c/t hotkeys ride the shared focused-row model (shell/useScreenOverlays.ts).

import type { CellFocusedEvent, ColDef, ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import { useCallback, useMemo, useRef, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { useToast } from "../../components/Toast";
import { api } from "../../lib/api";
import { formatMoney, formatNumber, formatTime, formatTimeUtc } from "../../lib/format";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../../lib/tokens";
import { useLoader } from "../../lib/useLoader";
import { ChainInspector } from "../../shell/ChainInspector";
import { ConfirmChip } from "../../shell/ConfirmChip";
import { SlideOverStackProvider } from "../../shell/SlideOverStackProvider";
import { ShellEmptyState, SkeletonRows } from "../../shell/states";
import { TickerPane } from "../../shell/TickerPane";
import { useScreenOverlays } from "../../shell/useScreenOverlays";
import { isCancelable, orderRows, statusChip, type OrderRow, type StatusTone } from "./orderLifecycle";

interface BlotterProps {
  active: boolean;
  refreshKey: number;
}

export function Blotter({ active, refreshKey }: BlotterProps) {
  const toast = useToast();
  const events = useLoader(() => api.events(1000), [refreshKey]);
  const rows = useMemo(() => orderRows(events.data ?? []), [events.data]);

  // ---- Optimistic cancels (cleared implicitly: once the cancel event lands, the projection
  // reports a terminal status and the flag no longer matters; rollback on API failure).
  const [canceling, setCanceling] = useState<ReadonlySet<string>>(new Set());
  const cancel = useCallback(
    (row: OrderRow) => {
      setCanceling((prev) => new Set(prev).add(row.client_order_id));
      api
        .cancelOrder(row.client_order_id)
        .then(() => {
          toast.success(`Cancel requested for ${row.client_order_id}`);
        })
        .catch((err: unknown) => {
          setCanceling((prev) => {
            const next = new Set(prev);
            next.delete(row.client_order_id);
            return next;
          });
          toast.error(
            `Cancel ${row.client_order_id} failed: ${err instanceof Error ? err.message : String(err)}`,
          );
        });
    },
    [toast],
  );

  // ---- Focused row (AG Grid cell focus) + shared overlays: click/c → chain, t → chart.
  const [focusedRow, setFocusedRow] = useState<OrderRow | null>(null);
  const focusedRowRef = useRef<OrderRow | null>(null);
  focusedRowRef.current = focusedRow;
  const overlays = useScreenOverlays(active, {
    correlationId: () => focusedRowRef.current?.correlation_id ?? null,
    ticker: () => focusedRowRef.current?.symbol ?? null,
  });

  const onCellFocused = useCallback((e: CellFocusedEvent<OrderRow>) => {
    if (e.rowIndex == null) return;
    const node = e.api.getDisplayedRowAtIndex(e.rowIndex);
    setFocusedRow(node?.data ?? null);
  }, []);

  // ---- `x` on the focused cancelable row → confirm chip (§5.4 / §7 single confirm).
  const [pendingCancel, setPendingCancel] = useState<OrderRow | null>(null);
  useHotkeys(
    "x",
    () => {
      const row = focusedRowRef.current;
      if (row && isCancelable(row) && !canceling.has(row.client_order_id)) {
        setPendingCancel(row);
      }
    },
    {
      enabled: active && !overlays.overlayOpen && pendingCancel === null,
      enableOnFormTags: false,
    },
    [active, overlays.overlayOpen, pendingCancel, canceling],
  );

  const copyOrderId = useCallback(
    (id: string) => {
      void navigator.clipboard
        .writeText(id)
        .then(() => toast.info(`Copied ${id}`))
        .catch(() => toast.error(`Could not copy ${id}`));
    },
    [toast],
  );

  const columns = useMemo<ColDef<OrderRow>[]>(
    () => buildColumns(canceling, copyOrderId),
    [canceling, copyOrderId],
  );

  const initialLoading = events.loading && !events.data;

  return (
    <SlideOverStackProvider>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: layout.x3,
          height: `calc(100vh - var(--status-bar-height) - ${layout.x6 * 2}px)`,
          minHeight: 0,
        }}
      >
        {/* Header strip: count + the pending-cancel confirm chip (keyboard flow: x → Enter). */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: layout.x3,
            flexShrink: 0,
            minHeight: shape.rowHeightTicker,
          }}
        >
          <span
            className="tp-numeric"
            style={{ color: colors.textSecondary, fontSize: typeTokens.scale.meta }}
          >
            {rows.length} order{rows.length === 1 ? "" : "s"} ·{" "}
            {rows.filter(isCancelable).length} open
          </span>
          <span style={{ flex: 1 }} />
          {pendingCancel && (
            <ConfirmChip
              label={`Cancel ${pendingCancel.symbol} ${pendingCancel.client_order_id}`}
              onConfirm={() => {
                const row = pendingCancel;
                setPendingCancel(null);
                if (row) cancel(row);
              }}
              onCancel={() => setPendingCancel(null)}
            />
          )}
        </div>

        <div style={{ flex: 1, minHeight: 0 }}>
          {initialLoading ? (
            <SkeletonRows rows={3} />
          ) : events.error ? (
            <ShellEmptyState message={`Could not load orders: ${events.error}`} />
          ) : rows.length === 0 ? (
            <ShellEmptyState message="No orders yet — approvals and cycle trades will appear here as they execute." />
          ) : (
            <div className="ag-theme-tp" style={{ height: "100%", width: "100%" }}>
              <AgGridReact<OrderRow>
                rowData={rows}
                columnDefs={columns}
                defaultColDef={{ sortable: true, resizable: true, flex: 1 }}
                getRowId={(p) => p.data.client_order_id}
                rowHeight={shape.rowHeightDense}
                headerHeight={shape.rowHeightDense}
                onRowClicked={(e) => {
                  // The coid cell's click-to-copy must not ALSO open the Chain Inspector. AG
                  // Grid's own row listener fires before React's synthetic stopPropagation can,
                  // so the guard has to live here, on the underlying DOM event's target.
                  const target = e.event?.target as HTMLElement | null;
                  if (target?.closest("[data-no-row-click]")) return;
                  if (e.data) overlays.openChain(e.data.correlation_id);
                }}
                onCellFocused={onCellFocused}
                animateRows
              />
            </div>
          )}
        </div>
      </div>

      <ChainInspector
        correlationId={overlays.correlationId}
        onClose={overlays.closeChain}
        returnFocusRef={overlays.lastRowRef}
      />
      <TickerPane
        ticker={overlays.ticker}
        onClose={overlays.closeTicker}
        returnFocusRef={overlays.lastRowRef}
      />
    </SlideOverStackProvider>
  );
}

// ---- Columns ------------------------------------------------------------------------------------

const RIGHT_ALIGNED = "ag-right-aligned-cell";

// §5.4 chip tones → §3.1 tokens. `lossOutline` = --loss border/text, transparent fill; the
// filled/gain-loss chips get a low-alpha fill of the same reserved color.
function chipColors(tone: StatusTone): { border: string; color: string; background: string } {
  switch (tone) {
    case "muted":
      return { border: colors.border, color: colors.textMuted, background: "transparent" };
    case "accent":
      return {
        border: colors.accent,
        color: colors.accent,
        background: withAlpha(colors.accent, 0.1),
      };
    case "gain":
      return { border: colors.gain, color: colors.gain, background: withAlpha(colors.gain, 0.1) };
    case "loss":
      return { border: colors.loss, color: colors.loss, background: withAlpha(colors.loss, 0.1) };
    case "lossOutline":
      return { border: colors.loss, color: colors.loss, background: "transparent" };
  }
}

function buildColumns(
  canceling: ReadonlySet<string>,
  copyOrderId: (id: string) => void,
): ColDef<OrderRow>[] {
  const StatusCell = (params: ICellRendererParams<OrderRow>) => {
    const row = params.data;
    if (!row) return null;
    const inFlight = canceling.has(row.client_order_id) && isCancelable(row);
    const chip = statusChip(row.status, row.side);
    const palette = chipColors(inFlight ? "muted" : chip.tone);
    return (
      <span
        style={{
          border: `1px solid ${palette.border}`,
          borderRadius: shape.radiusBadge,
          padding: `0 ${layout.x1}px`,
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.meta,
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: palette.color,
          background: palette.background,
        }}
      >
        {inFlight ? "canceling…" : chip.label}
      </span>
    );
  };

  const CoidCell = (params: ICellRendererParams<OrderRow, string>) => {
    const id = params.value;
    if (!id) return null;
    return (
      <button
        type="button"
        title={`${id} — click to copy`}
        data-no-row-click // consumed by onRowClicked's guard (see grid props)
        onClick={() => copyOrderId(id)}
        className="tp-numeric"
        style={{
          cursor: "copy",
          border: "none",
          background: "transparent",
          padding: 0,
          color: colors.textSecondary,
          fontSize: typeTokens.scale.body,
          maxWidth: "100%",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {id}
      </button>
    );
  };

  return [
    {
      headerName: "Submitted",
      field: "occurred_at",
      cellRenderer: (p: ICellRendererParams<OrderRow, string>) =>
        p.value ? (
          <span className="tp-numeric" title={formatTimeUtc(p.value)} style={{ color: colors.textSecondary }}>
            {formatTime(p.value)}
          </span>
        ) : null,
      minWidth: 150,
      sort: "desc",
    },
    { headerName: "Sleeve", field: "sleeve_id", minWidth: 110 },
    {
      headerName: "Ticker",
      field: "symbol",
      cellRenderer: (p: ICellRendererParams<OrderRow, string>) => (
        <span className="tp-numeric" style={{ fontWeight: 600, color: colors.textPrimary }}>
          {p.value}
        </span>
      ),
      maxWidth: 100,
    },
    {
      headerName: "Side",
      field: "side",
      cellRenderer: (p: ICellRendererParams<OrderRow, string>) => {
        const side = (p.value ?? "").toLowerCase();
        if (!side) return null;
        return (
          <span
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: typeTokens.scale.meta,
              fontWeight: 600,
              letterSpacing: "0.04em",
              color: side === "sell" ? colors.loss : colors.gain,
            }}
          >
            {side.toUpperCase()}
          </span>
        );
      },
      maxWidth: 90,
    },
    {
      headerName: "Qty",
      field: "quantity",
      valueFormatter: (p) => formatNumber((p.value as string) ?? "", 4),
      cellClass: RIGHT_ALIGNED,
      headerClass: RIGHT_ALIGNED,
      maxWidth: 100,
    },
    { headerName: "Type", field: "order_type", maxWidth: 100 },
    { headerName: "Status", field: "status", cellRenderer: StatusCell, minWidth: 130 },
    {
      headerName: "Fill price",
      field: "fill_price",
      valueFormatter: (p) => (p.value ? formatMoney(p.value as string) : "—"),
      cellClass: RIGHT_ALIGNED,
      headerClass: RIGHT_ALIGNED,
      maxWidth: 120,
    },
    {
      headerName: "Client order id",
      field: "client_order_id",
      cellRenderer: CoidCell,
      minWidth: 180,
    },
  ];
}
