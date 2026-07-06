// Signals `/signals` (UI Phase C, docs/UI-DESIGN.md §5.3): a single AG Grid on the §3
// `.ag-theme-tp` adapter with a filter bar above — source (multi), provenance
// (REAL/BACKFILLED/SYNTHETIC), ticker, unresolved-only, date range. NO conviction-band filter:
// conviction is per-ticker (GET /conviction), not per-signal — see signalFilters.ts and the
// phase report for the flagged gap. No raw payload in the grid (operator feedback: "no raw
// payload noise") — drill in via row click → Chain Inspector.
//
// Concise at any volume (binding operator feedback): a small default window, newest first,
// with a Load-more control that widens the backend fetch window (GET /signals/feed?limit=…,
// capped at the backend's 1000) — never a thousand-row wall by default. AG Grid virtualizes
// its own rows (§3.5).
//
// Provenance + correlation ids are joined from the events window (lib/signalMeta.ts) because
// the feed DTO carries neither; unknown rows render "—" and open no chain — never guessed.
// Filter state is addressable as /signals?provenance=…&source=… (§7).

import type { CellFocusedEvent, ColDef, ICellRendererParams } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import { useCallback, useMemo, useRef, useState } from "react";
import { useToast } from "../../components/Toast";
import { api } from "../../lib/api";
import { formatNumber, formatTime, formatTimeUtc } from "../../lib/format";
import { signalMetaFromEvents } from "../../lib/signalMeta";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../../lib/tokens";
import type { Provenance, SignalSourceType } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";
import { ChainInspector } from "../../shell/ChainInspector";
import { SlideOverStackProvider } from "../../shell/SlideOverStackProvider";
import { ShellEmptyState, SkeletonRows } from "../../shell/states";
import { TickerPane } from "../../shell/TickerPane";
import { useScreenQueryParam } from "../../shell/useHistoryRoute";
import { useScreenOverlays } from "../../shell/useScreenOverlays";
import {
  applySignalFilters,
  buildSignalRows,
  hasActiveFilters,
  provenanceFromParam,
  sourcesFromParam,
  sourcesToParam,
  type SignalFilters,
  type SignalRow,
} from "./signalFilters";

interface SignalsProps {
  active: boolean;
  refreshKey: number;
}

// Fetch-window steps (§5.3 "small default window … pagination/windowing to reach the rest").
// 1000 is the backend's hard cap on GET /signals/feed.
const WINDOW_STEPS = [100, 300, 1000] as const;

const SOURCE_OPTIONS: readonly { value: SignalSourceType; label: string }[] = [
  { value: "statement", label: "Statement" },
  { value: "contract", label: "Contract" },
  { value: "disclosed_trade", label: "Disclosed trade" },
];

const PROVENANCE_OPTIONS: readonly (Provenance | "ALL")[] = [
  "ALL",
  "REAL",
  "BACKFILLED",
  "SYNTHETIC",
];

export function Signals({ active, refreshKey }: SignalsProps) {
  const toast = useToast();
  const [windowLimit, setWindowLimit] = useState<number>(WINDOW_STEPS[0]);
  const signals = useLoader(() => api.signalFeed(windowLimit), [refreshKey, windowLimit]);
  const events = useLoader(() => api.events(1000), [refreshKey]);

  // ---- Filter state. provenance + source round-trip through the URL (§7); the rest is local.
  const [provenanceParam, setProvenanceParam] = useScreenQueryParam("provenance", active, "replace");
  const [sourceParam, setSourceParam] = useScreenQueryParam("source", active, "replace");
  const [ticker, setTicker] = useState("");
  const [unresolvedOnly, setUnresolvedOnly] = useState(false);
  const [from, setFrom] = useState<string | null>(null);
  const [to, setTo] = useState<string | null>(null);

  const filters: SignalFilters = useMemo(
    () => ({
      sources: sourcesFromParam(sourceParam),
      provenance: provenanceFromParam(provenanceParam),
      ticker,
      unresolvedOnly,
      from,
      to,
    }),
    [sourceParam, provenanceParam, ticker, unresolvedOnly, from, to],
  );

  const clearFilters = useCallback(() => {
    setProvenanceParam(null);
    setSourceParam(null);
    setTicker("");
    setUnresolvedOnly(false);
    setFrom(null);
    setTo(null);
  }, [setProvenanceParam, setSourceParam]);

  // ---- Rows: newest first, provenance/correlation joined from the events window.
  const meta = useMemo(() => signalMetaFromEvents(events.data ?? []), [events.data]);
  const allRows = useMemo(() => {
    const rows = buildSignalRows(signals.data ?? [], meta);
    return rows.sort((a, b) => b.occurred_at.localeCompare(a.occurred_at));
  }, [signals.data, meta]);
  const rows = useMemo(() => applySignalFilters(allRows, filters), [allRows, filters]);

  // ---- Focused row + overlays (shared model): click/c → Chain Inspector, t → Ticker Pane.
  const focusedRowRef = useRef<SignalRow | null>(null);
  const overlays = useScreenOverlays(active, {
    correlationId: () => focusedRowRef.current?.correlationId ?? null,
    ticker: () => focusedRowRef.current?.ticker ?? null,
  });

  const openChainForRow = useCallback(
    (row: SignalRow) => {
      if (row.correlationId) {
        overlays.openChain(row.correlationId);
      } else {
        toast.info(
          `No chain for ${row.ticker ?? row.signal_id}: its signal.received event is outside the loaded events window`,
        );
      }
    },
    [overlays, toast],
  );

  const onCellFocused = useCallback((e: CellFocusedEvent<SignalRow>) => {
    if (e.rowIndex == null) return;
    const node = e.api.getDisplayedRowAtIndex(e.rowIndex);
    focusedRowRef.current = node?.data ?? null;
  }, []);

  const columns = useMemo<ColDef<SignalRow>[]>(() => SIGNAL_COLUMNS, []);

  const initialLoading = signals.loading && !signals.data;
  const canLoadMore = windowLimit < WINDOW_STEPS[WINDOW_STEPS.length - 1]!;

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
        {/* ---- Filter bar --------------------------------------------------------------------- */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            flexWrap: "wrap",
            gap: layout.x2,
            flexShrink: 0,
          }}
        >
          <FilterGroup label="Source">
            {SOURCE_OPTIONS.map((opt) => {
              const selected = filters.sources.includes(opt.value);
              return (
                <FilterChip
                  key={opt.value}
                  label={opt.label}
                  selected={selected}
                  onClick={() => {
                    const next = selected
                      ? filters.sources.filter((s) => s !== opt.value)
                      : [...filters.sources, opt.value];
                    setSourceParam(sourcesToParam(next));
                  }}
                />
              );
            })}
          </FilterGroup>

          <FilterGroup label="Provenance">
            {PROVENANCE_OPTIONS.map((opt) => (
              <FilterChip
                key={opt}
                label={opt}
                selected={filters.provenance === opt}
                // Provenance chips are labels ABOUT the data, but the chips themselves are
                // interactive controls — selection reads as accent like every other filter chip;
                // the reserved provenance colors appear only on the data badges in the grid.
                onClick={() => setProvenanceParam(opt === "ALL" ? null : opt)}
              />
            ))}
          </FilterGroup>

          <FilterGroup label="Ticker">
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="e.g. LMT"
              aria-label="Filter by ticker"
              className="tp-numeric"
              style={{
                width: layout.x8 * 3,
                height: shape.rowHeightTicker,
                boxSizing: "border-box",
                border: `1px solid ${colors.border}`,
                borderRadius: shape.radiusButton,
                background: colors.bgSurface,
                color: colors.textPrimary,
                fontSize: typeTokens.scale.body,
                padding: `0 ${layout.x2}px`,
                outline: "none",
                textTransform: "uppercase",
              }}
            />
            <FilterChip
              label="Unresolved only"
              selected={unresolvedOnly}
              onClick={() => setUnresolvedOnly((v) => !v)}
            />
          </FilterGroup>

          <FilterGroup label="Date">
            <DateInput value={from} onChange={setFrom} ariaLabel="From date" />
            <span style={{ color: colors.textMuted, fontSize: typeTokens.scale.body }}>–</span>
            <DateInput value={to} onChange={setTo} ariaLabel="To date" />
          </FilterGroup>

          <span style={{ flex: 1 }} />

          <span
            className="tp-numeric"
            style={{ color: colors.textSecondary, fontSize: typeTokens.scale.meta }}
          >
            {rows.length} of latest {signals.data?.length ?? 0}
          </span>
          {canLoadMore && (
            <FilterChip
              label={`Load more (${WINDOW_STEPS.find((s) => s > windowLimit) ?? windowLimit})`}
              selected={false}
              onClick={() =>
                setWindowLimit((limit) => WINDOW_STEPS.find((s) => s > limit) ?? limit)
              }
            />
          )}
          {hasActiveFilters(filters) && (
            <FilterChip label="Clear filters" selected={false} onClick={clearFilters} />
          )}
        </div>

        {/* ---- Grid --------------------------------------------------------------------------- */}
        <div style={{ flex: 1, minHeight: 0 }}>
          {initialLoading ? (
            <SkeletonRows rows={3} />
          ) : signals.error ? (
            <ShellEmptyState message={`Could not load signals: ${signals.error}`} />
          ) : rows.length === 0 ? (
            <ShellEmptyState
              message={
                hasActiveFilters(filters)
                  ? "No signals match these filters."
                  : "No signals yet — the collector is gathering and classifying real data."
              }
              action={
                hasActiveFilters(filters) ? (
                  <FilterChip label="Clear filters" selected={false} onClick={clearFilters} />
                ) : undefined
              }
            />
          ) : (
            <div className="ag-theme-tp" style={{ height: "100%", width: "100%" }}>
              <AgGridReact<SignalRow>
                rowData={rows}
                columnDefs={columns}
                defaultColDef={{ sortable: true, resizable: true, flex: 1 }}
                getRowId={(p) => p.data.signal_id}
                rowHeight={shape.rowHeightDense}
                headerHeight={shape.rowHeightDense}
                onRowClicked={(e) => e.data && openChainForRow(e.data)}
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

function TimestampCell(params: ICellRendererParams<SignalRow, string>) {
  const v = params.value;
  if (!v) return null;
  return (
    <span className="tp-numeric" title={formatTimeUtc(v)} style={{ color: colors.textSecondary }}>
      {formatTime(v)}
    </span>
  );
}

function SourceCell(params: ICellRendererParams<SignalRow, SignalSourceType>) {
  const v = params.value;
  if (!v) return null;
  return (
    <span
      style={{
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusBadge,
        padding: `0 ${layout.x1}px`,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 500,
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        color: colors.textSecondary,
      }}
    >
      {v.replace(/_/g, " ")}
    </span>
  );
}

// §3.1 provenance badges: REAL = unmarked (blank cell — the default state), BACKFILLED = --chain
// violet OUTLINE badge, SYNTHETIC = --synthetic magenta FILLED badge, unknown (outside the
// events-window join) = muted em-dash.
function ProvenanceCell(params: ICellRendererParams<SignalRow>) {
  const provenance = params.data?.resolvedProvenance ?? null;
  if (provenance === "REAL") return null;
  if (provenance === null) {
    return (
      <span title="Provenance unknown — signal.received event outside the loaded events window" style={{ color: colors.textMuted }}>
        —
      </span>
    );
  }
  const isSynthetic = provenance === "SYNTHETIC";
  const tone = isSynthetic ? colors.synthetic : colors.chain;
  return (
    <span
      style={{
        border: `1px solid ${tone}`,
        borderRadius: shape.radiusBadge,
        padding: `0 ${layout.x1}px`,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
        letterSpacing: "0.04em",
        color: isSynthetic ? colors.bgBase : tone,
        background: isSynthetic ? tone : "transparent",
      }}
    >
      {provenance}
    </span>
  );
}

function TickerCell(params: ICellRendererParams<SignalRow, string | null>) {
  const v = params.value;
  if (v == null || v === "") {
    return (
      <span title="Unresolved entity — asset name never matched a ticker" style={{ color: colors.textMuted }}>
        —
      </span>
    );
  }
  return <span className="tp-numeric" style={{ fontWeight: 600, color: colors.textPrimary }}>{v}</span>;
}

// Direction: the signal's bullish/bearish read as a signed, fixed-decimal figure — the one
// place §3.1's side/direction semantics sanction --gain/--loss on this grid.
function DirectionCell(params: ICellRendererParams<SignalRow, number>) {
  const v = params.value;
  if (v == null || Number.isNaN(v)) return <span style={{ color: colors.textMuted }}>—</span>;
  const tone = v > 0 ? colors.gain : v < 0 ? colors.loss : colors.textSecondary;
  return (
    <span className="tp-numeric" style={{ color: tone, fontWeight: 600 }}>
      {v > 0 ? "+" : ""}
      {formatNumber(v, 2)}
    </span>
  );
}

const SIGNAL_COLUMNS: ColDef<SignalRow>[] = [
  {
    headerName: "Disclosed",
    field: "occurred_at",
    cellRenderer: TimestampCell,
    minWidth: 150,
    sort: "desc",
  },
  { headerName: "Source", field: "source_type", cellRenderer: SourceCell, minWidth: 130 },
  {
    headerName: "Provenance",
    colId: "provenance",
    valueGetter: (p) => p.data?.resolvedProvenance ?? null,
    cellRenderer: ProvenanceCell,
    minWidth: 110,
  },
  {
    headerName: "Filer",
    field: "politician_id",
    valueFormatter: (p) => (p.value as string | null) ?? "—",
    minWidth: 140,
  },
  { headerName: "Ticker", field: "ticker", cellRenderer: TickerCell, maxWidth: 110 },
  {
    headerName: "Direction",
    field: "sentiment",
    cellRenderer: DirectionCell,
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    maxWidth: 110,
  },
  {
    headerName: "Weight",
    field: "politician_weight",
    valueFormatter: (p) => formatNumber(p.value as number, 2),
    cellClass: RIGHT_ALIGNED,
    headerClass: RIGHT_ALIGNED,
    maxWidth: 100,
  },
];

// ---- Filter-bar chrome ----------------------------------------------------------------------------

function FilterGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: layout.x1 }}>
      <span
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.meta,
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: colors.textMuted,
          marginRight: layout.x1,
        }}
      >
        {label}
      </span>
      {children}
    </div>
  );
}

function FilterChip({
  label,
  selected,
  onClick,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      style={{
        cursor: "pointer",
        height: shape.rowHeightTicker,
        border: `1px solid ${selected ? colors.accent : colors.border}`,
        borderRadius: shape.radiusButton,
        background: selected ? withAlpha(colors.accent, 0.12) : "transparent",
        color: selected ? colors.accent : colors.textSecondary,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.body,
        padding: `0 ${layout.x2}px`,
        transition: "background-color var(--duration-overlay) var(--ease-out), border-color var(--duration-overlay) var(--ease-out)",
      }}
    >
      {label}
    </button>
  );
}

function DateInput({
  value,
  onChange,
  ariaLabel,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
  ariaLabel: string;
}) {
  return (
    <input
      type="date"
      value={value ?? ""}
      aria-label={ariaLabel}
      onChange={(e) => onChange(e.target.value || null)}
      className="tp-numeric"
      style={{
        height: shape.rowHeightTicker,
        boxSizing: "border-box",
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusButton,
        background: colors.bgSurface,
        color: colors.textPrimary,
        fontSize: typeTokens.scale.body,
        padding: `0 ${layout.x1}px`,
        outline: "none",
        colorScheme: "dark",
      }}
    />
  );
}
