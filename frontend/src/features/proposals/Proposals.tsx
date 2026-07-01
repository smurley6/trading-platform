// Proposals screen — the human-in-the-loop surface for the daily trading cycle. Three sections:
//
//   1. Proposals (actionable): the trades the cycle found. In MANUAL mode each row is approved by
//      a per-row "Buy" button (or "Buy all" for the blanket set); the cycle mode is shown as a
//      badge with a one-line explainer. In AUTO mode these have already been placed, so the rows
//      preview the next round (foresight between auto-trades).
//   2. Predicted / incoming: read-only watch-tier candidates the cycle expects to act on next.
//   3. Classified signal feed: the most recent ingested + classified signals.
//
// All data comes from GET /proposals (mode + proposals + predicted) and GET /signals/feed. Buys
// POST /proposals/execute and then bump the local refresh; the app-wide live event stream also
// drives refreshes, exactly like the blotter's optimistic cancel. Lists are usually EMPTY at
// first (the DB is freshly collecting real data) — every empty state is an intentional message.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { Button, StackLayout } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { DataGrid } from "../../components/DataGrid";
import { SignalDetail } from "../../components/SignalDetail";
import { useToast } from "../../components/Toast";
import { Badge, EmptyState, Figure, LoadingState, Panel, SectionHeader } from "../../components/ui";
import { api } from "../../lib/api";
import { formatNumber, formatTime } from "../../lib/format";
import { color, eyebrowStyle, numeric, radius, space, text, transitions } from "../../lib/tokens";
import type { PredictedItem, ProposalItem, SignalFeedItem } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

interface ProposalsProps {
  refreshKey: number;
}

// A buy awaiting confirmation: the symbols passed to executeProposals (empty = "all" on the
// backend), the busy key ("*" = blanket), and the human-readable symbol list for the dialog body.
interface PendingBuy {
  symbols: string[];
  key: string;
  displaySymbols: string[];
}

// Pull a short analyzed-text snippet from a signal's raw payload for the evidence drill-down.
const TEXT_KEYS = [
  "text",
  "title",
  "headline",
  "summary",
  "description",
  "body",
  "statement",
  "content",
];

function signalSnippet(sig: SignalFeedItem): string {
  const rp = sig.raw_payload;
  if (!rp) return "—";
  for (const [k, v] of Object.entries(rp)) {
    if (TEXT_KEYS.includes(k.toLowerCase()) && typeof v === "string" && v.trim()) {
      const t = v.trim();
      return t.length > 120 ? `${t.slice(0, 120)}…` : t;
    }
  }
  return "—";
}

export function Proposals({ refreshKey }: ProposalsProps) {
  const toast = useToast();
  const proposals = useLoader(() => api.proposals(), [refreshKey]);
  const signals = useLoader(() => api.signalFeed(300), [refreshKey]);

  // Local bump so a buy refreshes this screen immediately, without waiting for the live event
  // stream to round-trip (the stream also bumps the app refreshKey, which reloads us again).
  const [localKey, setLocalKey] = useState(0);
  // In-flight buys, keyed by symbol ("*" = the blanket "Buy all"). Drives the button busy state.
  const [buying, setBuying] = useState<Set<string>>(new Set());
  // The buy awaiting confirmation in the ConfirmDialog (null = dialog closed).
  const [pendingBuy, setPendingBuy] = useState<PendingBuy | null>(null);
  // Evidence drill-down: the selected proposal and (within it) an expanded contributing signal.
  const [selectedProposal, setSelectedProposal] = useState<ProposalItem | null>(null);
  const [expandedSignal, setExpandedSignal] = useState<SignalFeedItem | null>(null);

  const reloadAfterBuy = (proposals as { reload: () => void }).reload;

  async function buy(symbols: string[], key: string): Promise<void> {
    setBuying((prev) => new Set(prev).add(key));
    try {
      const result = await api.executeProposals(symbols);
      toast.success(`Placed ${result.orders.length} order(s)`);
      // Refresh both the proposals and the signal feed view.
      reloadAfterBuy();
      setLocalKey((k) => k + 1);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBuying((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  }

  const mode = proposals.data?.mode ?? "manual";
  const proposalRows = proposals.data?.proposals ?? [];
  const predictedRows = proposals.data?.predicted ?? [];

  // localKey is part of the dependency chain so a successful buy re-reads the feed too.
  const feed = useMemo(
    () => [...(signals.data ?? [])].reverse(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [signals.data, localKey],
  );

  // Index the loaded feed by signal_id so the evidence panel can resolve a proposal's
  // contributing_signal_ids back to full source records.
  const signalsById = useMemo(() => {
    const map = new Map<string, SignalFeedItem>();
    for (const s of signals.data ?? []) map.set(s.signal_id, s);
    return map;
  }, [signals.data]);

  const anyBuying = buying.size > 0;

  // Open the confirm dialog for a single-row buy. Empty symbols list ("Buy all") is handled below.
  const requestBuy = (pending: PendingBuy) => setPendingBuy(pending);

  const confirmPending = () => {
    if (!pendingBuy) return;
    const { symbols, key } = pendingBuy;
    setPendingBuy(null);
    void buy(symbols, key);
  };

  // Select a proposal for the evidence drill-down; reset any expanded contributing signal.
  const openEvidence = (row: ProposalItem) => {
    setSelectedProposal(row);
    setExpandedSignal(null);
  };

  const proposalsLoading = proposals.loading && !proposals.data;

  return (
    <StackLayout gap={2}>
      {/* ---- Section 1: actionable proposals --------------------------------------------- */}
      <Panel>
        <SectionHeader
          eyebrow="Daily cycle"
          title={
            <span style={{ display: "inline-flex", alignItems: "center", gap: space.md }}>
              Proposals
              <ModeBadge mode={mode} />
            </span>
          }
          caption={
            mode === "manual"
              ? "Manual — review & approve; nothing trades until you click Buy. Click a row to see why."
              : "Auto — trades placed automatically; these preview the upcoming round. Click a row to see why."
          }
          actions={
            proposalRows.length > 0 && mode === "manual" ? (
              <Button
                variant="cta"
                disabled={anyBuying}
                onClick={() =>
                  requestBuy({
                    symbols: [],
                    key: "*",
                    displaySymbols: proposalRows.map((r) => r.symbol),
                  })
                }
                style={{ height: 32 }}
              >
                {buying.has("*") ? "Buying…" : `Buy all (${proposalRows.length})`}
              </Button>
            ) : undefined
          }
        />

        {proposalsLoading ? (
          <LoadingState label="Loading proposals…" />
        ) : proposals.error ? (
          <EmptyState
            title="Couldn’t load proposals"
            message={proposals.error}
            icon={<span aria-hidden>⚠</span>}
          />
        ) : proposalRows.length === 0 ? (
          <EmptyState
            title="No actionable proposals"
            message="The daily cycle has nothing to trade right now. Candidates will appear here as signals accumulate."
            icon={<span aria-hidden>◎</span>}
          />
        ) : (
          <DataGrid<ProposalItem>
            rows={proposalRows}
            columns={proposalColumns(mode, buying, anyBuying, (row) =>
              requestBuy({
                symbols: [row.symbol],
                key: row.symbol,
                displaySymbols: [row.symbol],
              }),
            )}
            getRowId={(r) => r.symbol}
            onRowClick={openEvidence}
            height={Math.min(120 + proposalRows.length * 44, 460)}
          />
        )}
      </Panel>

      {/* ---- Evidence drill-down: "why this proposal?" ----------------------------------- */}
      {selectedProposal && (
        <Panel raised>
          <SectionHeader
            eyebrow="Evidence"
            title={
              <span style={{ display: "inline-flex", alignItems: "center", gap: space.sm }}>
                Why{" "}
                <Figure color={color.accentText} weight={700}>
                  {selectedProposal.symbol}
                </Figure>
                ?
              </span>
            }
            caption={
              <span style={{ display: "inline-flex", alignItems: "center", gap: space.sm }}>
                Conviction
                <Figure color={color.text}>
                  {formatNumber(selectedProposal.conviction_score, 1)}
                </Figure>
              </span>
            }
            actions={
              <button
                type="button"
                onClick={() => {
                  setSelectedProposal(null);
                  setExpandedSignal(null);
                }}
                aria-label="Close evidence"
                style={{
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 28,
                  height: 28,
                  borderRadius: radius.md,
                  background: "transparent",
                  border: `1px solid ${color.hairline}`,
                  color: color.muted,
                  fontSize: 16,
                  lineHeight: 1,
                  transition: transitions("background", "color", "border-color"),
                }}
              >
                ×
              </button>
            }
          />

          <div style={eyebrowStyle()}>Rationale</div>
          <div
            style={{
              marginTop: space.xs,
              padding: space.md,
              borderRadius: radius.md,
              background: color.panelSoft,
              border: `1px solid ${color.hairline}`,
              ...text.body,
              color: color.textDim,
            }}
          >
            {selectedProposal.rationale || "—"}
          </div>

          <div style={{ marginTop: space.lg }}>
            <div style={eyebrowStyle()}>
              Contributing signals ({selectedProposal.contributing_signal_ids.length})
            </div>
            {selectedProposal.contributing_signal_ids.length === 0 ? (
              <div style={{ ...text.body, color: color.muted, marginTop: space.sm }}>
                No contributing signals recorded for this proposal.
              </div>
            ) : (
              <div
                style={{
                  marginTop: space.sm,
                  display: "flex",
                  flexDirection: "column",
                  gap: space.sm,
                }}
              >
                {selectedProposal.contributing_signal_ids.map((id, i) => {
                  const sig = signalsById.get(id);
                  if (!sig) {
                    return (
                      <div
                        key={`${id}-${i}`}
                        style={{
                          padding: `${space.sm}px ${space.md}px`,
                          borderRadius: radius.md,
                          border: `1px solid ${color.hairline}`,
                          background: color.panelInset,
                          color: color.muted,
                          ...text.caption,
                        }}
                      >
                        signal <span style={numeric({ color: color.textDim })}>{id}</span> — not in
                        recent feed
                      </div>
                    );
                  }
                  return (
                    <button
                      key={`${id}-${i}`}
                      type="button"
                      onClick={() => setExpandedSignal(sig)}
                      style={{
                        textAlign: "left",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: space.sm,
                        flexWrap: "wrap",
                        padding: `${space.sm}px ${space.md}px`,
                        borderRadius: radius.md,
                        background: color.panelSoft,
                        border: `1px solid ${color.hairline}`,
                        color: color.textDim,
                        ...text.caption,
                        transition: transitions("background", "border-color"),
                      }}
                    >
                      <Badge tone="accent">{sig.source_type}</Badge>
                      <span style={numeric({ color: color.text, fontWeight: 700 })}>
                        {sig.ticker}
                      </span>
                      <span style={{ color: color.faint }}>·</span>
                      <span style={{ color: color.muted }}>
                        sentiment{" "}
                        <Figure pnl={sig.sentiment}>{formatNumber(sig.sentiment, 2)}</Figure>
                      </span>
                      <span style={{ color: color.faint }}>·</span>
                      <span style={{ color: color.muted, flex: 1, minWidth: 160 }}>
                        {signalSnippet(sig)}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {expandedSignal && (
            <div style={{ marginTop: space.md }}>
              <SignalDetail signal={expandedSignal} onClose={() => setExpandedSignal(null)} />
            </div>
          )}
        </Panel>
      )}

      {/* ---- Section 2: predicted / incoming --------------------------------------------- */}
      <Panel>
        <SectionHeader
          eyebrow="Foresight"
          title="Predicted / incoming"
          caption="Watch-tier candidates the cycle expects to act on next — likely upcoming buys. Read-only."
          actions={
            predictedRows.length > 0 ? (
              <Badge tone="muted">{predictedRows.length} watching</Badge>
            ) : undefined
          }
        />
        {proposalsLoading ? (
          <LoadingState label="Loading candidates…" />
        ) : proposals.error ? (
          <EmptyState
            title="Couldn’t load candidates"
            message={proposals.error}
            icon={<span aria-hidden>⚠</span>}
          />
        ) : predictedRows.length === 0 ? (
          <EmptyState
            title="Nothing incoming"
            message="No candidates near the trade threshold yet."
            icon={<span aria-hidden>◔</span>}
          />
        ) : (
          <DataGrid<PredictedItem>
            rows={predictedRows}
            columns={predictedColumns}
            getRowId={(r) => r.symbol}
            height={Math.min(120 + predictedRows.length * 44, 400)}
          />
        )}
      </Panel>

      {/* ---- Section 3: classified signal feed ------------------------------------------- */}
      <Panel>
        <SectionHeader
          eyebrow="Live feed"
          title="Classified signal feed"
          caption="Most recent ingested + classified signals."
          actions={
            feed.length > 0 ? (
              <Badge tone="accent" dot>
                {feed.length} signals
              </Badge>
            ) : undefined
          }
        />
        {signals.loading && !signals.data ? (
          <LoadingState label="Loading signals…" />
        ) : signals.error ? (
          <EmptyState
            title="Couldn’t load signals"
            message={signals.error}
            icon={<span aria-hidden>⚠</span>}
          />
        ) : feed.length === 0 ? (
          <EmptyState
            title="No signals yet"
            message="No real data yet — the collector is gathering and classifying signals."
            icon={<span aria-hidden>◌</span>}
          />
        ) : (
          <DataGrid<SignalFeedItem>
            rows={feed}
            columns={signalColumns}
            getRowId={(r) => r.signal_id}
            height={420}
          />
        )}
      </Panel>

      {/* ---- Buy confirmation -------------------------------------------------------------- */}
      <ConfirmDialog
        open={pendingBuy !== null}
        title="Place paper orders?"
        body={
          pendingBuy ? (
            <>
              Place {pendingBuy.displaySymbols.length} PAPER order(s) for:{" "}
              <strong>{pendingBuy.displaySymbols.join(", ")}</strong> on the political sleeve.
            </>
          ) : null
        }
        confirmLabel="Place orders"
        busy={pendingBuy ? buying.has(pendingBuy.key) : false}
        onConfirm={confirmPending}
        onCancel={() => setPendingBuy(null)}
      />
    </StackLayout>
  );
}

// ---- Cycle mode badge (mirrors the Markets FeedBadge / App LIVE-PAPER badge) ----------------

function ModeBadge({ mode }: { mode: "auto" | "manual" }) {
  const isAuto = mode === "auto";
  return (
    <Badge
      tone={isAuto ? "danger" : "accent"}
      dot
      title={
        isAuto
          ? "Auto mode — trades are placed automatically by the cycle"
          : "Manual mode — proposals wait for your approval"
      }
    >
      {isAuto ? "AUTO" : "MANUAL"}
    </Badge>
  );
}

// ---- Column definitions ---------------------------------------------------------------------

// A mono/tabular text span for symbol/ticker cells so identifiers align like a terminal.
function SymbolCell(params: ICellRendererParams<unknown, string>) {
  return <span style={numeric({ color: color.text, fontWeight: 700 })}>{params.value ?? "—"}</span>;
}

function proposalColumns(
  mode: "auto" | "manual",
  buying: Set<string>,
  anyBuying: boolean,
  onBuy: (row: ProposalItem) => void,
): ColDef<ProposalItem>[] {
  const BuyCell = (params: ICellRendererParams<ProposalItem>) => {
    const row = params.data;
    if (!row) return null;
    const inFlight = buying.has(row.symbol);
    return (
      <Button
        variant="primary"
        disabled={anyBuying}
        // Stop the row-click (evidence drill-down) from also firing when the Buy button is clicked.
        onClick={(e) => {
          e.stopPropagation();
          onBuy(row);
        }}
        style={{ height: 28, padding: "0 14px" }}
      >
        {inFlight ? "Buying…" : "Buy"}
      </Button>
    );
  };

  const SideCell = (params: ICellRendererParams<ProposalItem, string>) => {
    const side = params.value;
    if (!side) return null;
    return <Badge tone={side === "buy" ? "ok" : "danger"}>{side.toUpperCase()}</Badge>;
  };

  const columns: ColDef<ProposalItem>[] = [
    { headerName: "Symbol", field: "symbol", maxWidth: 120, cellRenderer: SymbolCell },
    { headerName: "Side", field: "side", maxWidth: 100, cellRenderer: SideCell },
    {
      headerName: "Quantity",
      field: "quantity",
      maxWidth: 130,
      type: "rightAlignedColumn",
      valueFormatter: (p) => formatNumber(p.value as string, 4),
    },
    {
      headerName: "Conviction",
      field: "conviction_score",
      maxWidth: 130,
      sort: "desc",
      type: "rightAlignedColumn",
      valueFormatter: (p) => formatNumber(p.value as number, 1),
    },
    { headerName: "Rationale", field: "rationale", flex: 2 },
  ];

  // The Buy action is only meaningful in manual mode (auto rows are previews of placed trades).
  if (mode === "manual") {
    columns.push({
      headerName: "",
      colId: "buy",
      maxWidth: 120,
      sortable: false,
      filter: false,
      cellRenderer: BuyCell,
    });
  }
  return columns;
}

const predictedColumns: ColDef<PredictedItem>[] = [
  { headerName: "Symbol", field: "symbol", maxWidth: 120, cellRenderer: SymbolCell },
  {
    headerName: "Conviction",
    field: "conviction_score",
    maxWidth: 140,
    sort: "desc",
    type: "rightAlignedColumn",
    valueFormatter: (p) => formatNumber(p.value as number, 1),
  },
  {
    headerName: "Corroborating sources",
    field: "distinct_source_types",
    flex: 2,
    valueFormatter: (p) => {
      const sources = p.value as string[] | undefined;
      return sources && sources.length > 0 ? sources.join(", ") : "—";
    },
  },
];

// Sentiment cell: mono/tabular figure colored by sign (positive green, negative red).
function SentimentCell(params: ICellRendererParams<SignalFeedItem, number>) {
  const v = params.value;
  if (v == null) return <span style={{ color: color.muted }}>—</span>;
  return <Figure pnl={v}>{formatNumber(v, 2)}</Figure>;
}

const signalColumns: ColDef<SignalFeedItem>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    type: "rightAlignedColumn",
    valueFormatter: (p) => formatTime(p.value as string),
  },
  { headerName: "Ticker", field: "ticker", maxWidth: 110, cellRenderer: SymbolCell },
  { headerName: "Source", field: "source_type" },
  {
    headerName: "Sentiment",
    field: "sentiment",
    maxWidth: 120,
    type: "rightAlignedColumn",
    cellRenderer: SentimentCell,
  },
  {
    // Optional on older backends (the feed schema may omit it) — render a neutral "—" then.
    headerName: "Provenance",
    field: "provenance",
    maxWidth: 130,
    valueFormatter: (p) => (p.value as string | undefined) ?? "—",
  },
];
