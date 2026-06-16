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
import { Button, Card, H3, Spinner, StackLayout, Text } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { api } from "../../lib/api";
import { formatNumber, formatTime } from "../../lib/format";
import type { PredictedItem, ProposalItem, SignalFeedItem } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

interface ProposalsProps {
  refreshKey: number;
}

export function Proposals({ refreshKey }: ProposalsProps) {
  const proposals = useLoader(() => api.proposals(), [refreshKey]);
  const signals = useLoader(() => api.signalFeed(100), [refreshKey]);

  // Local bump so a buy refreshes this screen immediately, without waiting for the live event
  // stream to round-trip (the stream also bumps the app refreshKey, which reloads us again).
  const [localKey, setLocalKey] = useState(0);
  // In-flight buys, keyed by symbol ("*" = the blanket "Buy all"). Drives the button busy state.
  const [buying, setBuying] = useState<Set<string>>(new Set());
  const [buyError, setBuyError] = useState<string | null>(null);

  const reloadAfterBuy = (proposals as { reload: () => void }).reload;

  async function buy(symbols: string[], key: string): Promise<void> {
    setBuyError(null);
    setBuying((prev) => new Set(prev).add(key));
    try {
      await api.executeProposals(symbols);
      // Refresh both the proposals and the signal feed view.
      reloadAfterBuy();
      setLocalKey((k) => k + 1);
    } catch (err) {
      setBuyError(err instanceof Error ? err.message : String(err));
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

  const anyBuying = buying.size > 0;

  return (
    <StackLayout gap={2}>
      {/* ---- Section 1: actionable proposals --------------------------------------------- */}
      <Card>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            justifyContent: "space-between",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <H3 style={{ margin: 0 }}>Proposals</H3>
            <ModeBadge mode={mode} />
          </div>
          {proposalRows.length > 0 && mode === "manual" && (
            <Button
              variant="cta"
              disabled={anyBuying}
              onClick={() => void buy([], "*")}
              style={{ height: 30 }}
            >
              {buying.has("*") ? "Buying…" : `Buy all (${proposalRows.length})`}
            </Button>
          )}
        </div>

        <Text color="secondary" styleAs="label">
          {mode === "manual"
            ? "Manual: review & approve — nothing trades until you click Buy."
            : "Auto: trades placed automatically — these are the upcoming proposals."}
        </Text>

        {buyError && (
          <Text color="error" style={{ display: "block", marginTop: 8 }}>
            {buyError}
          </Text>
        )}

        <div style={{ marginTop: 12 }}>
          {proposals.loading && !proposals.data ? (
            <Spinner />
          ) : proposals.error ? (
            <Text color="error">Could not load proposals: {proposals.error}</Text>
          ) : proposalRows.length === 0 ? (
            <Text color="secondary">
              No actionable proposals yet — the daily cycle has nothing to trade right now.
            </Text>
          ) : (
            <DataGrid<ProposalItem>
              rows={proposalRows}
              columns={proposalColumns(mode, buying, anyBuying, (sym) => void buy([sym], sym))}
              getRowId={(r) => r.symbol}
              height={Math.min(120 + proposalRows.length * 44, 460)}
            />
          )}
        </div>
      </Card>

      {/* ---- Section 2: predicted / incoming --------------------------------------------- */}
      <Card>
        <H3 style={{ marginTop: 0 }}>Predicted / incoming</H3>
        <Text color="secondary" styleAs="label">
          Watch-tier candidates the cycle expects to act on next — likely upcoming buys (foresight
          between auto-trades). Read-only.
        </Text>
        <div style={{ marginTop: 12 }}>
          {proposals.loading && !proposals.data ? (
            <Spinner />
          ) : predictedRows.length === 0 ? (
            <Text color="secondary">
              Nothing incoming yet — no candidates near the trade threshold.
            </Text>
          ) : (
            <DataGrid<PredictedItem>
              rows={predictedRows}
              columns={predictedColumns}
              getRowId={(r) => r.symbol}
              height={Math.min(120 + predictedRows.length * 44, 400)}
            />
          )}
        </div>
      </Card>

      {/* ---- Section 3: classified signal feed ------------------------------------------- */}
      <Card>
        <H3 style={{ marginTop: 0 }}>Classified signal feed</H3>
        <Text color="secondary" styleAs="label">
          Most recent ingested + classified signals.
        </Text>
        <div style={{ marginTop: 12 }}>
          {signals.loading && !signals.data ? (
            <Spinner />
          ) : signals.error ? (
            <Text color="error">Could not load signals: {signals.error}</Text>
          ) : feed.length === 0 ? (
            <Text color="secondary">No real data yet — collecting.</Text>
          ) : (
            <DataGrid<SignalFeedItem>
              rows={feed}
              columns={signalColumns}
              getRowId={(r) => r.signal_id}
              height={420}
            />
          )}
        </div>
      </Card>
    </StackLayout>
  );
}

// ---- Cycle mode badge (mirrors the Markets FeedBadge / App LIVE-PAPER badge) ----------------

function ModeBadge({ mode }: { mode: "auto" | "manual" }) {
  const isAuto = mode === "auto";
  return (
    <span
      title={
        isAuto
          ? "Auto mode — trades are placed automatically by the cycle"
          : "Manual mode — proposals wait for your approval"
      }
      style={{
        padding: "2px 10px",
        borderRadius: 12,
        fontSize: 12,
        fontWeight: 700,
        whiteSpace: "nowrap",
        background: isAuto ? "rgba(239,68,68,0.18)" : "rgba(76,154,255,0.18)",
        color: isAuto ? "#f87171" : "#7cb3ff",
        border: `1px solid ${isAuto ? "#ef4444" : "#4c9aff"}`,
      }}
    >
      {isAuto ? "AUTO" : "MANUAL"}
    </span>
  );
}

// ---- Column definitions ---------------------------------------------------------------------

function proposalColumns(
  mode: "auto" | "manual",
  buying: Set<string>,
  anyBuying: boolean,
  onBuy: (symbol: string) => void,
): ColDef<ProposalItem>[] {
  const BuyCell = (params: ICellRendererParams<ProposalItem>) => {
    const row = params.data;
    if (!row) return null;
    const inFlight = buying.has(row.symbol);
    return (
      <Button
        variant="primary"
        disabled={anyBuying}
        onClick={() => onBuy(row.symbol)}
        style={{ height: 28, padding: "0 14px" }}
      >
        {inFlight ? "Buying…" : "Buy"}
      </Button>
    );
  };

  const columns: ColDef<ProposalItem>[] = [
    { headerName: "Symbol", field: "symbol", maxWidth: 120 },
    { headerName: "Side", field: "side", maxWidth: 100 },
    {
      headerName: "Quantity",
      field: "quantity",
      maxWidth: 130,
      valueFormatter: (p) => formatNumber(p.value as string, 4),
    },
    {
      headerName: "Conviction",
      field: "conviction_score",
      maxWidth: 130,
      sort: "desc",
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
  { headerName: "Symbol", field: "symbol", maxWidth: 120 },
  {
    headerName: "Conviction",
    field: "conviction_score",
    maxWidth: 140,
    sort: "desc",
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

const signalColumns: ColDef<SignalFeedItem>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    valueFormatter: (p) => formatTime(p.value as string),
  },
  { headerName: "Ticker", field: "ticker", maxWidth: 110 },
  { headerName: "Source", field: "source_type" },
  {
    headerName: "Sentiment",
    field: "sentiment",
    maxWidth: 120,
    valueFormatter: (p) => formatNumber(p.value as number, 2),
  },
  {
    // Optional on older backends (the feed schema may omit it) — render a neutral "—" then.
    headerName: "Provenance",
    field: "provenance",
    maxWidth: 130,
    valueFormatter: (p) => (p.value as string | undefined) ?? "—",
  },
];
