// Political signal feed (FR-UI-6): politician, weight, source corroboration, conviction,
// action. The signals table uses AG Grid; the live conviction view uses FINOS Perspective.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { StackLayout } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { SignalDetail } from "../../components/SignalDetail";
import { StreamingTable } from "../../components/StreamingTable";
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
import { color, numeric, pillStyle, space, text } from "../../lib/tokens";
import type {
  ConvictionAction,
  ConvictionItem,
  SignalFeedItem,
  SignalSourceType,
} from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

// Human-readable labels + pill tones for the categorical signal fields, so the grid reads as
// status chips rather than raw enum strings.
const SOURCE_LABELS: Record<SignalSourceType, string> = {
  statement: "Statement",
  contract: "Contract",
  disclosed_trade: "Disclosed trade",
};

const ACTION_META: Record<
  ConvictionAction,
  { label: string; tone: "accent" | "ok" | "danger" | "warn" | "muted" }
> = {
  trade: { label: "TRADE", tone: "ok" },
  watch: { label: "WATCH", tone: "warn" },
  ignore: { label: "IGNORE", tone: "muted" },
};

// Cell renderer: a small status pill for the source type.
function SourceCell(params: ICellRendererParams<SignalFeedItem, SignalSourceType>) {
  const value = params.value;
  if (value == null) return null;
  return (
    <span style={{ ...pillStyle("accent"), padding: "2px 8px", fontSize: 10 }}>
      {SOURCE_LABELS[value] ?? value}
    </span>
  );
}

// Cell renderer: sentiment reads as a signed, sign-colored figure (positive = bullish).
function SentimentCell(params: ICellRendererParams<SignalFeedItem, number>) {
  const value = params.value;
  if (value == null || Number.isNaN(value)) return null;
  const tone = value > 0 ? color.ok : value < 0 ? color.dangerText : color.textDim;
  const sign = value > 0 ? "+" : "";
  return (
    <span style={{ ...numeric(), fontWeight: 600, color: tone }}>
      {sign}
      {formatNumber(value, 2)}
    </span>
  );
}

const signalColumns: ColDef<SignalFeedItem>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    valueFormatter: (p) => formatTime(p.value as string),
    maxWidth: 130,
  },
  { headerName: "Ticker", field: "ticker", maxWidth: 110 },
  {
    headerName: "Source",
    field: "source_type",
    cellRenderer: SourceCell,
    minWidth: 150,
  },
  { headerName: "Politician", field: "politician_id" },
  {
    headerName: "Sentiment",
    field: "sentiment",
    type: "rightAlignedColumn",
    cellRenderer: SentimentCell,
  },
  {
    headerName: "Weight",
    field: "politician_weight",
    type: "rightAlignedColumn",
    valueFormatter: (p) => formatNumber(p.value as number, 2),
  },
  {
    headerName: "Reliability",
    field: "source_reliability",
    type: "rightAlignedColumn",
    valueFormatter: (p) => formatNumber(p.value as number, 2),
  },
];

interface SignalFeedProps {
  refreshKey: number;
}

export function SignalFeed({ refreshKey }: SignalFeedProps) {
  const signals = useLoader(() => api.signalFeed(300), [refreshKey]);
  const conviction = useLoader(() => api.conviction(), [refreshKey]);
  const [selected, setSelected] = useState<SignalFeedItem | null>(null);

  const feed = useMemo(() => [...(signals.data ?? [])].reverse(), [signals.data]);

  // Corroboration = number of distinct source types backing each ticker (FR-POL-4).
  const corroboration = useMemo(() => {
    const sources = new Map<string, Set<string>>();
    for (const s of signals.data ?? []) {
      const set = sources.get(s.ticker) ?? new Set<string>();
      set.add(s.source_type);
      sources.set(s.ticker, set);
    }
    return sources;
  }, [signals.data]);

  const convictionRows: Record<string, unknown>[] = (conviction.data ?? []).map(
    (c: ConvictionItem) => ({
      ticker: c.ticker,
      score: Math.round(c.score * 10) / 10,
      action: c.action,
      corroboration: corroboration.get(c.ticker)?.size ?? 0,
      signals: c.contributing_signal_ids.length,
    }),
  );

  // Header KPIs — distinct tickers under coverage, tradeable candidates, and the top score.
  const distinctTickers = useMemo(
    () => new Set((signals.data ?? []).map((s) => s.ticker)).size,
    [signals.data],
  );
  const convictionData = conviction.data ?? [];
  const tradeCandidates = convictionData.filter((c) => c.action === "trade").length;
  const topScore = convictionData.reduce((max, c) => Math.max(max, c.score), 0);

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
          label="Signals ingested"
          value={signals.data?.length ?? "—"}
          hint="Trailing feed window"
        />
        <MetricTile label="Tickers covered" value={distinctTickers || "—"} tone="accent" />
        <MetricTile
          label="Trade candidates"
          value={tradeCandidates || "—"}
          tone={tradeCandidates > 0 ? "ok" : "neutral"}
          hint="Action = trade"
        />
        <MetricTile
          label="Top conviction"
          value={convictionData.length ? formatNumber(topScore, 1) : "—"}
          tone={topScore > 0 ? "accent" : "neutral"}
        />
      </div>

      <Panel>
        <SectionHeader
          eyebrow="FINOS Perspective"
          title="Live conviction"
          caption="Streaming view — sorted by conviction score"
          actions={
            <Badge tone="accent" dot>
              LIVE
            </Badge>
          }
        />
        {conviction.loading && !conviction.data ? (
          <LoadingState label="Loading conviction…" />
        ) : conviction.error ? (
          <EmptyState
            title="Couldn't load conviction"
            message={conviction.error}
            style={{ color: color.dangerText }}
          />
        ) : convictionRows.length === 0 ? (
          <EmptyState
            title="No conviction yet"
            message="Nothing has crossed the corroboration window. Scores appear once multiple sources back the same ticker."
          />
        ) : (
          <>
            <div
              style={{ display: "flex", flexWrap: "wrap", gap: space.sm, marginBottom: space.md }}
            >
              {(["trade", "watch", "ignore"] as ConvictionAction[]).map((action) => {
                const count = convictionData.filter((c) => c.action === action).length;
                if (count === 0) return null;
                const meta = ACTION_META[action];
                return (
                  <Badge key={action} tone={meta.tone}>
                    {meta.label}
                    <span style={{ ...numeric(), marginLeft: 6, opacity: 0.85 }}>{count}</span>
                  </Badge>
                );
              })}
            </div>
            <StreamingTable
              data={convictionRows}
              config={{
                sort: [["score", "desc"]],
                columns: ["ticker", "score", "action", "corroboration", "signals"],
              }}
            />
          </>
        )}
      </Panel>

      <Panel>
        <SectionHeader
          eyebrow="Political signals"
          title="Signal feed"
          caption="Most recent political signals, newest first — click a row to inspect the source text."
          actions={
            feed.length > 0 ? (
              <span style={{ ...text.caption, ...numeric(), color: color.muted }}>
                {feed.length} events
              </span>
            ) : undefined
          }
        />
        {signals.loading && !signals.data ? (
          <LoadingState label="Loading signals…" />
        ) : signals.error ? (
          <EmptyState
            title="Couldn't load signals"
            message={signals.error}
            style={{ color: color.dangerText }}
          />
        ) : feed.length === 0 ? (
          <EmptyState
            title="No signals yet"
            message="The feed is collecting real data. New political signals will stream in here as they're ingested."
          />
        ) : (
          <DataGrid<SignalFeedItem>
            rows={feed}
            columns={signalColumns}
            getRowId={(r) => r.signal_id}
            height={420}
            onRowClick={setSelected}
          />
        )}
      </Panel>

      {selected && <SignalDetail signal={selected} onClose={() => setSelected(null)} />}
    </StackLayout>
  );
}
