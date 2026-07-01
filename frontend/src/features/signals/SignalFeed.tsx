// Political signal feed (FR-UI-6): politician, weight, source corroboration, conviction,
// action. The signals table uses AG Grid; the live conviction view uses FINOS Perspective.

import type { ColDef } from "ag-grid-community";
import { Card, H3, Spinner, StackLayout, Text } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { SignalDetail } from "../../components/SignalDetail";
import { StreamingTable } from "../../components/StreamingTable";
import { api } from "../../lib/api";
import { formatNumber, formatTime } from "../../lib/format";
import type { ConvictionItem, SignalFeedItem } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

const signalColumns: ColDef<SignalFeedItem>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    valueFormatter: (p) => formatTime(p.value as string),
  },
  { headerName: "Ticker", field: "ticker", maxWidth: 110 },
  { headerName: "Source", field: "source_type" },
  { headerName: "Politician", field: "politician_id" },
  {
    headerName: "Sentiment",
    field: "sentiment",
    valueFormatter: (p) => formatNumber(p.value as number, 2),
  },
  {
    headerName: "Weight",
    field: "politician_weight",
    valueFormatter: (p) => formatNumber(p.value as number, 2),
  },
  {
    headerName: "Reliability",
    field: "source_reliability",
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

  return (
    <StackLayout gap={2}>
      <Card>
        <H3 style={{ marginTop: 0 }}>Live conviction</H3>
        <Text color="secondary" styleAs="label">
          Streaming view (FINOS Perspective) — sorted by conviction score
        </Text>
        <div style={{ marginTop: 12 }}>
          {conviction.loading && !conviction.data ? (
            <Spinner />
          ) : conviction.error ? (
            <Text color="error">Could not load conviction: {conviction.error}</Text>
          ) : convictionRows.length === 0 ? (
            <Text color="secondary">No conviction yet within the corroboration window.</Text>
          ) : (
            <StreamingTable
              data={convictionRows}
              config={{
                sort: [["score", "desc"]],
                columns: ["ticker", "score", "action", "corroboration", "signals"],
              }}
            />
          )}
        </div>
      </Card>

      <Card>
        <H3 style={{ marginTop: 0 }}>Signal feed</H3>
        <Text color="secondary" styleAs="label">
          Most recent political signals — newest first. Click a row to see the source text that was
          analyzed.
        </Text>
        <div style={{ marginTop: 12 }}>
          {signals.loading && !signals.data ? (
            <Spinner />
          ) : signals.error ? (
            <Text color="error">Could not load signals: {signals.error}</Text>
          ) : feed.length === 0 ? (
            <Text color="secondary">No signals yet — the feed is collecting real data.</Text>
          ) : (
            <DataGrid<SignalFeedItem>
              rows={feed}
              columns={signalColumns}
              getRowId={(r) => r.signal_id}
              height={420}
              onRowClick={setSelected}
            />
          )}
        </div>
      </Card>

      {selected && <SignalDetail signal={selected} onClose={() => setSelected(null)} />}
    </StackLayout>
  );
}
