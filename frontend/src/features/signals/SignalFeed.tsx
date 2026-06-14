// Political signal feed (FR-UI-6): politician, weight, source corroboration, conviction,
// action. The signals table uses AG Grid; the live conviction view uses FINOS Perspective.

import type { ColDef } from "ag-grid-community";
import { Card, H3, StackLayout, Text } from "@salt-ds/core";
import { useMemo } from "react";
import { DataGrid } from "../../components/DataGrid";
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
        <H3>Live conviction</H3>
        <Text color="secondary" styleAs="label">
          Streaming view (FINOS Perspective) — sorted by conviction score
        </Text>
        {convictionRows.length > 0 ? (
          <StreamingTable
            data={convictionRows}
            config={{
              sort: [["score", "desc"]],
              columns: ["ticker", "score", "action", "corroboration", "signals"],
            }}
          />
        ) : (
          <Text color="secondary">No conviction yet within the corroboration window.</Text>
        )}
      </Card>

      <Card>
        <H3>Signal feed</H3>
        <DataGrid<SignalFeedItem>
          rows={feed}
          columns={signalColumns}
          getRowId={(r) => r.signal_id}
          height={420}
        />
      </Card>
    </StackLayout>
  );
}
