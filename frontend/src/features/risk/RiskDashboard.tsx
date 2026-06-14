// Risk dashboard (FR-UI-6): risk verdicts, reconciliation drift, and manual controls.

import type { ColDef } from "ag-grid-community";
import { Button, Card, H3, StackLayout, Text } from "@salt-ds/core";
import { useState } from "react";
import { DataGrid } from "../../components/DataGrid";
import { ApiError, api } from "../../lib/api";
import { formatTime } from "../../lib/format";
import { riskRows } from "../../lib/projections";
import type { RiskRow } from "../../lib/projections";
import type { Drift, IngestResult, LiveReadiness, ReconcileReport, Sleeve } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

const riskColumns: ColDef<RiskRow>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    valueFormatter: (p) => formatTime(p.value as string),
  },
  { headerName: "Verdict", field: "verdict", maxWidth: 130 },
  { headerName: "Requested", field: "requested_quantity", maxWidth: 130 },
  { headerName: "Approved", field: "approved_quantity", maxWidth: 130 },
  { headerName: "Reason", field: "reason", flex: 2 },
];

const driftColumns: ColDef<Drift>[] = [
  { headerName: "Kind", field: "kind" },
  { headerName: "Key", field: "key" },
  { headerName: "Local", field: "local" },
  { headerName: "Broker", field: "broker" },
];

interface RiskDashboardProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

export function RiskDashboard({ refreshKey, sleeves }: RiskDashboardProps) {
  const events = useLoader(() => api.events(1000), [refreshKey]);
  const [reports, setReports] = useState<ReconcileReport[] | null>(null);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const drifts: Drift[] = (reports ?? []).flatMap((r) => r.drifts);

  async function runReconcile(): Promise<void> {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.reconcile();
      setReports(result);
      const clean = result.every((r) => r.clean);
      setMessage(clean ? "Reconciliation clean — no drift." : "Drift detected.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runIngest(): Promise<void> {
    setBusy(true);
    setMessage(null);
    try {
      setIngestResult(await api.ingest());
      setMessage("Ingestion cycle complete.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <StackLayout gap={2}>
      <Card>
        <H3>Live trading</H3>
        <Text color="secondary" styleAs="label">
          A sleeve can only go live after passing the readiness preflight (§11). Live trades real
          money through Alpaca.
        </Text>
        <StackLayout gap={1} style={{ marginTop: 8 }}>
          {sleeves.map((s) => (
            <LiveTradingRow key={s.id} sleeve={s} />
          ))}
        </StackLayout>
      </Card>

      <Card>
        <H3>Controls</H3>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <Button disabled={busy} onClick={() => void runReconcile()}>
            Run reconciliation
          </Button>
          <Button disabled={busy} onClick={() => void runIngest()}>
            Run ingest cycle
          </Button>
          {message && <Text color="secondary">{message}</Text>}
        </div>
        {ingestResult && (
          <Text color="secondary" styleAs="label">
            Ingested {ingestResult.signals_ingested} signals · {ingestResult.orders.length} orders
            {ingestResult.failed_sources.length > 0
              ? ` · failed: ${ingestResult.failed_sources.join(", ")}`
              : ""}
          </Text>
        )}
      </Card>

      {drifts.length > 0 && (
        <Card>
          <H3>Reconciliation drift</H3>
          <DataGrid<Drift>
            rows={drifts}
            columns={driftColumns}
            getRowId={(r) => `${r.kind}|${r.key}`}
            height={200}
          />
        </Card>
      )}

      <Card>
        <H3>Risk verdicts</H3>
        <DataGrid<RiskRow>
          rows={riskRows(events.data ?? [])}
          columns={riskColumns}
          getRowId={(r) => `${r.decision_id}-${r.occurred_at}`}
          height={420}
        />
      </Card>
    </StackLayout>
  );
}

function LiveTradingRow({ sleeve }: { sleeve: Sleeve }) {
  const [readiness, setReadiness] = useState<LiveReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isLive = sleeve.mode === "live";

  async function check(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      setReadiness(await api.liveReadiness(sleeve.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function setMode(mode: "live" | "paper"): Promise<void> {
    if (mode === "live" && !window.confirm(`Enable LIVE trading for ${sleeve.id}? Real money.`)) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.setSleeveMode(sleeve.id, mode, mode === "live");
      setReadiness(result.readiness);
      // The header LIVE indicator updates via the event stream → sleeves reload.
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8,
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <strong>{sleeve.id}</strong>
        <span
          style={{
            padding: "1px 8px",
            borderRadius: 10,
            fontSize: 11,
            fontWeight: 700,
            background: isLive ? "rgba(239,68,68,0.2)" : "rgba(76,154,255,0.18)",
            color: isLive ? "#f87171" : "#7cb3ff",
            border: `1px solid ${isLive ? "#ef4444" : "#4c9aff"}`,
          }}
        >
          {isLive ? "LIVE" : "PAPER"}
        </span>
        <div style={{ flex: 1 }} />
        <Button disabled={busy} onClick={() => void check()}>
          Check readiness
        </Button>
        {isLive ? (
          <Button disabled={busy} onClick={() => void setMode("paper")}>
            Return to paper
          </Button>
        ) : (
          <Button
            disabled={busy || (readiness !== null && !readiness.ready)}
            onClick={() => void setMode("live")}
          >
            Go live
          </Button>
        )}
      </div>
      {error && <Text color="error">{error}</Text>}
      {readiness && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {readiness.checks.map((c) => (
            <div key={c.name} style={{ fontSize: 13 }}>
              <span style={{ color: c.passed ? "#4ade80" : "#f87171" }}>
                {c.passed ? "✓" : "✗"}
              </span>{" "}
              <span style={{ fontWeight: 600 }}>{c.name}</span>{" "}
              <span style={{ color: "#9ca3af" }}>— {c.detail}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
