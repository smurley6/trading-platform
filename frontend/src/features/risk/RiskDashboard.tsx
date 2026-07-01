// Risk dashboard (FR-UI-6): risk verdicts, reconciliation drift, and manual controls.

import type { ColDef } from "ag-grid-community";
import { Button, Card, H3, Spinner, StackLayout, Text } from "@salt-ds/core";
import { useState } from "react";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { DataGrid } from "../../components/DataGrid";
import { InfoButton } from "../../components/InfoButton";
import { useToast } from "../../components/Toast";
import { Backtest } from "../backtest/Backtest";
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
  const toast = useToast();
  const events = useLoader(() => api.events(1000), [refreshKey]);
  const [reports, setReports] = useState<ReconcileReport[] | null>(null);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const drifts: Drift[] = (reports ?? []).flatMap((r) => r.drifts);
  const verdictRows = riskRows(events.data ?? []);

  async function runReconcile(): Promise<void> {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.reconcile();
      setReports(result);
      const clean = result.every((r) => r.clean);
      setMessage(clean ? "Reconciliation clean — no drift." : "Drift detected.");
      if (clean) {
        toast.success("Reconciliation clean — no drift");
      } else {
        toast.info("Drift detected — see Reconciliation drift");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessage(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  async function runIngest(): Promise<void> {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.ingest();
      setIngestResult(result);
      setMessage("Ingestion cycle complete.");
      toast.success(`Ingested ${result.signals_ingested} signals`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessage(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <StackLayout gap={2}>
      <Card>
        <H3 style={{ marginTop: 0 }}>Live trading</H3>
        <Text color="secondary" styleAs="label">
          A sleeve can only go live after passing the readiness preflight (§11). Live trades real
          money through Alpaca.
        </Text>
        <StackLayout gap={1} style={{ marginTop: 8 }}>
          {sleeves.length > 0 ? (
            sleeves.map((s) => <LiveTradingRow key={s.id} sleeve={s} />)
          ) : (
            <Text color="secondary">No sleeves available to promote to live.</Text>
          )}
        </StackLayout>
      </Card>

      <Card>
        <H3 style={{ marginTop: 0 }}>Controls</H3>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
            <Button disabled={busy} onClick={() => void runReconcile()}>
              Run reconciliation
            </Button>
            <InfoButton title="Run reconciliation">
              Compares the platform's recorded positions and open orders against the broker's actual
              state and reports any drift (mismatched quantities or order statuses). Read-only — it
              detects discrepancies, it never places or cancels anything.
            </InfoButton>
          </div>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
            <Button disabled={busy} onClick={() => void runIngest()}>
              Run ingest cycle
            </Button>
            <InfoButton title="Run ingest cycle">
              Polls every enabled data source once, classifies new political signals into the event
              store, and runs them through the strategy + risk evaluation. In manual mode it records
              decisions but places NO orders; in auto mode it would place the proposed trades.
            </InfoButton>
          </div>
          {message && (
            <Text color="secondary" style={{ alignSelf: "center" }}>
              {message}
            </Text>
          )}
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
          <H3 style={{ marginTop: 0 }}>Reconciliation drift</H3>
          <DataGrid<Drift>
            rows={drifts}
            columns={driftColumns}
            getRowId={(r) => `${r.kind}|${r.key}`}
            height={200}
          />
        </Card>
      )}

      <Card>
        <H3 style={{ marginTop: 0 }}>Risk verdicts</H3>
        {events.loading && !events.data ? (
          <Spinner />
        ) : events.error ? (
          <Text color="error">Could not load risk verdicts: {events.error}</Text>
        ) : verdictRows.length > 0 ? (
          <DataGrid<RiskRow>
            rows={verdictRows}
            columns={riskColumns}
            getRowId={(r) => `${r.decision_id}-${r.occurred_at}`}
            height={420}
          />
        ) : (
          <Text color="secondary">
            No risk verdicts yet — they appear here as signals are evaluated.
          </Text>
        )}
      </Card>

      <Backtest refreshKey={refreshKey} />
    </StackLayout>
  );
}

function LiveTradingRow({ sleeve }: { sleeve: Sleeve }) {
  const toast = useToast();
  const [readiness, setReadiness] = useState<LiveReadiness | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
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
    setBusy(true);
    setError(null);
    try {
      const result = await api.setSleeveMode(sleeve.id, mode, mode === "live");
      setReadiness(result.readiness);
      // The header LIVE indicator updates via the event stream → sleeves reload.
      toast.success(`${sleeve.id} → ${mode.toUpperCase()}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  async function confirmGoLive(): Promise<void> {
    setConfirmOpen(false);
    await setMode("live");
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
        <InfoButton title="Check readiness" align="right">
          Runs the live-trading preflight for this sleeve WITHOUT changing anything: verifies the
          master switch (ALLOW_LIVE), live broker credentials, hardened risk caps, kill-switch
          wiring, clean reconciliation, and that all signals are REAL-provenance. Every check must
          pass before "Go live" is enabled. Read-only — it places no orders.
        </InfoButton>
        {isLive ? (
          <Button disabled={busy} onClick={() => void setMode("paper")}>
            Return to paper
          </Button>
        ) : (
          <Button
            disabled={busy || (readiness !== null && !readiness.ready)}
            onClick={() => setConfirmOpen(true)}
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
      <ConfirmDialog
        open={confirmOpen}
        title="Go live?"
        body={`Enable LIVE trading for ${sleeve.id}? This places orders with REAL money.`}
        tone="danger"
        busy={busy}
        onConfirm={() => void confirmGoLive()}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}
