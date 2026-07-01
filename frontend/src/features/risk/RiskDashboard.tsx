// Risk dashboard (FR-UI-6): risk verdicts, reconciliation drift, and manual controls.

import type { ColDef, ICellRendererParams } from "ag-grid-community";
import { Button, StackLayout } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { DataGrid } from "../../components/DataGrid";
import { InfoButton } from "../../components/InfoButton";
import { useToast } from "../../components/Toast";
import {
  Badge,
  EmptyState,
  LoadingState,
  MetricTile,
  Panel,
  SectionHeader,
} from "../../components/ui";
import { Backtest } from "../backtest/Backtest";
import { ApiError, api } from "../../lib/api";
import { formatTime } from "../../lib/format";
import { riskRows } from "../../lib/projections";
import type { RiskRow } from "../../lib/projections";
import {
  color,
  numeric,
  pillStyle,
  radius,
  shadow,
  space,
  text,
  transitions,
} from "../../lib/tokens";
import type { Drift, IngestResult, LiveReadiness, ReconcileReport, Sleeve } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";

// Map a risk verdict string to a semantic pill tone. Approvals read green, rejections red,
// partial fills amber, anything unknown stays muted.
type VerdictTone = "ok" | "danger" | "warn" | "muted";
function verdictTone(verdict: string): VerdictTone {
  const v = verdict.toLowerCase();
  if (v.includes("approve") || v.includes("accept") || v.includes("pass")) return "ok";
  if (v.includes("reject") || v.includes("deny") || v.includes("block") || v.includes("fail"))
    return "danger";
  if (v.includes("partial") || v.includes("reduce") || v.includes("scale")) return "warn";
  return "muted";
}

const rightNumericClass = "ag-right-aligned-cell";

const riskColumns: ColDef<RiskRow>[] = [
  {
    headerName: "Time",
    field: "occurred_at",
    valueFormatter: (p) => formatTime(p.value as string),
    maxWidth: 190,
  },
  {
    headerName: "Verdict",
    field: "verdict",
    maxWidth: 150,
    cellRenderer: (p: ICellRendererParams<RiskRow, string>) => {
      const verdict = p.value ?? "";
      if (!verdict) return null;
      return <span style={{ ...pillStyle(verdictTone(verdict)) }}>{verdict.toUpperCase()}</span>;
    },
  },
  {
    headerName: "Requested",
    field: "requested_quantity",
    maxWidth: 140,
    type: "rightAlignedColumn",
    cellClass: rightNumericClass,
  },
  {
    headerName: "Approved",
    field: "approved_quantity",
    maxWidth: 140,
    type: "rightAlignedColumn",
    cellClass: rightNumericClass,
  },
  { headerName: "Reason", field: "reason", flex: 2 },
];

const driftColumns: ColDef<Drift>[] = [
  {
    headerName: "Kind",
    field: "kind",
    cellRenderer: (p: ICellRendererParams<Drift, string>) =>
      p.value ? <span style={{ ...pillStyle("warn") }}>{p.value.toUpperCase()}</span> : null,
    maxWidth: 160,
  },
  { headerName: "Key", field: "key" },
  { headerName: "Local", field: "local", type: "rightAlignedColumn", cellClass: rightNumericClass },
  {
    headerName: "Broker",
    field: "broker",
    type: "rightAlignedColumn",
    cellClass: rightNumericClass,
  },
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
  const verdictRows = useMemo(() => riskRows(events.data ?? []), [events.data]);

  // Verdict tallies power the summary tiles at the top of the screen.
  const tallies = useMemo(() => {
    let approved = 0;
    let rejected = 0;
    for (const row of verdictRows) {
      const tone = verdictTone(row.verdict);
      if (tone === "ok") approved += 1;
      else if (tone === "danger") rejected += 1;
    }
    return { approved, rejected, total: verdictRows.length };
  }, [verdictRows]);

  const liveCount = sleeves.filter((s) => s.mode === "live").length;

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
      {/* Summary tiles — verdict tallies + live posture at a glance. */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: space.lg,
        }}
      >
        <MetricTile label="Verdicts" value={tallies.total} hint="Evaluated signals in view" />
        <MetricTile label="Approved" value={tallies.approved} tone="ok" />
        <MetricTile label="Rejected" value={tallies.rejected} tone="danger" />
        <MetricTile
          label="Live Sleeves"
          value={liveCount}
          tone={liveCount > 0 ? "danger" : "neutral"}
          hint={liveCount > 0 ? "Trading real money" : "All on paper"}
        />
      </div>

      <Panel>
        <SectionHeader
          eyebrow="Deployment"
          title="Live trading"
          caption="A sleeve can only go live after passing the readiness preflight (§11). Live trades real money through Alpaca."
          actions={
            liveCount > 0 ? (
              <Badge tone="danger" dot>
                {liveCount} LIVE
              </Badge>
            ) : (
              <Badge tone="muted">ALL PAPER</Badge>
            )
          }
        />
        <StackLayout gap={1}>
          {sleeves.length > 0 ? (
            sleeves.map((s) => <LiveTradingRow key={s.id} sleeve={s} />)
          ) : (
            <EmptyState message="No sleeves available to promote to live." />
          )}
        </StackLayout>
      </Panel>

      <Panel>
        <SectionHeader
          eyebrow="Operations"
          title="Controls"
          caption="Manual, read-only diagnostics — reconcile against the broker or run a single ingest cycle."
        />
        <div
          style={{
            display: "flex",
            gap: space.md,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: space.xs }}>
            <Button disabled={busy} onClick={() => void runReconcile()}>
              Run reconciliation
            </Button>
            <InfoButton title="Run reconciliation">
              Compares the platform's recorded positions and open orders against the broker's actual
              state and reports any drift (mismatched quantities or order statuses). Read-only — it
              detects discrepancies, it never places or cancels anything.
            </InfoButton>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: space.xs }}>
            <Button disabled={busy} onClick={() => void runIngest()}>
              Run ingest cycle
            </Button>
            <InfoButton title="Run ingest cycle">
              Polls every enabled data source once, classifies new political signals into the event
              store, and runs them through the strategy + risk evaluation. In manual mode it records
              decisions but places NO orders; in auto mode it would place the proposed trades.
            </InfoButton>
          </div>
          {message && <span style={{ ...text.caption, color: color.muted }}>{message}</span>}
        </div>
        {ingestResult && (
          <div
            style={{
              marginTop: space.md,
              display: "flex",
              flexWrap: "wrap",
              gap: space.sm,
              alignItems: "center",
            }}
          >
            <Badge tone="accent">{ingestResult.signals_ingested} SIGNALS</Badge>
            <Badge tone="muted">{ingestResult.orders.length} ORDERS</Badge>
            {ingestResult.failed_sources.length > 0 && (
              <Badge tone="danger">FAILED: {ingestResult.failed_sources.join(", ")}</Badge>
            )}
          </div>
        )}
      </Panel>

      {drifts.length > 0 && (
        <Panel>
          <SectionHeader
            eyebrow="Reconciliation"
            title="Reconciliation drift"
            caption="Positions or orders where local state diverges from the broker."
            actions={<Badge tone="danger" dot>{`${drifts.length} DRIFT`}</Badge>}
          />
          <DataGrid<Drift>
            rows={drifts}
            columns={driftColumns}
            getRowId={(r) => `${r.kind}|${r.key}`}
            height={200}
          />
        </Panel>
      )}

      <Panel>
        <SectionHeader
          eyebrow="Audit"
          title="Risk verdicts"
          caption="Every signal that passed through risk evaluation, newest first."
          actions={
            verdictRows.length > 0 ? (
              <Badge tone="muted">{`${verdictRows.length} ROWS`}</Badge>
            ) : undefined
          }
        />
        {events.loading && !events.data ? (
          <LoadingState label="Loading risk verdicts…" />
        ) : events.error ? (
          <EmptyState title="Couldn't load risk verdicts" message={events.error} />
        ) : verdictRows.length > 0 ? (
          <DataGrid<RiskRow>
            rows={verdictRows}
            columns={riskColumns}
            getRowId={(r) => `${r.decision_id}-${r.occurred_at}`}
            height={420}
          />
        ) : (
          <EmptyState message="No risk verdicts yet — they appear here as signals are evaluated." />
        )}
      </Panel>

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

  const passedCount = readiness ? readiness.checks.filter((c) => c.passed).length : 0;

  return (
    <div
      style={{
        border: `1px solid ${isLive ? color.danger : color.hairline}`,
        borderRadius: radius.lg,
        padding: space.lg,
        background: color.panelSoft,
        boxShadow: isLive ? shadow.glowDanger : "none",
        display: "flex",
        flexDirection: "column",
        gap: space.md,
        transition: transitions("border-color", "box-shadow"),
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: space.md }}>
        <span style={{ ...text.h3, ...numeric(), color: color.text }}>{sleeve.id}</span>
        <Badge tone={isLive ? "danger" : "accent"} dot={isLive}>
          {isLive ? "LIVE" : "PAPER"}
        </Badge>
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
      {error && <span style={{ ...text.caption, color: color.dangerText }}>{error}</span>}
      {readiness && (
        <div style={{ display: "flex", flexDirection: "column", gap: space.sm }}>
          <div style={{ display: "flex", alignItems: "center", gap: space.sm }}>
            <span style={{ ...eyebrowInline }}>Preflight</span>
            <Badge tone={readiness.ready ? "ok" : "danger"} dot>
              {`${passedCount}/${readiness.checks.length} PASSED`}
            </Badge>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: space.xxs }}>
            {readiness.checks.map((c) => (
              <div
                key={c.name}
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: space.sm,
                  ...text.caption,
                }}
              >
                <span
                  style={{
                    ...numeric(),
                    color: c.passed ? color.ok : color.dangerText,
                    fontWeight: 700,
                    width: 14,
                    flexShrink: 0,
                  }}
                >
                  {c.passed ? "✓" : "✗"}
                </span>
                <span style={{ ...text.label, color: color.textDim, flexShrink: 0 }}>{c.name}</span>
                <span style={{ color: color.muted }}>— {c.detail}</span>
              </div>
            ))}
          </div>
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

const eyebrowInline = {
  ...text.micro,
  textTransform: "uppercase" as const,
  color: color.muted,
};
