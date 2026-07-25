// System `/system` (UI Phase D, docs/UI-DESIGN.md §5.5) — the operator's platform-health screen,
// rebuilt from the Phase-A-only chrome onto the §3 token system. Four panes:
//   Risk           — configured-vs-ceiling cap bars, a drawdown gauge, kill-switch history (from
//                    events), and the risk-gated live-promotion control (per-sleeve preflight).
//   Reconciliation — last run + drift (from the event log), and a Run reconcile button.
//   Jobs           — the scheduler loops with interval / last-run / next-run + order-sync state.
//   Data sources   — the 13 ingest adapters as a READ-ONLY mirror of config (enabled + reliability
//                    prior); no toggles that could drift from env truth.
//
// Every value comes from §3 tokens (zero hardcoded hex/px/ms). Data: GET /system/status (risk caps,
// jobs, data sources) + GET /events (kill-switch history, reconciliation, drawdown from the pool
// equity curve). "Run reconcile" writes via POST /reconcile; the live event stream then refreshes
// the pane. This is a read/monitor screen — no focused-row overlays, so it needs no `active` gate.

import { useCallback, useMemo, useState } from "react";
import { useToast } from "../../components/Toast";
import { api } from "../../lib/api";
import { formatTime, formatTimeUtc } from "../../lib/format";
import { maxDrawdownPct, poolEquitySeries } from "../../lib/projections";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../../lib/tokens";
import type { DataSourceStatus, JobStatus, RiskCap, Sleeve } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";
import { FeedWarningBanner, SkeletonRows } from "../../shell/states";
import { LivePromotion, SystemButton } from "./LivePromotion";
import { killSwitchHistory, reconciliationSummary } from "./systemData";

interface SystemProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

export function System({ refreshKey, sleeves }: SystemProps) {
  const status = useLoader(() => api.systemStatus(), [refreshKey]);
  const events = useLoader(() => api.events(1000), [refreshKey]);

  const eventList = useMemo(() => events.data ?? [], [events.data]);
  const killHistory = useMemo(() => killSwitchHistory(eventList), [eventList]);
  const recon = useMemo(() => reconciliationSummary(eventList), [eventList]);
  const currentDrawdown = useMemo(
    () => maxDrawdownPct(poolEquitySeries(eventList)),
    [eventList],
  );

  const risk = status.data?.risk;
  const jobs = status.data?.jobs ?? [];
  const dataSources = status.data?.data_sources ?? [];
  const drawdownCap = risk?.caps.find((c) => c.name === "drawdown");

  const loading = status.loading && !status.data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x4 }}>
      <h1
        style={{
          margin: 0,
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.screenTitle,
          fontWeight: 600,
          color: colors.textPrimary,
        }}
      >
        System
      </h1>

      {loading ? (
        <SkeletonRows rows={4} rowHeight={shape.rowHeightDense * 4} />
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(460px, 1fr))",
            gap: layout.x4,
            alignItems: "start",
          }}
        >
          {/* -------- Risk pane -------- */}
          <Pane title="Risk" caption="Configured caps vs hardened live ceilings">
            {risk?.enabled ? (
              <>
                <div style={{ display: "flex", flexDirection: "column", gap: layout.x3 }}>
                  {risk.caps.map((cap) => (
                    <CapBar key={cap.name} cap={cap} />
                  ))}
                </div>
                <DrawdownGauge current={currentDrawdown} capPct={drawdownCap?.configured} />
              </>
            ) : (
              <FeedWarningBanner feed="Risk caps" configFlag="MAX_TRADE_PCT_OF_SLEEVE" />
            )}

            <Subhead>Kill switch</Subhead>
            <KillSwitchPanel
              engaged={risk?.kill_switch_engaged ?? false}
              history={killHistory}
            />

            <Subhead>Live promotion</Subhead>
            {sleeves.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: layout.x2 }}>
                {sleeves.map((s) => (
                  <LivePromotion key={s.id} sleeve={s} onChanged={() => status.reload()} />
                ))}
              </div>
            ) : (
              <Muted>No sleeves to promote.</Muted>
            )}
          </Pane>

          {/* -------- Reconciliation pane -------- */}
          <ReconciliationPane
            lastRunAt={recon.lastRunAt}
            driftCount={recon.driftCount}
            checkedPositions={recon.checkedPositions}
            checkedOrders={recon.checkedOrders}
            drifts={recon.drifts}
          />

          {/* -------- Jobs pane -------- */}
          <Pane title="Jobs" caption="Scheduler loops · last / next run · interval">
            <JobsTable jobs={jobs} />
          </Pane>

          {/* -------- Data sources pane -------- */}
          <Pane
            title="Data sources"
            caption={`${dataSources.filter((s) => s.enabled).length} of ${dataSources.length} enabled · read-only mirror of config`}
          >
            <DataSourcesTable sources={dataSources} />
          </Pane>
        </div>
      )}
    </div>
  );
}

// ---- Reconciliation pane (owns the Run-reconcile action) ---------------------------------------

function ReconciliationPane({
  lastRunAt,
  driftCount,
  checkedPositions,
  checkedOrders,
  drifts,
}: {
  lastRunAt: string | null;
  driftCount: number;
  checkedPositions: number;
  checkedOrders: number;
  drifts: { kind: string; key: string; local: string; broker: string }[];
}) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);

  const runReconcile = useCallback(async () => {
    setBusy(true);
    try {
      // Read-only diagnostic (no state mutation), so no §7 confirm. The appended reconciliation
      // events drive the live-stream refresh that repopulates this pane.
      const reports = await api.reconcile();
      const clean = reports.every((r) => r.clean);
      if (clean) toast.success("Reconciliation clean — no drift");
      else toast.info("Drift detected — see below");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [toast]);

  return (
    <Pane
      title="Reconciliation"
      caption="Local projections vs the broker (pool-level)"
      actions={
        <SystemButton disabled={busy} onClick={() => void runReconcile()}>
          Run reconcile
        </SystemButton>
      }
    >
      <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "auto 1fr", gap: layout.x2 }}>
        <Field label="Last run">
          {lastRunAt ? (
            <span className="tp-numeric" title={formatTimeUtc(lastRunAt)}>
              {formatTime(lastRunAt)}
            </span>
          ) : (
            <Muted>never</Muted>
          )}
        </Field>
        <Field label="Checked">
          <span className="tp-numeric">
            {checkedPositions} positions · {checkedOrders} orders
          </span>
        </Field>
        <Field label="Drift">
          <span
            className="tp-numeric"
            style={{ color: driftCount > 0 ? colors.warn : colors.textSecondary, fontWeight: 600 }}
          >
            {driftCount}
          </span>
        </Field>
      </dl>

      {drifts.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
          <Subhead>Drift rows</Subhead>
          {drifts.map((d) => (
            <div
              key={`${d.kind}|${d.key}`}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: layout.x2,
                fontSize: typeTokens.scale.meta,
              }}
            >
              <Tag tone="warn">{d.kind}</Tag>
              <span className="tp-numeric" style={{ color: colors.textPrimary, flex: 1 }}>
                {d.key}
              </span>
              <span className="tp-numeric" style={{ color: colors.textMuted }}>
                local {d.local} · broker {d.broker}
              </span>
            </div>
          ))}
        </div>
      )}
    </Pane>
  );
}

// ---- Risk pane pieces --------------------------------------------------------------------------

// A configured-vs-ceiling bar (§5.5). The track is the hardened ceiling; the fill is the configured
// value's fraction of it. Neutral fill while configured ≤ ceiling; --warn if it EXCEEDS the ceiling
// (a real hardening violation — a sleeve can't go live then). No --gain/--loss: this isn't P&L.
function CapBar({ cap }: { cap: RiskCap }) {
  const configured = Number(cap.configured);
  const ceiling = Number(cap.ceiling);
  const over = ceiling > 0 && configured > ceiling;
  const fraction = ceiling > 0 ? Math.min(configured / ceiling, 1) : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontSize: typeTokens.scale.body, color: colors.textSecondary }}>
          {cap.label}
        </span>
        <span className="tp-numeric" style={{ fontSize: typeTokens.scale.body }}>
          <span style={{ color: over ? colors.warn : colors.textPrimary, fontWeight: 600 }}>
            {configured}
            {cap.unit}
          </span>
          <span style={{ color: colors.textMuted }}>
            {" "}
            / {ceiling}
            {cap.unit} ceiling
          </span>
        </span>
      </div>
      <MeterTrack fraction={fraction} tone={over ? colors.warn : colors.textSecondary} />
    </div>
  );
}

// The drawdown gauge (§5.5): current pool drawdown against the configured max-drawdown cap. Neutral
// until it nears the cap; --warn once current ≥ cap (the circuit-breaker threshold). Current
// drawdown is computed client-side from the pool equity curve (there is no live-drawdown endpoint).
function DrawdownGauge({ current, capPct }: { current: number; capPct?: string }) {
  const cap = capPct !== undefined ? Number(capPct) : null;
  const breach = cap !== null && cap > 0 && current >= cap;
  const fraction = cap !== null && cap > 0 ? Math.min(current / cap, 1) : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x1, marginTop: layout.x2 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontSize: typeTokens.scale.body, color: colors.textSecondary }}>
          Drawdown (pool)
        </span>
        <span className="tp-numeric" style={{ fontSize: typeTokens.scale.body }}>
          <span style={{ color: breach ? colors.warn : colors.textPrimary, fontWeight: 600 }}>
            {current.toFixed(2)}%
          </span>
          {cap !== null && <span style={{ color: colors.textMuted }}> / {cap}% cap</span>}
        </span>
      </div>
      <MeterTrack fraction={fraction} tone={breach ? colors.warn : colors.textSecondary} />
    </div>
  );
}

// A compositor-only meter: a fixed-height track with a scaleX-transformed fill (no width animation,
// §3.5). Non-semantic neutral by default; the caller passes --warn for a threshold breach.
function MeterTrack({ fraction, tone }: { fraction: number; tone: string }) {
  return (
    <div
      style={{
        height: layout.x2,
        borderRadius: shape.radiusBadge,
        background: colors.bgRaised,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          width: "100%",
          transformOrigin: "left",
          transform: `scaleX(${Math.max(0, Math.min(fraction, 1))})`,
          background: tone,
          transition: "transform var(--duration-flash) var(--ease-out)",
        }}
      />
    </div>
  );
}

function KillSwitchPanel({
  engaged,
  history,
}: {
  engaged: boolean;
  history: { time: string; engaged: boolean }[];
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x2 }}>
      <div style={{ display: "flex", alignItems: "center", gap: layout.x2 }}>
        <span style={{ fontSize: typeTokens.scale.body, color: colors.textSecondary }}>
          Current
        </span>
        {/* Kill switch ENGAGED is a sanctioned --loss meaning (§3.1). */}
        <span
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: typeTokens.scale.meta,
            fontWeight: 600,
            letterSpacing: "0.04em",
            color: engaged ? colors.loss : colors.textSecondary,
          }}
        >
          {engaged ? "● ENGAGED" : "○ CLEAR"}
        </span>
      </div>
      {history.length === 0 ? (
        <Muted>No kill-switch changes recorded.</Muted>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
          {history.slice(0, 6).map((h, i) => (
            <div
              key={`${h.time}-${i}`}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: typeTokens.scale.meta,
              }}
            >
              <span
                style={{ color: h.engaged ? colors.loss : colors.textSecondary, fontWeight: 600 }}
              >
                {h.engaged ? "ENGAGED" : "RELEASED"}
              </span>
              <span className="tp-numeric" title={formatTimeUtc(h.time)} style={{ color: colors.textMuted }}>
                {formatTime(h.time)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Jobs pane ---------------------------------------------------------------------------------

function formatInterval(seconds: number): string {
  if (seconds <= 0) return "—";
  if (seconds % 3600 === 0) return `${seconds / 3600}h`;
  if (seconds % 60 === 0) return `${seconds / 60}m`;
  return `${seconds}s`;
}

function JobsTable({ jobs }: { jobs: JobStatus[] }) {
  if (jobs.length === 0) return <Muted>No scheduler loops configured.</Muted>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
      {jobs.map((job) => (
        <div
          key={job.name}
          title={job.description}
          style={{
            display: "grid",
            gridTemplateColumns: "1fr auto auto",
            alignItems: "center",
            gap: layout.x2,
            padding: `${layout.x1}px 0`,
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: layout.x2 }}>
            <Tag tone={job.enabled ? "on" : "off"}>{job.enabled ? "running" : "off"}</Tag>
            <span style={{ color: colors.textPrimary, fontSize: typeTokens.scale.body }}>
              {job.name}
            </span>
          </div>
          <span className="tp-numeric" style={{ color: colors.textSecondary, fontSize: typeTokens.scale.meta }}>
            every {formatInterval(job.interval_seconds)}
          </span>
          <span
            className="tp-numeric"
            style={{ color: colors.textMuted, fontSize: typeTokens.scale.meta, textAlign: "right", minWidth: layout.x8 * 4 }}
            title={
              job.last_run
                ? `last ${formatTimeUtc(job.last_run)}`
                : "not run in this process yet"
            }
          >
            {job.next_run ? `next ${formatTime(job.next_run)}` : job.last_run ? `last ${formatTime(job.last_run)}` : "idle"}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---- Data sources pane -------------------------------------------------------------------------

function DataSourcesTable({ sources }: { sources: DataSourceStatus[] }) {
  if (sources.length === 0) return <Muted>No data sources configured.</Muted>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
      {sources.map((source) => (
        <div
          key={source.source_id}
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr auto",
            alignItems: "center",
            gap: layout.x2,
            padding: `${layout.x1}px 0`,
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <Tag tone={source.enabled ? "on" : "off"}>{source.enabled ? "on" : "off"}</Tag>
          <span
            style={{
              color: source.enabled ? colors.textPrimary : colors.textMuted,
              fontSize: typeTokens.scale.body,
            }}
            title={source.source_id}
          >
            {source.label}
          </span>
          <span
            className="tp-numeric"
            title="Reliability prior (source_reliability term in conviction scoring)"
            style={{ color: colors.textSecondary, fontSize: typeTokens.scale.meta }}
          >
            rel {source.reliability_prior.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---- Shared pane chrome ------------------------------------------------------------------------

function Pane({
  title,
  caption,
  actions,
  children,
}: {
  title: string;
  caption?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        gap: layout.x3,
        padding: layout.x4,
        background: colors.bgSurface,
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusPane,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: layout.x2 }}>
        <h2
          style={{
            margin: 0,
            fontFamily: "var(--font-sans)",
            fontSize: typeTokens.scale.paneTitle,
            fontWeight: 600,
            color: colors.textPrimary,
          }}
        >
          {title}
        </h2>
        {caption && (
          <span style={{ fontSize: typeTokens.scale.meta, color: colors.textMuted }}>
            {caption}
          </span>
        )}
        <span style={{ flex: 1 }} />
        {actions}
      </div>
      {children}
    </section>
  );
}

function Subhead({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        marginTop: layout.x2,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        color: colors.textMuted,
      }}
    >
      {children}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt style={{ fontSize: typeTokens.scale.body, color: colors.textMuted }}>{label}</dt>
      <dd style={{ margin: 0, fontSize: typeTokens.scale.body, color: colors.textPrimary }}>
        {children}
      </dd>
    </>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ fontSize: typeTokens.scale.body, color: colors.textMuted }}>{children}</span>
  );
}

function Tag({ children, tone }: { children: React.ReactNode; tone: "on" | "off" | "warn" }) {
  const color =
    tone === "warn" ? colors.warn : tone === "on" ? colors.textSecondary : colors.textMuted;
  return (
    <span
      style={{
        border: `1px solid ${tone === "off" ? colors.border : withAlpha(color, 0.5)}`,
        borderRadius: shape.radiusBadge,
        padding: `0 ${layout.x1}px`,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
        letterSpacing: "0.04em",
        textTransform: "uppercase",
        color,
      }}
    >
      {children}
    </span>
  );
}
