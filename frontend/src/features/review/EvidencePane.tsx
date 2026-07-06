// Review evidence pane (UI Phase C §5.2, right 62%): proposal header + proposed size, the
// conviction decomposition (the REAL terms the existing surface carries — see convictionTerms.ts
// for exactly what is and is not derivable client-side), the SOURCE SIGNALS list, the RISK
// PRE-CHECK strip (GET /risk/precheck — loading/disabled/error states handled explicitly since
// the endpoint may not be deployed yet), the CHAIN summary strip, and the [A]pprove/[R]eject
// footer with the §7 single-confirm keystroke (ConfirmChip).
//
// Purely presentational — Review.tsx owns all state (selection, pre-check fetch, pending
// confirm, overlays) and passes it down. Every visual value is a §3 token.

import type { ReactNode } from "react";
import { formatNumber, formatTime, formatTimeUtc } from "../../lib/format";
import { colors, layout, shape, type as typeTokens, withAlpha } from "../../lib/tokens";
import type { ProposalItem, RiskPrecheckResult } from "../../lib/types";
import { ConfirmChip } from "../../shell/ConfirmChip";
import { SkeletonRows } from "../../shell/states";
import type { ConvictionTerms } from "./convictionTerms";

export type PrecheckState =
  | { state: "loading" }
  | { state: "done"; data: RiskPrecheckResult }
  | { state: "error"; status: number | null; message: string };

export type PendingAction = { type: "approve" | "reject"; symbol: string } | null;

interface EvidencePaneProps {
  proposal: ProposalItem;
  terms: ConvictionTerms;
  precheck: PrecheckState;
  /** Correlation id of the proposal's evidence chain (first contributing signal), if resolvable. */
  chainCorrelationId: string | null;
  pendingAction: PendingAction;
  busy: boolean;
  onRequestApprove: () => void;
  onRequestReject: () => void;
  onConfirmPending: () => void;
  onCancelPending: () => void;
  onOpenChart: () => void;
  onOpenChain: () => void;
  onOpenSource: (url: string) => void;
  /** Ref target for Enter/ArrowRight "focus detail" (§5.2). */
  paneRef: React.RefObject<HTMLDivElement>;
}

export function EvidencePane({
  proposal,
  terms,
  precheck,
  chainCorrelationId,
  pendingAction,
  busy,
  onRequestApprove,
  onRequestReject,
  onConfirmPending,
  onCancelPending,
  onOpenChart,
  onOpenChain,
  onOpenSource,
  paneRef,
}: EvidencePaneProps) {
  const sideColor = proposal.side === "buy" ? colors.gain : colors.loss;
  const notional =
    precheck.state === "done" && precheck.data.enabled
      ? precheck.data.checks.find((c) => c.name === "trade_size")?.projected
      : null;

  return (
    <div
      ref={paneRef}
      tabIndex={-1}
      aria-label={`Evidence for ${proposal.symbol}`}
      style={{
        height: "100%",
        overflowY: "auto",
        padding: layout.x4,
        display: "flex",
        flexDirection: "column",
        gap: layout.x4,
        outline: "none",
      }}
    >
      {/* ---- Header: symbol, side, proposed size, [t chart] ---------------------------------- */}
      <div style={{ display: "flex", alignItems: "baseline", gap: layout.x3 }}>
        <span
          className="tp-numeric"
          style={{ fontSize: typeTokens.scale.screenTitle, fontWeight: 600, color: colors.textPrimary }}
        >
          {proposal.symbol}
        </span>
        <span
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: typeTokens.scale.body,
            fontWeight: 600,
            color: sideColor,
            letterSpacing: "0.04em",
          }}
        >
          {proposal.side.toUpperCase()}
        </span>
        <span style={{ flex: 1 }} />
        <PaneKeyButton label="chart" hotkey="t" onClick={onOpenChart} />
      </div>

      <div
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.default,
          color: colors.textPrimary,
        }}
      >
        Proposed: <span style={{ color: sideColor, fontWeight: 600 }}>{proposal.side.toUpperCase()}</span>{" "}
        <span className="tp-numeric">{formatNumber(proposal.quantity, 4)}</span> sh
        {notional ? (
          <span style={{ color: colors.textSecondary }}>
            {" "}
            (<span className="tp-numeric">{notional}</span> at market, per pre-check)
          </span>
        ) : null}
        <span style={{ color: colors.textSecondary }}>
          {" "}
          · target <span className="tp-numeric">{formatNumber(proposal.target_quantity, 4)}</span> · held{" "}
          <span className="tp-numeric">{formatNumber(proposal.current_quantity, 4)}</span>
        </span>
      </div>

      {/* ---- Conviction decomposition -------------------------------------------------------- */}
      <Section
        title={
          <>
            CONVICTION{" "}
            <span className="tp-numeric" style={{ color: colors.textPrimary }}>
              {formatNumber(proposal.conviction_score, 0)}
            </span>
          </>
        }
      >
        <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
          <MetaLine>
            weighted signal terms (s·w·r){" "}
            <span className="tp-numeric" style={{ color: colors.textPrimary }}>
              {terms.resolved.length > 0
                ? terms.resolved.map((t) => signed(t.contribution)).join(" · ")
                : "—"}
            </span>{" "}
            {terms.resolved.length > 0 && (
              <>
                = <span className="tp-numeric" style={{ color: colors.textPrimary }}>{signed(terms.rawWeightedSum)}</span>
              </>
            )}
          </MetaLine>
          <MetaLine>
            corroboration{" "}
            <span className="tp-numeric" style={{ color: colors.textPrimary }}>
              {terms.distinctSources.length}
            </span>{" "}
            distinct source type{terms.distinctSources.length === 1 ? "" : "s"}
            {terms.distinctSources.length > 0 ? ` (${terms.distinctSources.join(", ")})` : ""} ·{" "}
            <span className="tp-numeric" style={{ color: colors.textPrimary }}>
              {terms.resolved.length + terms.missingIds.length}
            </span>{" "}
            signal{terms.resolved.length + terms.missingIds.length === 1 ? "" : "s"}
          </MetaLine>
          <MetaLine muted>
            decay · corroboration factor · scale applied engine-side (not exposed by GET
            /proposals) — rationale: {proposal.rationale || "—"}
          </MetaLine>
          {terms.missingIds.length > 0 && (
            <MetaLine muted>
              {terms.missingIds.length} contributing signal{terms.missingIds.length === 1 ? "" : "s"}{" "}
              outside the loaded feed window
            </MetaLine>
          )}
        </div>
      </Section>

      {/* ---- Source signals ------------------------------------------------------------------- */}
      <Section title={<>SOURCE SIGNALS ({terms.resolved.length})</>}>
        {terms.resolved.length === 0 ? (
          <MetaLine muted>No contributing signals resolvable from the recent feed.</MetaLine>
        ) : (
          <div style={{ display: "flex", flexDirection: "column" }}>
            {terms.resolved.map((t) => (
              <div
                key={t.signalId}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: layout.x2,
                  minHeight: shape.rowHeightDense,
                  borderBottom: `1px solid ${colors.border}`,
                  fontSize: typeTokens.scale.body,
                }}
              >
                <SourceBadge source={t.sourceType} />
                <span
                  style={{
                    fontFamily: "var(--font-sans)",
                    color: colors.textPrimary,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    flex: 1,
                    minWidth: 0,
                  }}
                >
                  {t.filer ?? "—"}
                </span>
                <span
                  className="tp-numeric"
                  title={formatTimeUtc(t.disclosedAt)}
                  style={{ color: colors.textSecondary, flexShrink: 0 }}
                >
                  {formatTime(t.disclosedAt)}
                </span>
                <span
                  className="tp-numeric"
                  title="sentiment × weight × reliability"
                  style={{ color: colors.textSecondary, flexShrink: 0 }}
                >
                  {signed(t.contribution)}
                </span>
                {t.url ? (
                  <button
                    type="button"
                    onClick={() => onOpenSource(t.url!)}
                    title={t.url}
                    style={{
                      cursor: "pointer",
                      border: "none",
                      background: "transparent",
                      color: colors.accent,
                      fontFamily: "var(--font-sans)",
                      fontSize: typeTokens.scale.meta,
                      padding: 0,
                      flexShrink: 0,
                    }}
                  >
                    source ↗
                  </button>
                ) : (
                  <span
                    style={{ color: colors.textMuted, fontSize: typeTokens.scale.meta, flexShrink: 0 }}
                  >
                    no link
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* ---- Risk pre-check ------------------------------------------------------------------- */}
      <Section title="RISK PRE-CHECK">
        <PrecheckStrip precheck={precheck} />
      </Section>

      {/* ---- Chain summary -------------------------------------------------------------------- */}
      <Section
        title={
          <span style={{ display: "inline-flex", alignItems: "center", gap: layout.x2 }}>
            CHAIN
            <PaneKeyButton
              label="chain"
              hotkey="c"
              onClick={onOpenChain}
              disabled={chainCorrelationId === null}
              title={
                chainCorrelationId === null
                  ? "No signal.received event in the loaded events window"
                  : undefined
              }
            />
          </span>
        }
      >
        <div
          className="tp-numeric"
          style={{ fontSize: typeTokens.scale.body, color: colors.chain, display: "flex", gap: layout.x2, flexWrap: "wrap" }}
        >
          <span>
            signal ×{terms.resolved.length + terms.missingIds.length}
          </span>
          <Arrow />
          <span>decision (awaiting approval)</span>
          <Arrow />
          <span>
            risk{" "}
            {precheck.state === "done" && precheck.data.enabled && precheck.data.verdict ? (
              <span
                style={{
                  color: precheck.data.verdict === "rejected" ? colors.loss : colors.textPrimary,
                }}
              >
                {precheck.data.verdict}
              </span>
            ) : (
              <span style={{ color: colors.textMuted }}>pre-check pending</span>
            )}
          </span>
          <Arrow />
          <span style={{ color: colors.textMuted }}>(intent pending)</span>
        </div>
      </Section>

      <div style={{ flex: 1 }} />

      {/* ---- Approve / Reject footer ---------------------------------------------------------- */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: layout.x4,
          paddingTop: layout.x3,
          borderTop: `1px solid ${colors.border}`,
          minHeight: shape.rowHeightDense,
          alignItems: "center",
        }}
      >
        {pendingAction ? (
          <ConfirmChip
            label={
              pendingAction.type === "approve"
                ? `Approve ${pendingAction.symbol}`
                : `Reject ${pendingAction.symbol}`
            }
            tone={pendingAction.type === "approve" ? "accent" : "loss"}
            onConfirm={onConfirmPending}
            onCancel={onCancelPending}
          />
        ) : (
          <>
            <FooterButton
              label="[A]pprove"
              color={colors.accent}
              disabled={busy}
              onClick={onRequestApprove}
            />
            <FooterButton
              label="[R]eject"
              color={colors.textSecondary}
              disabled={busy}
              onClick={onRequestReject}
            />
          </>
        )}
      </div>
    </div>
  );
}

// ---- Risk pre-check strip ---------------------------------------------------------------------

function PrecheckStrip({ precheck }: { precheck: PrecheckState }) {
  if (precheck.state === "loading") {
    return <SkeletonRows rows={1} rowHeight={shape.rowHeightDense} />;
  }
  if (precheck.state === "error") {
    return (
      <MetaLine muted={precheck.status === 404}>
        {precheck.status === 404 ? (
          <>Risk pre-check endpoint unavailable (backend route not deployed yet).</>
        ) : (
          <span style={{ color: colors.warn }}>Risk pre-check failed: {precheck.message}</span>
        )}
      </MetaLine>
    );
  }
  const data = precheck.data;
  if (!data.enabled) {
    return (
      <MetaLine>
        <span style={{ color: colors.warn }}>
          Risk pre-check disabled — no pre-checker configured for this sleeve.
        </span>
      </MetaLine>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x2 }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: layout.x2 }}>
        {data.checks.map((check) => (
          <span
            key={check.name}
            title={[
              check.reason,
              check.limit ? `limit ${check.limit}` : null,
              check.current ? `current ${check.current}` : null,
              check.projected ? `projected ${check.projected}` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: layout.x1,
              border: `1px solid ${check.passed ? colors.border : colors.loss}`,
              borderRadius: shape.radiusBadge,
              padding: `0 ${layout.x2}px`,
              height: shape.rowHeightTicker,
              fontFamily: "var(--font-sans)",
              fontSize: typeTokens.scale.body,
              color: check.passed ? colors.textPrimary : colors.loss,
              background: check.passed ? "transparent" : withAlpha(colors.loss, 0.08),
            }}
          >
            <span aria-hidden style={{ color: check.passed ? colors.textPrimary : colors.loss }}>
              {check.passed ? "✓" : "✗"}
            </span>
            {check.label}
          </span>
        ))}
      </div>
      <MetaLine>
        verdict{" "}
        <span
          className="tp-numeric"
          style={{
            fontWeight: 600,
            color: data.verdict === "rejected" ? colors.loss : data.verdict === "modified" ? colors.warn : colors.textPrimary,
          }}
        >
          {data.verdict || "—"}
        </span>{" "}
        · approved qty{" "}
        <span className="tp-numeric" style={{ color: colors.textPrimary }}>
          {formatNumber(data.approved_quantity, 4)}
        </span>
        /{formatNumber(data.requested_quantity, 4)} — {data.reason}
      </MetaLine>
    </div>
  );
}

// ---- Small shared bits --------------------------------------------------------------------------

function Section({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.meta,
          fontWeight: 600,
          letterSpacing: "0.04em",
          color: colors.textSecondary,
          borderBottom: `1px solid ${colors.border}`,
          paddingBottom: layout.x1,
          marginBottom: layout.x2,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function MetaLine({ children, muted = false }: { children: ReactNode; muted?: boolean }) {
  return (
    <div
      style={{
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.body,
        color: muted ? colors.textMuted : colors.textSecondary,
        lineHeight: 1.5,
      }}
    >
      {children}
    </div>
  );
}

function SourceBadge({ source }: { source: string }) {
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
        flexShrink: 0,
      }}
    >
      {source.replace(/_/g, " ")}
    </span>
  );
}

function PaneKeyButton({
  label,
  hotkey,
  onClick,
  disabled = false,
  title,
}: {
  label: string;
  hotkey: string;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        cursor: disabled ? "default" : "pointer",
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusButton,
        background: "transparent",
        color: disabled ? colors.textMuted : colors.accent,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        padding: `0 ${layout.x2}px`,
        height: shape.rowHeightTicker,
      }}
    >
      [{hotkey}] {label}
    </button>
  );
}

function FooterButton({
  label,
  color,
  disabled,
  onClick,
}: {
  label: string;
  color: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        cursor: disabled ? "default" : "pointer",
        border: `1px solid ${color}`,
        borderRadius: shape.radiusButton,
        background: "transparent",
        color,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.default,
        fontWeight: 600,
        padding: `${layout.x1}px ${layout.x4}px`,
      }}
    >
      {label}
    </button>
  );
}

function Arrow() {
  return (
    <span aria-hidden style={{ color: withAlpha(colors.chain, 0.5) }}>
      →
    </span>
  );
}

function signed(value: number): string {
  const formatted = formatNumber(Math.abs(value), 2);
  return value < 0 ? `−${formatted}` : `+${formatted}`;
}
