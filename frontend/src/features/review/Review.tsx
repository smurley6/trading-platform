// Review `/review` (UI Phase C, docs/UI-DESIGN.md §5.2 — the centerpiece). Master–detail split:
// queue (38%) / evidence (62%), keyboard-complete:
//
//   j/k move · Enter/ArrowRight focus detail · a approve (single confirm keystroke Enter) ·
//   r reject · t chart · c chain · o open primary source · Esc unwinds one layer.
//
// Replaces the old Proposals screen's internals at the same screen id ("proposals") and route
// (/review). Data comes from the EXISTING surfaces — GET /proposals (queue), GET /signals/feed
// (contributing-signal evidence), GET /events (signal → correlation-chain join) — plus the new
// Phase C read-only GET /risk/precheck. Approve = POST /proposals/execute for that symbol
// (optimistic removal, rollback toast on failure — §3.5 <50ms keyboard latency). Reject has NO
// backend surface, so it is a session-local dismissal (features/review/proposalResolutions.ts —
// compromise documented there and in the phase report). Approved/rejected rows leave with the
// ticker-insert animation reversed (.tp-row-exit); the status bar's unreviewed count subscribes
// to the same resolution store, so it decrements in the same frame.
//
// All new code is built exclusively on the §3 token system.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { useToast } from "../../components/Toast";
import { api, ApiError } from "../../lib/api";
import { formatTime } from "../../lib/format";
import { signalMetaFromEvents } from "../../lib/signalMeta";
import { colors, layout, motionTokens, shape, type as typeTokens } from "../../lib/tokens";
import type { CycleMode, ProposalItem, SignalFeedItem } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";
import { ChainInspector } from "../../shell/ChainInspector";
import { ConfirmChip } from "../../shell/ConfirmChip";
import { SlideOverStackProvider } from "../../shell/SlideOverStackProvider";
import { ShellEmptyState, SkeletonRows } from "../../shell/states";
import { TickerPane } from "../../shell/TickerPane";
import { useScreenQueryParam } from "../../shell/useHistoryRoute";
import { useScreenOverlays } from "../../shell/useScreenOverlays";
import { convictionTerms } from "./convictionTerms";
import { EvidencePane, type PendingAction, type PrecheckState } from "./EvidencePane";
import {
  getResolutions,
  pruneResolutions,
  resolveProposal,
  subscribeResolutions,
  unresolveProposal,
} from "../../lib/proposalResolutions";
import { formatAge, newestSignalAtMs, orderQueue } from "./reviewQueue";

interface ReviewProps {
  active: boolean;
  refreshKey: number;
}

export function Review({ active, refreshKey }: ReviewProps) {
  const toast = useToast();
  const proposals = useLoader(() => api.proposals(), [refreshKey]);
  const signals = useLoader(() => api.signalFeed(300), [refreshKey]);
  const events = useLoader(() => api.events(1000), [refreshKey]);

  const signalsById = useMemo(() => {
    const map = new Map<string, SignalFeedItem>();
    for (const s of signals.data ?? []) map.set(s.signal_id, s);
    return map;
  }, [signals.data]);

  const signalMeta = useMemo(() => signalMetaFromEvents(events.data ?? []), [events.data]);

  const resolutions = useSyncExternalStore(subscribeResolutions, getResolutions);

  // Keep the session-local resolution store bounded to the live proposal set (approvals become
  // durable server-side once the scan stops proposing the symbol).
  useEffect(() => {
    if (!proposals.data) return;
    pruneResolutions(new Set(proposals.data.proposals.map((p) => p.symbol)));
  }, [proposals.data]);

  const queue = useMemo(
    () => orderQueue(proposals.data?.proposals ?? [], signalsById, resolutions),
    [proposals.data, signalsById, resolutions],
  );

  // ---- Selection: symbol-keyed, addressable as /review?proposal=SYM (§7). replace-mode so j/k
  // scrubbing never spams history. When the selected symbol leaves the queue (approve/reject),
  // selection falls to the row that took its index.
  const [selectedParam, setSelectedParam] = useScreenQueryParam("proposal", active, "replace");
  const lastIndexRef = useRef(0);
  const paramIndex = selectedParam ? queue.findIndex((p) => p.symbol === selectedParam) : -1;
  const selectedIndex =
    queue.length === 0 ? -1 : paramIndex >= 0 ? paramIndex : Math.min(lastIndexRef.current, queue.length - 1);
  const selected: ProposalItem | null = selectedIndex >= 0 ? queue[selectedIndex]! : null;

  useEffect(() => {
    if (selectedIndex >= 0) lastIndexRef.current = selectedIndex;
  }, [selectedIndex]);
  useEffect(() => {
    const symbol = selected?.symbol ?? null;
    if (symbol !== selectedParam) setSelectedParam(symbol);
  }, [selected?.symbol, selectedParam, setSelectedParam]);

  const selectIndex = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, queue.length - 1));
      const row = queue[clamped];
      if (row) setSelectedParam(row.symbol);
    },
    [queue, setSelectedParam],
  );

  // ---- Evidence derivations for the selected proposal.
  const terms = useMemo(
    () => convictionTerms(selected?.contributing_signal_ids ?? [], signalsById),
    [selected, signalsById],
  );

  // Chain for a not-yet-executed proposal = its evidence chain: the ingestion correlation id of
  // its newest resolvable contributing signal (signal.received events carry it; the loaded
  // events window is the only client-side surface for that join — null when outside it).
  const chainCorrelationId = useMemo(() => {
    for (const term of terms.resolved) {
      const meta = signalMeta.get(term.signalId);
      if (meta) return meta.correlationId;
    }
    return null;
  }, [terms, signalMeta]);
  const chainIdRef = useRef<string | null>(null);
  chainIdRef.current = chainCorrelationId;
  const selectedRef = useRef<ProposalItem | null>(null);
  selectedRef.current = selected;

  // ---- Overlays (shared focused-row model): c = chain of the selected proposal, t = its chart.
  const overlays = useScreenOverlays(active, {
    correlationId: () => chainIdRef.current,
    ticker: () => selectedRef.current?.symbol ?? null,
  });

  // ---- Risk pre-check (§5.2): read-only GET /risk/precheck for the selected proposal, debounced
  // so j/k scrubbing doesn't burst requests. Loading/disabled/404 states render explicitly.
  const [precheck, setPrecheck] = useState<PrecheckState>({ state: "loading" });
  useEffect(() => {
    if (!selected) return;
    const { symbol, side, quantity } = selected;
    let cancelled = false;
    setPrecheck({ state: "loading" });
    const timer = window.setTimeout(() => {
      api
        .riskPrecheck({ symbol, side, quantity })
        .then((data) => {
          if (!cancelled) setPrecheck({ state: "done", data });
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setPrecheck({
            state: "error",
            status: err instanceof ApiError ? err.status : null,
            message: err instanceof Error ? err.message : String(err),
          });
        });
    }, 150);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [selected?.symbol, selected?.side, selected?.quantity, refreshKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Approve / reject with the §7 single-confirm keystroke, optimistic removal (§3.5).
  const [pendingAction, setPendingAction] = useState<PendingAction>(null);
  // Rows currently animating out (.tp-row-exit) — resolution lands when the exit completes.
  const [leaving, setLeaving] = useState<ReadonlyMap<string, "approved" | "rejected">>(new Map());

  // Idempotent (guarded by the leaving map): called from BOTH the row's animationend AND a
  // duration-based fallback timer — CSS animation clocks freeze in throttled/background tabs,
  // and the resolution must land regardless of whether the exit animation actually played.
  const leavingRef = useRef(leaving);
  leavingRef.current = leaving;
  const onRowExitEnd = useCallback((symbol: string) => {
    const resolution = leavingRef.current.get(symbol);
    if (!resolution) return;
    resolveProposal(symbol, resolution); // lands in the shared store -> queue + status bar update
    setLeaving((prev) => {
      if (!prev.has(symbol)) return prev;
      const next = new Map(prev);
      next.delete(symbol);
      return next;
    });
  }, []);
  const exitTimersRef = useRef<Set<number>>(new Set());
  useEffect(
    () => () => {
      for (const timer of exitTimersRef.current) window.clearTimeout(timer);
    },
    [],
  );

  // NOTE: plain callback with the action passed in — never side effects inside a setState
  // updater (React StrictMode double-invokes updaters, which double-fired the approve).
  const confirmPending = useCallback(
    (action: { type: "approve" | "reject"; symbol: string }) => {
      const { type, symbol } = action;
      const resolution = type === "approve" ? ("approved" as const) : ("rejected" as const);
      setPendingAction(null);
      setLeaving((prev) => new Map(prev).set(symbol, resolution));
      // Fallback for animationend (see onRowExitEnd): double the §3.4 row duration, then force.
      const timer = window.setTimeout(() => {
        exitTimersRef.current.delete(timer);
        onRowExitEnd(symbol);
      }, motionTokens.durationRow * 2);
      exitTimersRef.current.add(timer);
      if (type === "approve") {
        api
          .executeProposals([symbol])
          .then((result) => {
            toast.success(`Approved ${symbol} — ${result.orders.length} order(s) placed`);
          })
          .catch((err: unknown) => {
            // §3.5 rollback: the row returns to the queue, the failure toast persists.
            unresolveProposal(symbol);
            setLeaving((prev) => {
              const next = new Map(prev);
              next.delete(symbol);
              return next;
            });
            toast.error(
              `Approve ${symbol} failed: ${err instanceof Error ? err.message : String(err)}`,
            );
          });
      } else {
        toast.info(`Rejected ${symbol} — hidden for this session`);
      }
    },
    [toast, onRowExitEnd],
  );

  // ---- Screen-scoped hotkeys. Gated on active (keep-alive), no open overlay (inert background,
  // §3.5), and no pending confirm chip (the chip owns Enter/Esc while mounted).
  const keysEnabled = active && !overlays.overlayOpen && pendingAction === null;
  const evidenceRef = useRef<HTMLDivElement>(null);

  useHotkeys("j", () => selectIndex(selectedIndex + 1), { enabled: keysEnabled, enableOnFormTags: false }, [selectedIndex, selectIndex, keysEnabled]);
  useHotkeys("k", () => selectIndex(selectedIndex - 1), { enabled: keysEnabled, enableOnFormTags: false }, [selectedIndex, selectIndex, keysEnabled]);
  useHotkeys(
    "enter,right",
    () => {
      if (selected) evidenceRef.current?.focus();
    },
    { enabled: keysEnabled, enableOnFormTags: false },
    [selected, keysEnabled],
  );
  useHotkeys(
    "a",
    () => {
      if (selected) setPendingAction({ type: "approve", symbol: selected.symbol });
    },
    { enabled: keysEnabled, enableOnFormTags: false },
    [selected, keysEnabled],
  );
  useHotkeys(
    "r",
    () => {
      if (selected) setPendingAction({ type: "reject", symbol: selected.symbol });
    },
    { enabled: keysEnabled, enableOnFormTags: false },
    [selected, keysEnabled],
  );
  useHotkeys(
    "o",
    () => {
      const url = terms.resolved.map((t) => t.url).find((u): u is string => u !== null);
      if (url) window.open(url, "_blank", "noopener,noreferrer");
    },
    { enabled: keysEnabled, enableOnFormTags: false },
    [terms, keysEnabled],
  );

  // Cancel a pending confirm if the selection moves out from under it.
  useEffect(() => {
    if (pendingAction && pendingAction.symbol !== selected?.symbol) setPendingAction(null);
  }, [pendingAction, selected?.symbol]);

  // ---- Cycle-mode control (kept from the old Proposals screen so arming/disarming AUTO stays
  // reachable — the §4 status bar only DISPLAYS the mode). Committing action -> ConfirmChip.
  const mode: CycleMode | null = proposals.data?.mode ?? null;
  const [confirmingMode, setConfirmingMode] = useState(false);
  const [modeBusy, setModeBusy] = useState(false);
  const reloadProposals = proposals.reload;
  const flipMode = useCallback(() => {
    if (!mode) return;
    const next: CycleMode = mode === "auto" ? "manual" : "auto";
    setModeBusy(true);
    api
      .setCycleMode(next)
      .then(() => {
        toast.success(`Cycle mode set to ${next.toUpperCase()}`);
        reloadProposals();
      })
      .catch((err: unknown) => {
        toast.error(
          `Cycle mode change failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      })
      .finally(() => setModeBusy(false));
  }, [mode, reloadProposals, toast]);

  const initialLoading = proposals.loading && !proposals.data;
  const lastIngested = signals.data?.length ? signals.data[signals.data.length - 1]!.occurred_at : null;
  const nowMs = Date.now();

  return (
    <SlideOverStackProvider>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "38fr 62fr",
          gap: layout.x3,
          height: `calc(100vh - var(--status-bar-height) - ${layout.x6 * 2}px)`,
          minHeight: 0,
        }}
      >
        {/* ---- Queue (38%) ------------------------------------------------------------------ */}
        <Pane>
          <PaneHeader>
            <span>Queue ({queue.length})</span>
            <span style={{ flex: 1 }} />
            {confirmingMode ? (
              <ConfirmChip
                label={`Set ${mode === "auto" ? "MANUAL" : "AUTO"}`}
                onConfirm={() => {
                  setConfirmingMode(false);
                  flipMode();
                }}
                onCancel={() => setConfirmingMode(false)}
              />
            ) : (
              <button
                type="button"
                disabled={mode === null || modeBusy}
                onClick={() => setConfirmingMode(true)}
                title="Cycle trade mode — click to switch (confirm required)"
                style={{
                  cursor: mode === null || modeBusy ? "default" : "pointer",
                  border: `1px solid ${colors.border}`,
                  borderRadius: shape.radiusBadge,
                  background: "transparent",
                  fontFamily: "var(--font-sans)",
                  fontSize: typeTokens.scale.meta,
                  fontWeight: 600,
                  letterSpacing: "0.04em",
                  padding: `0 ${layout.x2}px`,
                  height: shape.rowHeightTicker - layout.x1,
                  color: mode === "auto" ? colors.warn : colors.textSecondary,
                }}
              >
                {mode ? mode.toUpperCase() : "—"}
              </button>
            )}
          </PaneHeader>
          <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
            {initialLoading ? (
              <div style={{ padding: layout.x3 }}>
                <SkeletonRows rows={3} />
              </div>
            ) : proposals.error ? (
              <ShellEmptyState message={`Could not load proposals: ${proposals.error}`} />
            ) : queue.length === 0 ? (
              <ShellEmptyState
                message={
                  <>
                    No proposals awaiting review.
                    <br />
                    {lastIngested ? (
                      <span className="tp-numeric">
                        Last signal ingested {formatTime(lastIngested)}.
                      </span>
                    ) : (
                      "The next ingest cycle will surface candidates here."
                    )}
                  </>
                }
              />
            ) : (
              queue.map((proposal, index) => (
                <QueueRow
                  key={proposal.symbol}
                  proposal={proposal}
                  age={formatAge(newestSignalAtMs(proposal, signalsById), nowMs)}
                  selected={index === selectedIndex}
                  leaving={leaving.has(proposal.symbol)}
                  onSelect={() => setSelectedParam(proposal.symbol)}
                  onExitEnd={() => onRowExitEnd(proposal.symbol)}
                />
              ))
            )}
          </div>
        </Pane>

        {/* ---- Evidence (62%) ----------------------------------------------------------------- */}
        <Pane>
          {selected ? (
            <EvidencePane
              proposal={selected}
              terms={terms}
              precheck={precheck}
              chainCorrelationId={chainCorrelationId}
              pendingAction={pendingAction}
              busy={leaving.size > 0}
              onRequestApprove={() => setPendingAction({ type: "approve", symbol: selected.symbol })}
              onRequestReject={() => setPendingAction({ type: "reject", symbol: selected.symbol })}
              onConfirmPending={() => {
                if (pendingAction) confirmPending(pendingAction);
              }}
              onCancelPending={() => setPendingAction(null)}
              onOpenChart={() => overlays.openTicker(selected.symbol)}
              onOpenChain={() => {
                if (chainCorrelationId) overlays.openChain(chainCorrelationId);
              }}
              onOpenSource={(url) => window.open(url, "_blank", "noopener,noreferrer")}
              paneRef={evidenceRef}
            />
          ) : (
            <ShellEmptyState message="Select a proposal to inspect its evidence." />
          )}
        </Pane>
      </div>

      <ChainInspector
        correlationId={overlays.correlationId}
        onClose={overlays.closeChain}
        returnFocusRef={overlays.lastRowRef}
      />
      <TickerPane
        ticker={overlays.ticker}
        onClose={overlays.closeTicker}
        returnFocusRef={overlays.lastRowRef}
      />
    </SlideOverStackProvider>
  );
}

// ---- Queue row ----------------------------------------------------------------------------------

function QueueRow({
  proposal,
  age,
  selected,
  leaving,
  onSelect,
  onExitEnd,
}: {
  proposal: ProposalItem;
  age: string;
  selected: boolean;
  leaving: boolean;
  onSelect: () => void;
  onExitEnd: () => void;
}) {
  const sideColor = proposal.side === "buy" ? colors.gain : colors.loss;
  return (
    <div
      role="row"
      tabIndex={0}
      className={leaving ? "tp-row-exit" : undefined}
      onAnimationEnd={(e) => {
        if (e.target === e.currentTarget && leaving) onExitEnd();
      }}
      onClick={onSelect}
      onFocus={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter") onSelect();
      }}
      aria-selected={selected}
      style={{
        display: "flex",
        alignItems: "center",
        gap: layout.x2,
        height: shape.rowHeightDense,
        padding: `0 ${layout.x2}px`,
        cursor: "pointer",
        borderBottom: `1px solid ${colors.border}`,
        background: selected ? colors.bgRaised : "transparent",
        outline: selected ? `1px solid ${colors.accent}` : "none",
        outlineOffset: -1,
      }}
    >
      <span
        aria-hidden
        style={{
          width: layout.x2,
          color: selected ? colors.accent : "transparent",
          fontSize: typeTokens.scale.body,
          flexShrink: 0,
        }}
      >
        ▸
      </span>
      <span
        className="tp-numeric"
        style={{ fontWeight: 600, color: colors.textPrimary, width: layout.x6 * 2, flexShrink: 0 }}
      >
        {proposal.symbol}
      </span>
      <span
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.meta,
          fontWeight: 600,
          letterSpacing: "0.04em",
          color: sideColor,
          width: layout.x8,
          flexShrink: 0,
        }}
      >
        {proposal.side.toUpperCase()}
      </span>
      <span
        className="tp-numeric"
        style={{ color: colors.textSecondary, fontSize: typeTokens.scale.body }}
      >
        conv{" "}
        <span style={{ color: colors.textPrimary, fontWeight: 600 }}>
          {Math.round(proposal.conviction_score)}
        </span>
      </span>
      <span style={{ flex: 1 }} />
      <span
        className="tp-numeric"
        style={{ color: colors.textMuted, fontSize: typeTokens.scale.meta, flexShrink: 0 }}
      >
        {age}
      </span>
    </div>
  );
}

// ---- Pane chrome --------------------------------------------------------------------------------

function Pane({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusPane,
        background: colors.bgSurface,
        overflow: "hidden",
      }}
    >
      {children}
    </div>
  );
}

function PaneHeader({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: layout.x2,
        padding: `${layout.x1}px ${layout.x2}px`,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.04em",
        color: colors.textSecondary,
        borderBottom: `1px solid ${colors.border}`,
        flexShrink: 0,
        minHeight: shape.rowHeightTicker,
      }}
    >
      {children}
    </div>
  );
}
