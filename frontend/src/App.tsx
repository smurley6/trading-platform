// App shell: the new §4 operator-console chrome (36px status bar, 48px icon nav rail, cmdk
// palette, global keys — shell/AppShell.tsx) wraps the SAME screen-mounting + live event-stream
// mechanism this file has always owned. Phase A ambiguity #4: existing screens keep their
// internals unchanged in this phase — only the container/chrome/theming around them changes; the
// old inline header/nav is fully replaced by AppShell, but the refreshKey/`/ws/events` mechanism
// below is untouched.
//
// UI Phase C (docs/UI-DESIGN.md §2, §5.2-5.4): the §2 "7 screens → 5" consolidation is complete —
// the Markets and Algorithms screens are DELETED (absorbed by the Ticker Pane and Overview's
// sleeves grid respectively in Phase B; their routes fall back to Overview), the "proposals"
// screen id/route (/review) renders the rebuilt Review master-detail loop, and Signals/Blotter
// are refit as single §3-themed AG Grids. Screens now receive `active` so their focused-row
// hotkeys (c/t, j/k, x — shell/useScreenOverlays.ts) stay gated to the visible screen.

import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Blotter } from "./features/blotter/Blotter";
import { Overview } from "./features/overview/Overview";
import { Review } from "./features/review/Review";
import { RiskDashboard } from "./features/risk/RiskDashboard";
import { Signals } from "./features/signals/Signals";
import { api } from "./lib/api";
import type { Sleeve, StoredEventDto } from "./lib/types";
import { type ConnectionStatus, useEventStream } from "./lib/useEventStream";
import { useLoader } from "./lib/useLoader";
import { AppShell } from "./shell/AppShell";
import { ShellEmptyState, SkeletonRows } from "./shell/states";
import { useHistoryRoute } from "./shell/useHistoryRoute";
import type { ScreenId } from "./shell/useHistoryRoute";

export function App() {
  const [view, navigate] = useHistoryRoute();
  const [refreshKey, setRefreshKey] = useState(0);
  const [killBusy, setKillBusy] = useState(false);
  const [wsState, setWsState] = useState<ConnectionStatus>("down");

  const sleeves = useLoader<Sleeve[]>(() => api.listSleeves(), [refreshKey]);
  const kill = useLoader(() => api.killSwitch(), [refreshKey]);

  // Coalesce bursts of live events into a single refresh (rAF-adjacent debounce — §3.5 "coalesced
  // live updates": one render per burst, not per message).
  const debounceRef = useRef<number | undefined>(undefined);
  const onEvent = useCallback((_event: StoredEventDto) => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => setRefreshKey((k) => k + 1), 300);
  }, []);
  useEventStream(onEvent, undefined, setWsState);
  useEffect(() => () => window.clearTimeout(debounceRef.current), []);

  const sleeveList = sleeves.data ?? [];
  const engaged = kill.data?.engaged ?? false;

  // Lazy keep-alive: a view mounts the first time it's visited, then stays mounted (hidden via
  // display:none) so switching screens never re-mounts its widgets (TradingView iframes, loaders).
  // Nothing loads until first visit, so the upfront RAM/compute cost stays near zero.
  const [visited, setVisited] = useState<ScreenId[]>([]);
  useEffect(() => {
    setVisited((prev) => (prev.includes(view) ? prev : [...prev, view]));
  }, [view]);
  const mountedViews = visited.includes(view) ? visited : [...visited, view];

  async function toggleKill(next: boolean): Promise<void> {
    setKillBusy(true);
    try {
      await api.setKillSwitch(next);
      setRefreshKey((k) => k + 1);
    } finally {
      setKillBusy(false);
    }
  }

  return (
    <AppShell
      screen={view}
      onNavigate={navigate}
      sleeves={sleeveList}
      refreshKey={refreshKey}
      wsState={wsState}
      killEngaged={engaged}
      killBusy={killBusy}
      onToggleKill={(n) => void toggleKill(n)}
    >
      {sleeves.loading && !sleeves.data ? (
        <div style={{ padding: 24 }}>
          <SkeletonRows rows={3} />
        </div>
      ) : sleeves.error ? (
        <div style={{ padding: 24 }}>
          <ShellEmptyState message={`Backend unreachable: ${sleeves.error}`} />
        </div>
      ) : (
        mountedViews.map((v) => (
          <ScreenHost
            key={v}
            view={v}
            active={v === view}
            refreshKey={refreshKey}
            sleeves={sleeveList}
          />
        ))
      )}
    </AppShell>
  );
}

// Keep-alive host for one screen. Hidden screens are FROZEN: the memo comparator ignores
// refreshKey/sleeves changes while `active` is false, so a live-event refresh re-renders (and
// re-fetches in) only the visible screen — background screens doing loader round-trips and
// Perspective/chart work on every WS burst was the source of scroll jank (§3.5 smoothness;
// visual-gate feedback, 2026-07-02). A screen catches up automatically on re-activation: the
// comparator sees `active` flip, re-renders with the CURRENT refreshKey, and its loaders re-run
// if the key advanced while it was hidden.
const ScreenHost = memo(
  function ScreenHost({
    view,
    active,
    refreshKey,
    sleeves,
  }: {
    view: ScreenId;
    active: boolean;
    refreshKey: number;
    sleeves: Sleeve[];
  }) {
    // Overview is the one no-scroll screen (§5.1, reduced to the status glance 2026-07-06): its
    // root uses height:100% + grid bands (`auto 240px minmax(0,1fr)`), which only lay out
    // correctly if this host resolves a definite height — without it the percentage collapses
    // to auto and the bands size to content. Lock the host to the shell-main viewport for
    // "dashboard"; every other screen keeps auto height and scrolls inside .tp-shell-main.
    return (
      <div
        style={{
          display: active ? "block" : "none",
          padding: 24,
          ...(view === "dashboard" ? { height: "100%", overflow: "hidden" } : {}),
        }}
      >
        <Screen view={view} active={active} refreshKey={refreshKey} sleeves={sleeves} />
      </div>
    );
  },
  (prev, next) =>
    prev.view === next.view &&
    prev.active === next.active &&
    (next.active ? prev.refreshKey === next.refreshKey && prev.sleeves === next.sleeves : true),
);

function Screen({
  view,
  active,
  refreshKey,
  sleeves,
}: {
  view: ScreenId;
  active: boolean;
  refreshKey: number;
  sleeves: Sleeve[];
}) {
  switch (view) {
    case "dashboard":
      return <Overview active={active} refreshKey={refreshKey} />;
    case "proposals":
      return <Review active={active} refreshKey={refreshKey} />;
    case "signals":
      return <Signals active={active} refreshKey={refreshKey} />;
    case "blotter":
      return <Blotter active={active} refreshKey={refreshKey} />;
    case "risk":
      return <RiskDashboard refreshKey={refreshKey} sleeves={sleeves} />;
  }
}
