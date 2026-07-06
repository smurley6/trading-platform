// App shell: the new §4 operator-console chrome (36px status bar, 48px icon nav rail, cmdk
// palette, global keys — shell/AppShell.tsx) wraps the SAME screen-mounting + live event-stream
// mechanism this file has always owned. Phase A ambiguity #4: existing screens keep their
// internals unchanged in this phase — only the container/chrome/theming around them changes; the
// old inline header/nav is fully replaced by AppShell, but the refreshKey/`/ws/events` mechanism
// below is untouched.
//
// UI Phase B (docs/UI-DESIGN.md §5.1): the "dashboard" screen id/route (`/`) now renders the
// rebuilt Overview (features/overview/Overview.tsx) instead of the old Dashboard — same
// screen-id/route mapping (useHistoryRoute is unchanged), only the component swapped. Algorithms
// and Markets keep their own routes/screens unchanged this phase (see MEMORY.md §8 for the
// deferral — §2's "fold into Overview"/"demote to Ticker Pane" absorption is not yet a route
// deletion).

import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Algorithms } from "./features/algorithms/Algorithms";
import { Blotter } from "./features/blotter/Blotter";
import { Overview } from "./features/overview/Overview";
import { Markets } from "./features/markets/Markets";
import { Proposals } from "./features/proposals/Proposals";
import { RiskDashboard } from "./features/risk/RiskDashboard";
import { SignalFeed } from "./features/signals/SignalFeed";
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
    return (
      <div style={{ display: active ? "block" : "none", padding: 24 }}>
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
      return <Overview active={active} refreshKey={refreshKey} sleeves={sleeves} />;
    case "algorithms":
      return <Algorithms refreshKey={refreshKey} sleeves={sleeves} />;
    case "markets":
      return <Markets refreshKey={refreshKey} />;
    case "proposals":
      return <Proposals refreshKey={refreshKey} />;
    case "signals":
      return <SignalFeed refreshKey={refreshKey} />;
    case "blotter":
      return <Blotter refreshKey={refreshKey} />;
    case "risk":
      return <RiskDashboard refreshKey={refreshKey} sleeves={sleeves} />;
  }
}
