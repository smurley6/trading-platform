// App shell: header (paper/live indicator + kill switch), navigation, and the live event
// stream that drives refreshes across every screen.

import { Spinner, Text } from "@salt-ds/core";
import { useCallback, useEffect, useRef, useState } from "react";
import { KillSwitchControl } from "./components/KillSwitchControl";
import { LiveIndicator } from "./components/LiveIndicator";
import { ModeToggle } from "./components/ModeToggle";
import { StatusStrip } from "./components/StatusStrip";
import { Algorithms } from "./features/algorithms/Algorithms";
import { Blotter } from "./features/blotter/Blotter";
import { Dashboard } from "./features/dashboard/Dashboard";
import { Markets } from "./features/markets/Markets";
import { Proposals } from "./features/proposals/Proposals";
import { RiskDashboard } from "./features/risk/RiskDashboard";
import { SignalFeed } from "./features/signals/SignalFeed";
import { api } from "./lib/api";
import type { Sleeve, StoredEventDto } from "./lib/types";
import { useEventStream } from "./lib/useEventStream";
import { useLoader } from "./lib/useLoader";

type View = "dashboard" | "algorithms" | "markets" | "proposals" | "signals" | "blotter" | "risk";

const NAV: { id: View; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "algorithms", label: "Algorithms" },
  { id: "markets", label: "Markets" },
  { id: "proposals", label: "Proposals" },
  { id: "signals", label: "Signal feed" },
  { id: "blotter", label: "Blotter" },
  { id: "risk", label: "Risk" },
];

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [refreshKey, setRefreshKey] = useState(0);
  const [killBusy, setKillBusy] = useState(false);
  const [connected, setConnected] = useState(false);
  const [signalCount, setSignalCount] = useState(0);

  const sleeves = useLoader<Sleeve[]>(() => api.listSleeves(), [refreshKey]);
  const kill = useLoader(() => api.killSwitch(), [refreshKey]);

  // Coalesce bursts of live events into a single refresh, and tally signal arrivals so the
  // header can flash a "+N" badge as fresh signals stream in.
  const debounceRef = useRef<number | undefined>(undefined);
  const onEvent = useCallback((event: StoredEventDto) => {
    if (event.event_type === "signal.received") setSignalCount((n) => n + 1);
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => setRefreshKey((k) => k + 1), 300);
  }, []);
  useEventStream(onEvent, setConnected);
  useEffect(() => () => window.clearTimeout(debounceRef.current), []);

  const sleeveList = sleeves.data ?? [];
  const engaged = kill.data?.engaged ?? false;

  // Lazy keep-alive: a view mounts the first time it's visited, then stays mounted (hidden via
  // display:none) so switching tabs never re-mounts its widgets (TradingView iframes, loaders).
  // Nothing loads until first visit, so the upfront RAM/compute cost stays near zero.
  const [visited, setVisited] = useState<View[]>([]);
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
    <div style={{ minHeight: "100vh" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          padding: "12px 20px",
          borderBottom: "1px solid rgba(255,255,255,0.1)",
          flexWrap: "wrap",
        }}
      >
        <strong style={{ fontSize: 18 }}>Trading Platform</strong>
        <ModeToggle sleeves={sleeveList} onChanged={() => setRefreshKey((k) => k + 1)} />
        <LiveIndicator connected={connected} signalCount={signalCount} />
        <div style={{ flex: 1 }} />
        <KillSwitchControl engaged={engaged} busy={killBusy} onToggle={(n) => void toggleKill(n)} />
      </header>

      <StatusStrip
        connected={connected}
        sleeves={sleeveList}
        killEngaged={engaged}
        refreshKey={refreshKey}
      />

      <nav style={{ display: "flex", gap: 8, padding: "10px 20px" }}>
        {NAV.map((item) => (
          <button
            key={item.id}
            onClick={() => setView(item.id)}
            style={{
              cursor: "pointer",
              borderRadius: 6,
              padding: "6px 14px",
              fontSize: 14,
              color: "inherit",
              border: "1px solid rgba(255,255,255,0.12)",
              background: view === item.id ? "rgba(76,154,255,0.2)" : "transparent",
            }}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main style={{ padding: "8px 20px 40px" }}>
        {sleeves.loading && !sleeves.data ? (
          <Spinner />
        ) : sleeves.error ? (
          <Text color="error">Could not reach backend: {sleeves.error}</Text>
        ) : (
          mountedViews.map((v) => (
            <div key={v} style={{ display: v === view ? "block" : "none" }}>
              <Screen view={v} refreshKey={refreshKey} sleeves={sleeveList} />
            </div>
          ))
        )}
      </main>
    </div>
  );
}

function Screen({
  view,
  refreshKey,
  sleeves,
}: {
  view: View;
  refreshKey: number;
  sleeves: Sleeve[];
}) {
  switch (view) {
    case "dashboard":
      return <Dashboard refreshKey={refreshKey} sleeves={sleeves} />;
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
