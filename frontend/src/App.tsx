// App shell: header (paper/live indicator + kill switch), navigation, and the live event
// stream that drives refreshes across every screen.

import { useCallback, useEffect, useRef, useState } from "react";
import { KillSwitchControl } from "./components/KillSwitchControl";
import { LiveIndicator } from "./components/LiveIndicator";
import { ModeToggle } from "./components/ModeToggle";
import { StatusStrip } from "./components/StatusStrip";
import { EmptyState, LoadingState } from "./components/ui";
import { Algorithms } from "./features/algorithms/Algorithms";
import { Blotter } from "./features/blotter/Blotter";
import { Dashboard } from "./features/dashboard/Dashboard";
import { Markets } from "./features/markets/Markets";
import { Proposals } from "./features/proposals/Proposals";
import { RiskDashboard } from "./features/risk/RiskDashboard";
import { SignalFeed } from "./features/signals/SignalFeed";
import { api } from "./lib/api";
import { color, radius, shadow, space, text, transitions, z } from "./lib/tokens";
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
      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: z.strip,
          background: "rgba(10,13,20,0.82)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          borderBottom: `1px solid ${color.hairline}`,
        }}
      >
        <header
          style={{
            display: "flex",
            alignItems: "center",
            gap: space.lg,
            padding: "12px 24px",
            flexWrap: "wrap",
          }}
        >
          <BrandLockup />
          <ModeToggle sleeves={sleeveList} onChanged={() => setRefreshKey((k) => k + 1)} />
          <LiveIndicator connected={connected} signalCount={signalCount} />
          <div style={{ flex: 1 }} />
          <KillSwitchControl
            engaged={engaged}
            busy={killBusy}
            onToggle={(n) => void toggleKill(n)}
          />
        </header>

        <StatusStrip
          connected={connected}
          sleeves={sleeveList}
          killEngaged={engaged}
          refreshKey={refreshKey}
        />

        <nav
          style={{
            display: "flex",
            gap: space.xs,
            padding: "8px 24px",
            flexWrap: "wrap",
            borderTop: `1px solid ${color.hairline}`,
          }}
        >
          {NAV.map((item) => {
            const active = view === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                aria-current={active ? "page" : undefined}
                style={{
                  cursor: "pointer",
                  borderRadius: radius.md,
                  padding: "6px 14px",
                  ...text.label,
                  fontWeight: active ? 700 : 500,
                  color: active ? color.accentText : color.muted,
                  border: `1px solid ${active ? color.accentBorder : "transparent"}`,
                  background: active ? color.accentSoft : "transparent",
                  transition: transitions("background", "color", "border-color"),
                }}
              >
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      <main style={{ padding: "24px 24px 56px", maxWidth: 1600, margin: "0 auto" }}>
        {sleeves.loading && !sleeves.data ? (
          <LoadingState label="Connecting to backend…" />
        ) : sleeves.error ? (
          <EmptyState
            title="Backend unreachable"
            message={`Could not reach backend: ${sleeves.error}`}
          />
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

// Brand lockup: a compact monogram mark + wordmark. Deliberately restrained — the mark carries
// the accent, the wordmark carries the hierarchy.
function BrandLockup() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div
        aria-hidden
        style={{
          width: 30,
          height: 30,
          borderRadius: radius.md,
          display: "grid",
          placeItems: "center",
          background: `linear-gradient(140deg, ${color.accentStrong}, ${color.accent})`,
          boxShadow: shadow.glowAccent,
          color: "#fff",
          fontWeight: 800,
          fontSize: 15,
          letterSpacing: "-0.03em",
        }}
      >
        T
      </div>
      <div style={{ display: "flex", flexDirection: "column", lineHeight: 1 }}>
        <strong style={{ ...text.h3, color: color.text, letterSpacing: "-0.01em" }}>
          Trading Platform
        </strong>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.18em",
            color: color.muted,
            textTransform: "uppercase",
            marginTop: 2,
          }}
        >
          Terminal
        </span>
      </div>
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
