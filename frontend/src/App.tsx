// App shell: header (paper/live indicator + kill switch), navigation, and the live event
// stream that drives refreshes across every screen.

import { Spinner, Text } from "@salt-ds/core";
import { useCallback, useEffect, useRef, useState } from "react";
import { KillSwitchControl } from "./components/KillSwitchControl";
import { Algorithms } from "./features/algorithms/Algorithms";
import { Backtest } from "./features/backtest/Backtest";
import { Blotter } from "./features/blotter/Blotter";
import { Dashboard } from "./features/dashboard/Dashboard";
import { Markets } from "./features/markets/Markets";
import { Proposals } from "./features/proposals/Proposals";
import { RiskDashboard } from "./features/risk/RiskDashboard";
import { SignalFeed } from "./features/signals/SignalFeed";
import { api } from "./lib/api";
import type { Sleeve } from "./lib/types";
import { useEventStream } from "./lib/useEventStream";
import { useLoader } from "./lib/useLoader";

type View =
  | "dashboard"
  | "algorithms"
  | "markets"
  | "proposals"
  | "signals"
  | "blotter"
  | "backtest"
  | "risk";

const NAV: { id: View; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "algorithms", label: "Algorithms" },
  { id: "markets", label: "Markets" },
  { id: "proposals", label: "Proposals" },
  { id: "signals", label: "Signal feed" },
  { id: "blotter", label: "Blotter" },
  { id: "backtest", label: "Backtest" },
  { id: "risk", label: "Risk" },
];

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [refreshKey, setRefreshKey] = useState(0);
  const [killBusy, setKillBusy] = useState(false);

  const sleeves = useLoader<Sleeve[]>(() => api.listSleeves(), [refreshKey]);
  const kill = useLoader(() => api.killSwitch(), [refreshKey]);

  // Coalesce bursts of live events into a single refresh.
  const debounceRef = useRef<number | undefined>(undefined);
  const bumpRefresh = useCallback(() => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => setRefreshKey((k) => k + 1), 300);
  }, []);
  useEventStream(bumpRefresh);
  useEffect(() => () => window.clearTimeout(debounceRef.current), []);

  const sleeveList = sleeves.data ?? [];
  const isLive = sleeveList.some((s) => s.mode === "live");
  const engaged = kill.data?.engaged ?? false;

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
        <span
          style={{
            padding: "2px 10px",
            borderRadius: 12,
            fontSize: 12,
            fontWeight: 700,
            background: isLive ? "rgba(239,68,68,0.2)" : "rgba(76,154,255,0.18)",
            color: isLive ? "#f87171" : "#7cb3ff",
            border: `1px solid ${isLive ? "#ef4444" : "#4c9aff"}`,
          }}
        >
          {isLive ? "LIVE" : "PAPER"}
        </span>
        <div style={{ flex: 1 }} />
        <KillSwitchControl engaged={engaged} busy={killBusy} onToggle={(n) => void toggleKill(n)} />
      </header>

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
          <Screen view={view} refreshKey={refreshKey} sleeves={sleeveList} />
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
    case "backtest":
      return <Backtest refreshKey={refreshKey} />;
    case "risk":
      return <RiskDashboard refreshKey={refreshKey} sleeves={sleeves} />;
  }
}
