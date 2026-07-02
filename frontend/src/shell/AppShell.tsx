// §4 app shell composition: 36px status bar + 48px icon nav rail + main content slot + cmdk
// command-palette skeleton + global keys. This is the ONLY new chrome Phase A introduces around
// existing screens — App.tsx still owns which screen mounts and the refreshKey/event-stream
// mechanism (Phase A ambiguity #4: container/chrome/theming only, screen internals untouched).

import { useCallback, useState } from "react";
import type { ReactNode } from "react";
import type { ConnectionStatus } from "../lib/useEventStream";
import type { Sleeve } from "../lib/types";
import { CommandPalette } from "./CommandPalette";
import { GlobalHotkeys } from "./GlobalHotkeys";
import { NavRail } from "./NavRail";
import { StatusBar } from "./StatusBar";
import type { ScreenId } from "./useHistoryRoute";

interface AppShellProps {
  screen: ScreenId;
  onNavigate: (screen: ScreenId) => void;
  sleeves: Sleeve[];
  refreshKey: number;
  wsState: ConnectionStatus;
  killEngaged: boolean;
  killBusy: boolean;
  onToggleKill: (next: boolean) => void;
  children: ReactNode;
}

export function AppShell({
  screen,
  onNavigate,
  sleeves,
  refreshKey,
  wsState,
  killEngaged,
  killBusy,
  onToggleKill,
  children,
}: AppShellProps) {
  const [paletteOpen, setPaletteOpen] = useState(false);

  // §4 "Esc closes topmost overlay": the palette is the only overlay this shell itself owns, so
  // closing it is the shell's whole contribution to that contract — per-row overlays (Ticker
  // Pane, Chain Inspector, Phase B/C) own their own Esc handling via Radix Dialog's built-in
  // behavior and layer on top of this without conflict (each Escape keydown closes exactly one
  // topmost layer; the browser event isn't re-dispatched once a Radix Dialog's own listener
  // has already handled it, since Radix stops propagation on its content).
  const handleEscape = useCallback(() => {
    setPaletteOpen(false);
  }, []);

  return (
    <div className="tp-shell tp-shell-root">
      <StatusBar
        sleeves={sleeves}
        refreshKey={refreshKey}
        wsState={wsState}
        killEngaged={killEngaged}
        killBusy={killBusy}
        onToggleKill={onToggleKill}
        onNavigate={onNavigate}
      />
      <div className="tp-shell-body">
        <NavRail active={screen} onNavigate={onNavigate} />
        <main className="tp-shell-main">{children}</main>
      </div>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} onNavigate={onNavigate} />
      <GlobalHotkeys
        paletteOpen={paletteOpen}
        onNavigate={onNavigate}
        onOpenPalette={() => setPaletteOpen(true)}
        onEscape={handleEscape}
      />
    </div>
  );
}
