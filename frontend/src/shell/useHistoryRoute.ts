// Minimal History-API hook mapping pathname <-> active screen — NO router dependency (§10 /
// pre-resolved ambiguity #2: no router is approved for this phase). Handles browser back/forward
// and lets screens deep-link via a plain URL. Query-param state (`/review?proposal=…`) is
// explicitly OUT of scope for Phase A per the task spec — only the bare pathname is mapped.

import { useCallback, useEffect, useState } from "react";

export type ScreenId =
  | "dashboard"
  | "algorithms"
  | "markets"
  | "proposals"
  | "signals"
  | "blotter"
  | "risk";

// Pathname <-> screen mapping (§2 IA route names, applied to the CURRENT 7 screen names since
// Phase A keeps every existing screen's identity — the §2 5-screen consolidation is Phase B+).
const PATH_TO_SCREEN: Record<string, ScreenId> = {
  "/": "dashboard",
  "/review": "proposals",
  "/signals": "signals",
  "/blotter": "blotter",
  "/system": "risk",
  "/markets": "markets",
  "/algorithms": "algorithms",
};

const SCREEN_TO_PATH: Record<ScreenId, string> = {
  dashboard: "/",
  proposals: "/review",
  signals: "/signals",
  blotter: "/blotter",
  risk: "/system",
  markets: "/markets",
  algorithms: "/algorithms",
};

function screenForPath(pathname: string): ScreenId {
  return PATH_TO_SCREEN[pathname] ?? "dashboard";
}

/**
 * Two-way binding between `window.location.pathname` and the active screen. Pushes a new
 * history entry on navigation (so back/forward works) without pulling in a router.
 */
export function useHistoryRoute(): [ScreenId, (screen: ScreenId) => void] {
  const [screen, setScreenState] = useState<ScreenId>(() =>
    screenForPath(window.location.pathname),
  );

  useEffect(() => {
    function onPopState(): void {
      setScreenState(screenForPath(window.location.pathname));
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((next: ScreenId) => {
    setScreenState((current) => {
      if (current === next) return current;
      const path = SCREEN_TO_PATH[next];
      if (window.location.pathname !== path) {
        window.history.pushState(null, "", path);
      }
      return next;
    });
  }, []);

  return [screen, navigate];
}
