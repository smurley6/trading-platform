// Minimal History-API hook mapping pathname <-> active screen — NO router dependency (§10 /
// pre-resolved ambiguity #2: no router is approved for this phase). Handles browser back/forward
// and lets screens deep-link via a plain URL. Query-param state (`/review?proposal=…`) was OUT of
// scope for Phase A; Phase B (docs/UI-DESIGN.md §2 Ticker Pane) needs exactly one query param —
// `?ticker=LMT` — to survive reload, so `useTickerQueryParam` below adds that single case without
// growing this into a general query-string router (`/review?proposal=…` is explicitly deferred to
// Phase C per the task spec).

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

/**
 * Two-way binding between the `?ticker=` query param and the Ticker Pane's open/closed state
 * (docs/UI-DESIGN.md §2: "Route-addressable as `?ticker=LMT` so deep links survive"). Reads the
 * initial value from `window.location.search` on mount (so a deep link opens the pane on load),
 * pushes a history entry when a ticker opens/changes/closes (so back/forward + reload all work),
 * and stays in sync with external navigation via `popstate`. Deliberately scoped to just this one
 * param — see the module docblock above for why a general query-string router is out of scope.
 */
export function useTickerQueryParam(): [string | null, (ticker: string | null) => void] {
  const [ticker, setTickerState] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("ticker"),
  );

  useEffect(() => {
    function onPopState(): void {
      setTickerState(new URLSearchParams(window.location.search).get("ticker"));
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const setTicker = useCallback((next: string | null) => {
    setTickerState((current) => {
      if (current === next) return current;
      const url = new URL(window.location.href);
      if (next) {
        url.searchParams.set("ticker", next);
      } else {
        url.searchParams.delete("ticker");
      }
      window.history.pushState(null, "", `${url.pathname}${url.search}`);
      return next;
    });
  }, []);

  return [ticker, setTicker];
}
