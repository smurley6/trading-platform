// Minimal History-API hook mapping pathname <-> active screen — NO router dependency (§10 /
// pre-resolved ambiguity #2: no router is approved). Handles browser back/forward and lets
// screens deep-link via a plain URL.
//
// UI Phase C (§2 "7 screens → 5"): the Markets and Algorithms screens are DELETED — Markets is
// absorbed by the Ticker Pane (`?ticker=` slide-over, Phase B) and Algorithms by Overview's
// sleeves grid (Phase B). Their old paths (`/markets`, `/algorithms`) simply fall through
// `screenForPath`'s default and land on Overview.
//
// Query-param state: Phase B added exactly one param (`?ticker=`); Phase C needs `/review?
// proposal=…` and `/signals?provenance=…&source=…` (§7 "every screen state that matters is
// addressable"), so the one-off ticker hook is generalized into `useScreenQueryParam` — still
// deliberately NOT a router: one hook instance binds one param for one keep-alive screen, gated
// on that screen being `active` so a hidden (display:none, keep-alive) screen never reacts to a
// param that now belongs to the visible one.

import { useCallback, useEffect, useRef, useState } from "react";

export type ScreenId = "dashboard" | "proposals" | "signals" | "blotter" | "risk";

// Pathname <-> screen mapping (§2 IA route names for the five §4 destinations, applied to the
// current screen ids). Unknown paths (including the retired /markets and /algorithms) fall back
// to Overview.
const PATH_TO_SCREEN: Record<string, ScreenId> = {
  "/": "dashboard",
  "/review": "proposals",
  "/signals": "signals",
  "/blotter": "blotter",
  "/system": "risk",
};

const SCREEN_TO_PATH: Record<ScreenId, string> = {
  dashboard: "/",
  proposals: "/review",
  signals: "/signals",
  blotter: "/blotter",
  risk: "/system",
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

function readParam(key: string): string | null {
  return new URLSearchParams(window.location.search).get(key);
}

function writeParam(key: string, value: string | null, mode: "push" | "replace"): void {
  const url = new URL(window.location.href);
  const current = url.searchParams.get(key);
  if (current === value) return; // URL already agrees — no redundant history entry
  if (value !== null) {
    url.searchParams.set(key, value);
  } else {
    url.searchParams.delete(key);
  }
  const next = `${url.pathname}${url.search}`;
  if (mode === "push") {
    window.history.pushState(null, "", next);
  } else {
    window.history.replaceState(null, "", next);
  }
}

/**
 * Two-way binding between ONE query param and one keep-alive screen's state (§7 route-addressable
 * state: `?ticker=`, `/review?proposal=…`, `/signals?provenance=…&source=…`).
 *
 * Semantics, matched to the keep-alive ScreenHost model (App.tsx):
 * - The initial value is read from the URL on mount (deep links survive reload). Screens mount on
 *   first VISIT, i.e. while active, so a deep-linked param opens on the screen it belongs to.
 * - `popstate` (back/forward) syncs state ONLY while `active` — a hidden screen must never pop a
 *   modal overlay (or stale filter state) over the visible one because the URL changed.
 * - Setting a value writes the URL: `mode: "push"` adds a history entry (overlay open/close — the
 *   Phase B ticker behavior, so Back closes the pane), `mode: "replace"` rewrites in place
 *   (selection/filter state that changes on every keystroke — j/k selection must not spam
 *   history).
 * - Navigating screens pushes a bare pathname (useHistoryRoute), so params do not leak across
 *   screens; the owning screen additionally clears its state on deactivation (see
 *   useScreenOverlays / the Review + Signals screens).
 */
export function useScreenQueryParam(
  key: string,
  active: boolean,
  mode: "push" | "replace" = "push",
): [string | null, (value: string | null) => void] {
  const [value, setValueState] = useState<string | null>(() => readParam(key));
  const modeRef = useRef(mode);
  modeRef.current = mode;

  useEffect(() => {
    if (!active) return;
    function onPopState(): void {
      setValueState(readParam(key));
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [key, active]);

  const setValue = useCallback(
    (next: string | null) => {
      setValueState((current) => {
        if (current === next) return current;
        writeParam(key, next, modeRef.current);
        return next;
      });
    },
    [key],
  );

  return [value, setValue];
}
