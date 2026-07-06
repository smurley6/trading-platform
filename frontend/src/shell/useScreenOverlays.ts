// Shared focused-row overlay model (UI Phase C, promoting the Phase B Overview-scoped pattern):
// every data screen (Overview, Review, Signals, Blotter) owns a Chain Inspector + Ticker Pane
// pair driven by its own "currently focused row", with the `c` / `t` hotkeys §4 promises
// "anywhere a row has a correlation id" bound per screen and gated on that screen being active.
//
// Extracted verbatim from Overview.tsx's Phase B wiring so all four screens share one behavior:
// - `c` opens the Chain Inspector for the focused row's correlation id; `t` opens the Ticker
//   Pane for its symbol. Each key is disabled only while ITS OWN overlay is already open, so the
//   two stay independently available — `t` on top of an open Chain Inspector (and vice versa) is
//   exactly how §3.5's slide-over stacking is exercised from the keyboard.
// - Both keys are also gated on `active` (the keep-alive ScreenHost keeps hidden screens mounted
//   with their hotkeys registered — a hidden screen must never open a modal overlay over the
//   visible one).
// - Overlays close the moment the screen deactivates: both slide-overs are portaled Radix
//   Dialogs, so ScreenHost's display:none cannot hide them — left open they would survive
//   navigation, sitting over the next screen with the modal scroll lock still engaged.
// - The Ticker Pane's open ticker is bound to `?ticker=` (§2 "route-addressable as ?ticker=LMT")
//   via useScreenQueryParam, which is itself active-gated so only the visible screen's instance
//   reacts to back/forward.

import { useCallback, useEffect, useRef, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { useScreenQueryParam } from "./useHistoryRoute";

export interface ScreenOverlays {
  /** null = Chain Inspector closed; a correlation id = open on that chain. */
  correlationId: string | null;
  /** null = Ticker Pane closed; a symbol = open on that ticker (synced to `?ticker=`). */
  ticker: string | null;
  openChain: (correlationId: string, rowEl?: HTMLElement | null) => void;
  closeChain: () => void;
  openTicker: (symbol: string, rowEl?: HTMLElement | null) => void;
  closeTicker: () => void;
  /** The row that invoked the current overlay — SlideOver returns focus here on close (§3.5). */
  lastRowRef: React.MutableRefObject<HTMLElement | null>;
  /** True while either slide-over is open — screens gate their own row hotkeys (j/k, a/r, x…)
   *  on this so keys never act on an inert, focus-trapped background (§3.5 overlay behavior). */
  overlayOpen: boolean;
}

export function useScreenOverlays(
  active: boolean,
  focusedRow: {
    /** Correlation id of the currently focused row, or null when none/no lineage. */
    correlationId: () => string | null;
    /** Ticker symbol of the currently focused row, or null when none/unresolved. */
    ticker: () => string | null;
  },
): ScreenOverlays {
  const [correlationId, setCorrelationId] = useState<string | null>(null);
  const [ticker, setTicker] = useScreenQueryParam("ticker", active, "push");
  const lastRowRef = useRef<HTMLElement | null>(null);

  // Fresh closures each render; refs keep the hotkey callbacks stable without stale reads.
  const focusedRef = useRef(focusedRow);
  focusedRef.current = focusedRow;

  const openChain = useCallback((id: string, rowEl?: HTMLElement | null) => {
    if (rowEl) lastRowRef.current = rowEl;
    setCorrelationId(id);
  }, []);

  const closeChain = useCallback(() => setCorrelationId(null), []);

  const openTicker = useCallback(
    (symbol: string, rowEl?: HTMLElement | null) => {
      if (rowEl) lastRowRef.current = rowEl;
      setTicker(symbol);
    },
    [setTicker],
  );

  const closeTicker = useCallback(() => setTicker(null), [setTicker]);

  // Close both overlays the moment this screen is deactivated (see module docblock).
  useEffect(() => {
    if (active) return;
    setCorrelationId(null);
    if (ticker !== null) setTicker(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only re-run on activation transitions
  }, [active]);

  useHotkeys(
    "c",
    () => {
      const id = focusedRef.current.correlationId();
      if (id) setCorrelationId(id);
    },
    { enabled: active && correlationId === null, enableOnFormTags: false },
    [active, correlationId],
  );
  useHotkeys(
    "t",
    () => {
      const symbol = focusedRef.current.ticker();
      if (symbol) setTicker(symbol);
    },
    { enabled: active && ticker === null, enableOnFormTags: false },
    [active, ticker, setTicker],
  );

  return {
    correlationId,
    ticker,
    openChain,
    closeChain,
    openTicker,
    closeTicker,
    lastRowRef,
    overlayOpen: correlationId !== null || ticker !== null,
  };
}
