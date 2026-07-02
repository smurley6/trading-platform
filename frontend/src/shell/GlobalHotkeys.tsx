// §4 global keys: `g o / g r / g s / g b / g y` navigate the five mapped destinations, `Ctrl+K`
// opens the command palette, `Esc` closes the topmost overlay. `t` (Ticker Pane) and `c` (Chain
// Inspector) are NOT wired here — both overlays are Phase B/C deliverables (docs/UI-DESIGN.md
// §6, §2); this component only claims the keys Phase A's shell actually owns, so those two
// letters stay free for the later phase's per-row bindings rather than being silently consumed.
//
// react-hotkeys-hook chords: `sequenceSplitKey` defaults to ">", so "g>o" means "press g, then
// o" (sequenceTimeoutMs defaults to 1000ms between the two keys). Disabled while the palette
// itself is open so the two overlay layers never fight over the same keys, and disabled on
// form fields (default `enableOnFormTags: false`) so typing "o" in an input never navigates.

import { useHotkeys } from "react-hotkeys-hook";
import type { ScreenId } from "./useHistoryRoute";

const NAV_CHORDS: { chord: string; screen: ScreenId }[] = [
  { chord: "g>o", screen: "dashboard" }, // Overview
  { chord: "g>r", screen: "proposals" }, // Review
  { chord: "g>s", screen: "signals" }, // Signals
  { chord: "g>b", screen: "blotter" }, // Blotter
  { chord: "g>y", screen: "risk" }, // System
];

export function GlobalHotkeys({
  paletteOpen,
  onNavigate,
  onOpenPalette,
  onEscape,
}: {
  paletteOpen: boolean;
  onNavigate: (screen: ScreenId) => void;
  onOpenPalette: () => void;
  onEscape: () => void;
}) {
  const enabled = !paletteOpen;

  // Hooks must run unconditionally in the same order every render, so all five are always
  // registered; each callback independently no-ops via the shared `enabled` flag.
  useHotkeys(NAV_CHORDS[0]!.chord, () => onNavigate(NAV_CHORDS[0]!.screen), {
    enabled,
    sequenceTimeoutMs: 1000,
  });
  useHotkeys(NAV_CHORDS[1]!.chord, () => onNavigate(NAV_CHORDS[1]!.screen), {
    enabled,
    sequenceTimeoutMs: 1000,
  });
  useHotkeys(NAV_CHORDS[2]!.chord, () => onNavigate(NAV_CHORDS[2]!.screen), {
    enabled,
    sequenceTimeoutMs: 1000,
  });
  useHotkeys(NAV_CHORDS[3]!.chord, () => onNavigate(NAV_CHORDS[3]!.screen), {
    enabled,
    sequenceTimeoutMs: 1000,
  });
  useHotkeys(NAV_CHORDS[4]!.chord, () => onNavigate(NAV_CHORDS[4]!.screen), {
    enabled,
    sequenceTimeoutMs: 1000,
  });

  useHotkeys(
    "ctrl+k,meta+k",
    (e) => {
      e.preventDefault();
      onOpenPalette();
    },
    { enabled: true, preventDefault: true },
  );

  useHotkeys("escape", () => onEscape(), { enabled: true, enableOnFormTags: true });

  return null;
}
