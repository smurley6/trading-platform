// §3.4 grid cell flash on live update: background pulses --gain/--loss at 12% alpha, 400ms decay,
// per-cell 400ms debounce (§3.5 "Quote flashes debounce per cell — min 400ms between flashes on
// the same cell"). Token-driven sibling of the legacy components/AnimatedPrice.tsx (which
// hardcodes hex/ms and predates the §3 system) — new grids use this instead.
//
// Implemented as a Web Animations API pulse (not a CSS class toggle) so a flash that's still
// decaying can be safely re-triggered without needing to force a reflow between an add/remove of
// the same class name; a no-op under jsdom (no Element.animate) keeps this test-safe.

import { useEffect, useRef } from "react";
import { colors, motionTokens, withAlpha } from "../../lib/tokens";

const FLASH_GAIN = withAlpha(colors.gain, 0.12);
const FLASH_LOSS = withAlpha(colors.loss, 0.12);

export function useCellFlash(value: number, key: string): React.RefObject<HTMLSpanElement> {
  const ref = useRef<HTMLSpanElement>(null);
  const previous = useRef<Map<string, number>>(new Map());
  const lastFlash = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const prev = previous.current.get(key);
    previous.current.set(key, value);
    if (prev === undefined || Number.isNaN(value) || Number.isNaN(prev) || value === prev) return;

    const now = Date.now();
    const last = lastFlash.current.get(key) ?? 0;
    if (now - last < motionTokens.durationFlash) return; // per-cell debounce, §3.5
    lastFlash.current.set(key, now);

    const el = ref.current;
    if (!el || typeof el.animate !== "function") return; // unsupported / jsdom -> no-op
    el.animate(
      [{ backgroundColor: value > prev ? FLASH_GAIN : FLASH_LOSS }, { backgroundColor: "transparent" }],
      { duration: motionTokens.durationFlash, easing: motionTokens.easeOut },
    );
  }, [value, key]);

  return ref;
}
