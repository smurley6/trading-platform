// Animated price cell, two effects layered:
//  1. NumberFlow morphs the digits (rolling/odometer) when the value changes.
//  2. A green (uptick) / red (downtick) background flash via the Web Animations API — the
//     classic trading-terminal cue (the same idea as AG Grid's `enableCellChangeFlash`).
// The flash is a safe no-op under jsdom (where `Element.animate` is absent), so tests don't break.

import NumberFlow from "@number-flow/react";
import { type CSSProperties, useEffect, useRef } from "react";

const FLASH_UP = "rgba(34,197,94,0.45)"; // green
const FLASH_DOWN = "rgba(239,68,68,0.45)"; // red
const FLASH_MS = 700;

export function AnimatedPrice({
  value,
  decimals = 2,
  style,
}: {
  value: number;
  decimals?: number;
  style?: CSSProperties;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const previous = useRef(value);

  useEffect(() => {
    const prev = previous.current;
    previous.current = value;
    if (Number.isNaN(value) || Number.isNaN(prev) || value === prev) return;
    const el = ref.current;
    if (!el || typeof el.animate !== "function") return; // unsupported / jsdom -> no-op
    el.animate(
      [
        { backgroundColor: value > prev ? FLASH_UP : FLASH_DOWN },
        { backgroundColor: "transparent" },
      ],
      { duration: FLASH_MS, easing: "ease-out" },
    );
  }, [value]);

  return (
    <span ref={ref} style={{ borderRadius: 4, padding: "1px 4px", ...style }}>
      {Number.isNaN(value) ? (
        "—"
      ) : (
        <NumberFlow value={value} format={{ maximumFractionDigits: decimals }} />
      )}
    </span>
  );
}
