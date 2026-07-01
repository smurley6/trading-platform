// LIVE/OFFLINE status pill for the app header, with a transient "+N" badge that flashes
// when new signal-received events stream in over /ws/events and fades after a few seconds.
//
// Presentational only: the parent owns the single useEventStream subscription and passes in
// `connected` plus a monotonically increasing `signalCount`. Each increment briefly surfaces a
// badge showing how many signals arrived in the latest burst.

import { useEffect, useRef, useState } from "react";

const FADE_MS = 3000;

export function LiveIndicator({
  connected,
  signalCount,
}: {
  connected: boolean;
  signalCount: number;
}) {
  const [burst, setBurst] = useState(0);
  const previous = useRef(signalCount);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    const delta = signalCount - previous.current;
    previous.current = signalCount;
    if (delta <= 0) return; // initial mount or no new signals
    setBurst((b) => b + delta);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setBurst(0), FADE_MS);
  }, [signalCount]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span
        style={{
          padding: "2px 10px",
          borderRadius: 12,
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: 0.4,
          background: connected ? "rgba(34,197,94,0.16)" : "rgba(148,163,184,0.14)",
          color: connected ? "#4ade80" : "#94a3b8",
          border: `1px solid ${connected ? "#22c55e" : "#64748b"}`,
        }}
      >
        {connected ? "LIVE" : "OFFLINE"}
      </span>
      {burst > 0 && (
        <span
          aria-live="polite"
          style={{
            padding: "2px 8px",
            borderRadius: 12,
            fontSize: 12,
            fontWeight: 700,
            background: "rgba(76,154,255,0.2)",
            color: "#7cb3ff",
            border: "1px solid #4c9aff",
            transition: `opacity ${FADE_MS}ms ease-out`,
          }}
        >
          +{burst}
        </span>
      )}
    </div>
  );
}
