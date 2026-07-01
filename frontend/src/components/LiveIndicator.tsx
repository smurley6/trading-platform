// LIVE/OFFLINE status pill for the app header, with a transient "+N" badge that flashes
// when new signal-received events stream in over /ws/events and fades after a few seconds.
//
// Presentational only: the parent owns the single useEventStream subscription and passes in
// `connected` plus a monotonically increasing `signalCount`. Each increment briefly surfaces a
// badge showing how many signals arrived in the latest burst.

import { useEffect, useRef, useState } from "react";
import { color, pillStyle } from "../lib/tokens";

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
      <span style={pillStyle(connected ? "ok" : "muted")}>
        {connected && (
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: color.ok,
              boxShadow: `0 0 6px ${color.ok}`,
            }}
          />
        )}
        {connected ? "LIVE" : "OFFLINE"}
      </span>
      {burst > 0 && (
        <span
          aria-live="polite"
          style={{ ...pillStyle("accent"), transition: `opacity ${FADE_MS}ms ease-out` }}
        >
          +{burst}
        </span>
      )}
    </div>
  );
}
