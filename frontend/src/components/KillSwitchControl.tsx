// Global kill switch (FR-RISK-3, FR-UI-6). Built on Salt; toggling posts to the backend,
// which records it as an immutable user-action event.

import { Button } from "@salt-ds/core";
import { color, pillStyle } from "../lib/tokens";

interface KillSwitchControlProps {
  engaged: boolean;
  busy: boolean;
  onToggle: (engaged: boolean) => void;
}

export function KillSwitchControl({ engaged, busy, onToggle }: KillSwitchControlProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={pillStyle(engaged ? "danger" : "ok")}>
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: engaged ? color.dangerText : color.ok,
            boxShadow: `0 0 6px ${engaged ? color.dangerText : color.ok}`,
          }}
        />
        {engaged ? "ORDERS HALTED" : "TRADING ARMED"}
      </span>
      <Button disabled={busy} onClick={() => onToggle(!engaged)}>
        {engaged ? "Release kill switch" : "Engage kill switch"}
      </Button>
    </div>
  );
}
