// Global kill switch (FR-RISK-3, FR-UI-6). Built on Salt; toggling posts to the backend,
// which records it as an immutable user-action event.

import { Button } from "@salt-ds/core";

interface KillSwitchControlProps {
  engaged: boolean;
  busy: boolean;
  onToggle: (engaged: boolean) => void;
}

export function KillSwitchControl({ engaged, busy, onToggle }: KillSwitchControlProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span
        style={{
          padding: "2px 10px",
          borderRadius: 12,
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: 0.4,
          background: engaged ? "rgba(239,68,68,0.18)" : "rgba(34,197,94,0.16)",
          color: engaged ? "#f87171" : "#4ade80",
          border: `1px solid ${engaged ? "#ef4444" : "#22c55e"}`,
        }}
      >
        {engaged ? "ORDERS HALTED" : "TRADING ARMED"}
      </span>
      <Button disabled={busy} onClick={() => onToggle(!engaged)}>
        {engaged ? "Release kill switch" : "Engage kill switch"}
      </Button>
    </div>
  );
}
