// Header Paper/Live switch. The PAPER pill is now a segmented toggle, but going LIVE is
// deliberately gated: clicking "Live" does NOT flip anything — it first runs the backend
// live-readiness checks (POST /sleeves/{id}/readiness) and shows them in a panel. Only an
// explicit "Confirm — go live" click flips the sleeves (POST /sleeves/{id}/mode with
// confirm=true), and only when every sleeve reports ready. Switching back to Paper is always
// allowed without confirmation (it's the safe direction). This mirrors the backend gate
// (ALLOW_LIVE + readiness + confirm) so the UI can never silently start real-money trading.

import { Button, Spinner, Text } from "@salt-ds/core";
import { useState } from "react";
import { api } from "../lib/api";
import type { LiveReadiness, Sleeve } from "../lib/types";

interface ModeToggleProps {
  sleeves: Sleeve[];
  onChanged: () => void;
}

export function ModeToggle({ sleeves, onChanged }: ModeToggleProps) {
  const isLive = sleeves.some((s) => s.mode === "live");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [readiness, setReadiness] = useState<LiveReadiness[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const allReady = readiness !== null && readiness.length > 0 && readiness.every((r) => r.ready);

  async function openReadiness(): Promise<void> {
    setBusy(true);
    setError(null);
    setReadiness(null);
    setOpen(true);
    try {
      setReadiness(await Promise.all(sleeves.map((s) => api.liveReadiness(s.id))));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load readiness checks");
    } finally {
      setBusy(false);
    }
  }

  async function confirmLive(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      for (const s of sleeves) await api.setSleeveMode(s.id, "live", true);
      onChanged();
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Go-live was rejected by the backend");
    } finally {
      setBusy(false);
    }
  }

  async function goPaper(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      for (const s of sleeves) await api.setSleeveMode(s.id, "paper", false);
      onChanged();
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not switch to paper");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ position: "relative" }}>
      <div
        role="tablist"
        aria-label="Trading mode"
        style={{
          display: "inline-flex",
          padding: 2,
          borderRadius: 12,
          background: "rgba(148,163,184,0.12)",
          border: "1px solid rgba(148,163,184,0.25)",
        }}
      >
        <Segment
          label="PAPER"
          active={!isLive}
          tone="paper"
          disabled={busy}
          onClick={() => {
            if (isLive) void goPaper();
          }}
        />
        <Segment
          label="LIVE"
          active={isLive}
          tone="live"
          disabled={busy}
          onClick={() => {
            if (!isLive) void openReadiness();
          }}
        />
      </div>

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            left: 0,
            zIndex: 50,
            width: 340,
            padding: 14,
            borderRadius: 10,
            background: "#0f172a",
            border: "1px solid rgba(148,163,184,0.3)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.5)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong style={{ fontSize: 14 }}>Go live — readiness</strong>
            <button
              onClick={() => setOpen(false)}
              style={{
                cursor: "pointer",
                background: "transparent",
                border: "none",
                color: "#94a3b8",
                fontSize: 16,
                lineHeight: 1,
              }}
              aria-label="Close"
            >
              ×
            </button>
          </div>
          <Text
            styleAs="label"
            color="secondary"
            style={{ display: "block", margin: "4px 0 10px" }}
          >
            Live trading uses REAL money. Every check must pass before you can confirm.
          </Text>

          {busy && !readiness ? (
            <Spinner size="small" />
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {(readiness ?? []).map((r) => (
                <div key={r.sleeve_id}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{r.sleeve_id}</span>
                    <span style={{ fontSize: 12, color: r.ready ? "#4ade80" : "#f87171" }}>
                      {r.ready ? "READY" : "NOT READY"}
                    </span>
                  </div>
                  <ul style={{ margin: "6px 0 0", paddingLeft: 0, listStyle: "none" }}>
                    {r.checks.map((c) => (
                      <li
                        key={c.name}
                        style={{ display: "flex", gap: 8, fontSize: 12, padding: "2px 0" }}
                        title={c.detail}
                      >
                        <span style={{ color: c.passed ? "#4ade80" : "#f87171" }}>
                          {c.passed ? "✓" : "✗"}
                        </span>
                        <span style={{ color: "#cbd5e1" }}>{c.name}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}

          {error && (
            <Text color="error" styleAs="label" style={{ display: "block", marginTop: 8 }}>
              {error}
            </Text>
          )}

          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <Button
              variant="cta"
              disabled={busy || !allReady}
              onClick={() => void confirmLive()}
              title={allReady ? "Switch all sleeves to live" : "All checks must pass first"}
            >
              Confirm — go live
            </Button>
            <Button variant="secondary" disabled={busy} onClick={() => setOpen(false)}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function Segment({
  label,
  active,
  tone,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  tone: "paper" | "live";
  disabled: boolean;
  onClick: () => void;
}) {
  const activeBg = tone === "live" ? "rgba(239,68,68,0.22)" : "rgba(76,154,255,0.22)";
  const activeColor = tone === "live" ? "#f87171" : "#7cb3ff";
  const activeBorder = tone === "live" ? "#ef4444" : "#4c9aff";
  return (
    <button
      role="tab"
      aria-selected={active}
      disabled={disabled}
      onClick={onClick}
      style={{
        cursor: disabled ? "default" : "pointer",
        padding: "3px 12px",
        borderRadius: 10,
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: 0.4,
        background: active ? activeBg : "transparent",
        color: active ? activeColor : "#94a3b8",
        border: `1px solid ${active ? activeBorder : "transparent"}`,
      }}
    >
      {label}
    </button>
  );
}
