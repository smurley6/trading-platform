// Thin always-visible status bar under the header: at a glance, is the pipeline fresh and is the
// book armed? Fed entirely by data the app already loads — the live WebSocket connection state,
// the signal feed (count + most-recent time), the cycle mode, sleeve mode, and kill switch — so
// it needs no new backend endpoint.

import { api } from "../lib/api";
import { color } from "../lib/tokens";
import type { Sleeve } from "../lib/types";
import { useLoader } from "../lib/useLoader";

interface StatusStripProps {
  connected: boolean;
  sleeves: Sleeve[];
  killEngaged: boolean;
  refreshKey: number;
}

function agoLabel(iso: string | undefined): string {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

export function StatusStrip({ connected, sleeves, killEngaged, refreshKey }: StatusStripProps) {
  const signals = useLoader(() => api.signalFeed(300), [refreshKey]);
  const proposals = useLoader(() => api.proposals(), [refreshKey]);

  const feed = signals.data ?? [];
  const lastSignal = feed.reduce<string | undefined>(
    (latest, s) => (!latest || s.occurred_at > latest ? s.occurred_at : latest),
    undefined,
  );
  const live = sleeves.some((s) => s.mode === "live");
  const cycleMode = proposals.data?.mode ?? "—";
  const proposalCount = proposals.data?.proposals.length ?? 0;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        flexWrap: "wrap",
        padding: "5px 20px",
        borderBottom: `1px solid ${color.hairline}`,
        fontSize: 12,
        color: color.muted,
      }}
    >
      <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: connected ? color.ok : color.dangerText,
            boxShadow: connected ? `0 0 6px ${color.ok}` : "none",
          }}
        />
        {connected ? "Live feed" : "Disconnected"}
      </span>
      <Sep />
      <Item label="last signal" value={agoLabel(lastSignal)} />
      <Sep />
      <Item label="signals" value={String(feed.length)} />
      <Sep />
      <Item label="cycle" value={cycleMode} />
      <Sep />
      <Item label="proposals" value={String(proposalCount)} />
      <Sep />
      <strong style={{ color: live ? color.dangerText : color.accentText }}>
        {live ? "LIVE" : "PAPER"}
      </strong>
      <Sep />
      <strong style={{ color: killEngaged ? color.dangerText : color.ok }}>
        {killEngaged ? "HALTED" : "ARMED"}
      </strong>
    </div>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <span>
      {label} <span style={{ color: color.textDim, fontWeight: 600 }}>{value}</span>
    </span>
  );
}

function Sep() {
  return <span style={{ color: color.hairline }}>·</span>;
}
