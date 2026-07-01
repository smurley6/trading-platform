// Shared signal-detail panel: shows the source record that was analyzed for a signal — the
// "Analyzed text" (when the payload carries a text-like field) plus every raw_payload field.
// Reused by the Signal feed (click a row) and the Proposals evidence drill-down (a proposal's
// contributing signals). Pass `onClose` to render a dismiss button; omit it for inline use.

import { Card, H3, Text } from "@salt-ds/core";
import { Fragment } from "react";
import { formatNumber, formatTime } from "../lib/format";
import { color } from "../lib/tokens";
import type { SignalFeedItem } from "../lib/types";

const TEXT_KEYS = [
  "text",
  "title",
  "headline",
  "summary",
  "description",
  "body",
  "statement",
  "content",
];

function analyzedText(rp: Record<string, unknown>): string | null {
  for (const [k, v] of Object.entries(rp)) {
    if (TEXT_KEYS.includes(k.toLowerCase()) && typeof v === "string" && v.trim()) return v;
  }
  return null;
}

function renderValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean" || typeof v === "bigint") return String(v);
  return JSON.stringify(v);
}

interface SignalDetailProps {
  signal: SignalFeedItem;
  onClose?: () => void;
}

export function SignalDetail({ signal, onClose }: SignalDetailProps) {
  const rp = signal.raw_payload ?? {};
  const entries = Object.entries(rp);
  const text = analyzedText(rp);

  return (
    <Card>
      <div
        style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}
      >
        <H3 style={{ marginTop: 0 }}>
          Signal detail — {signal.ticker}{" "}
          <span style={{ color: color.muted, fontWeight: 400, fontSize: 13 }}>
            ({signal.source_type})
          </span>
        </H3>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              cursor: "pointer",
              background: "transparent",
              border: "none",
              color: color.muted,
              fontSize: 18,
              lineHeight: 1,
            }}
          >
            ×
          </button>
        )}
      </div>
      <Text color="secondary" styleAs="label">
        {formatTime(signal.occurred_at)} · sentiment {formatNumber(signal.sentiment, 2)} · weight{" "}
        {formatNumber(signal.politician_weight, 2)} · reliability{" "}
        {formatNumber(signal.source_reliability, 2)}
        {signal.politician_id ? ` · ${signal.politician_id}` : ""}
      </Text>

      {text && (
        <div style={{ marginTop: 12 }}>
          <Text styleAs="label" color="secondary">
            Analyzed text
          </Text>
          <div
            style={{
              marginTop: 4,
              padding: 12,
              borderRadius: 8,
              background: color.panelSoft,
              border: `1px solid ${color.faintBorder}`,
              whiteSpace: "pre-wrap",
              fontSize: 14,
              lineHeight: 1.5,
            }}
          >
            {text}
          </div>
        </div>
      )}

      <div style={{ marginTop: 12 }}>
        <Text styleAs="label" color="secondary">
          Source record analyzed
        </Text>
        {entries.length === 0 ? (
          <Text color="secondary">No source payload was recorded for this signal.</Text>
        ) : (
          <div
            style={{
              marginTop: 6,
              display: "grid",
              gridTemplateColumns: "minmax(120px, max-content) 1fr",
              gap: "4px 16px",
              fontSize: 13,
            }}
          >
            {entries.map(([k, v]) => (
              <Fragment key={k}>
                <span style={{ color: color.muted }}>{k}</span>
                <span style={{ wordBreak: "break-word" }}>{renderValue(v)}</span>
              </Fragment>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
