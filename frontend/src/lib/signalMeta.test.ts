import { describe, expect, it } from "vitest";
import { signalMetaFromEvents } from "./signalMeta";
import type { StoredEventDto } from "./types";

let seq = 0;
function evt(
  event_type: string,
  payload: Record<string, unknown>,
  correlation_id: string,
): StoredEventDto {
  seq += 1;
  return {
    event_id: `e${seq}`,
    stream_id: `signal-${seq}`,
    seq,
    event_type,
    schema_version: 1,
    occurred_at: "2026-07-01T00:00:00.000Z",
    recorded_at: "2026-07-01T00:00:00.000Z",
    correlation_id,
    payload,
  };
}

describe("signalMetaFromEvents", () => {
  it("indexes signal.received events by payload signal_id with provenance", () => {
    const meta = signalMetaFromEvents([
      evt("signal.received", { signal_id: "s1", provenance: "BACKFILLED" }, "corr-1"),
      evt("order.intent", { client_order_id: "o1" }, "corr-2"),
    ]);
    expect(meta.get("s1")).toEqual({ correlationId: "corr-1", provenance: "BACKFILLED" });
    expect(meta.size).toBe(1);
  });

  it("treats unrecognized or missing provenance as null (unknown), never REAL", () => {
    const meta = signalMetaFromEvents([
      evt("signal.received", { signal_id: "s2" }, "corr-3"),
      evt("signal.received", { signal_id: "s3", provenance: "bogus" }, "corr-4"),
    ]);
    expect(meta.get("s2")).toEqual({ correlationId: "corr-3", provenance: null });
    expect(meta.get("s3")).toEqual({ correlationId: "corr-4", provenance: null });
  });

  it("normalizes provenance casing and skips events without a signal_id", () => {
    const meta = signalMetaFromEvents([
      evt("signal.received", { signal_id: "s4", provenance: "real" }, "corr-5"),
      evt("signal.received", { provenance: "REAL" }, "corr-6"),
    ]);
    expect(meta.get("s4")).toEqual({ correlationId: "corr-5", provenance: "REAL" });
    expect(meta.size).toBe(1);
  });
});
