import { describe, expect, it } from "vitest";
import type { StoredEventDto } from "../../lib/types";
import { killSwitchHistory, reconciliationSummary } from "./systemData";

function event(partial: Partial<StoredEventDto> & { event_type: string }): StoredEventDto {
  return {
    event_id: crypto.randomUUID(),
    stream_id: "s",
    seq: 0,
    schema_version: 1,
    occurred_at: "2026-07-24T10:00:00Z",
    recorded_at: "2026-07-24T10:00:00Z",
    correlation_id: "c",
    payload: {},
    ...partial,
  };
}

describe("killSwitchHistory", () => {
  it("extracts engaged/released changes newest-first", () => {
    const history = killSwitchHistory([
      event({ event_type: "risk.kill_switch_changed", payload: { engaged: true }, occurred_at: "2026-07-24T10:00:00Z" }),
      event({ event_type: "order.filled" }),
      event({ event_type: "risk.kill_switch_changed", payload: { engaged: false }, occurred_at: "2026-07-24T11:00:00Z" }),
    ]);
    expect(history.map((h) => h.engaged)).toEqual([false, true]); // newest-first
    expect(history[0]?.time).toBe("2026-07-24T11:00:00Z");
  });

  it("returns empty when no kill-switch events", () => {
    expect(killSwitchHistory([event({ event_type: "order.filled" })])).toEqual([]);
  });
});

describe("reconciliationSummary", () => {
  it("returns a null last-run when reconciliation never ran", () => {
    const summary = reconciliationSummary([event({ event_type: "order.filled" })]);
    expect(summary.lastRunAt).toBeNull();
    expect(summary.drifts).toEqual([]);
  });

  it("reports the latest run's completed stats + only that run's drift rows", () => {
    const summary = reconciliationSummary([
      // An older run with a drift that must NOT leak into the latest run's drift list.
      event({ event_type: "reconciliation.drift", correlation_id: "run-1", payload: { kind: "position", key: "OLD", local: "1", broker: "0" } }),
      event({ event_type: "reconciliation.completed", correlation_id: "run-1", payload: { drift_count: 1, checked_positions: 2, checked_orders: 3 } }),
      // The latest run: two drifts under run-2.
      event({ event_type: "reconciliation.drift", correlation_id: "run-2", payload: { kind: "position", key: "AAPL", local: "5", broker: "4" } }),
      event({ event_type: "reconciliation.drift", correlation_id: "run-2", payload: { kind: "order", key: "coid-1", local: "filled", broker: "missing" } }),
      event({ event_type: "reconciliation.completed", correlation_id: "run-2", occurred_at: "2026-07-24T12:00:00Z", payload: { drift_count: 2, checked_positions: 6, checked_orders: 7 } }),
    ]);
    expect(summary.lastRunAt).toBe("2026-07-24T12:00:00Z");
    expect(summary.driftCount).toBe(2);
    expect(summary.checkedPositions).toBe(6);
    expect(summary.drifts.map((d) => d.key)).toEqual(["AAPL", "coid-1"]);
  });
});
