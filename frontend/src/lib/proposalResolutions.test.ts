import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getResolutions,
  pruneResolutions,
  resetResolutionsForTest,
  resolveProposal,
  subscribeResolutions,
  unresolveProposal,
} from "./proposalResolutions";

beforeEach(() => {
  resetResolutionsForTest();
});

describe("proposalResolutions store", () => {
  it("records approvals and rejections as immutable snapshots", () => {
    const before = getResolutions();
    resolveProposal("LMT", "approved");
    resolveProposal("NOC", "rejected");
    const after = getResolutions();
    expect(after).not.toBe(before); // referential change for useSyncExternalStore
    expect(after.get("LMT")).toBe("approved");
    expect(after.get("NOC")).toBe("rejected");
  });

  it("notifies subscribers on change and supports unsubscribe", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeResolutions(listener);
    resolveProposal("LMT", "approved");
    expect(listener).toHaveBeenCalledTimes(1);
    resolveProposal("LMT", "approved"); // no-op: same resolution
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
    resolveProposal("RTX", "rejected");
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("rolls back with unresolveProposal (§3.5 optimistic rollback)", () => {
    resolveProposal("LMT", "approved");
    unresolveProposal("LMT");
    expect(getResolutions().has("LMT")).toBe(false);
  });

  it("prunes symbols the server no longer proposes, keeping live ones", () => {
    resolveProposal("GONE", "approved");
    resolveProposal("LIVE", "rejected");
    pruneResolutions(new Set(["LIVE"]));
    expect(getResolutions().has("GONE")).toBe(false);
    expect(getResolutions().get("LIVE")).toBe("rejected");
  });
});
