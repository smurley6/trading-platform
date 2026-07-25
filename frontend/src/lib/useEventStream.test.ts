// The /ws/events handshake carries the API key as a Sec-WebSocket-Protocol token, because a
// browser cannot set request headers on a WebSocket (backend task-22). These tests pin that
// contract: exactly [sentinel, key] when a key is configured, and NO subprotocol argument at all
// when it isn't (so the socket opens exactly as it does in today's unauthenticated deployment).

import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useEventStream } from "./useEventStream";

type Ctor = [url: string, protocols?: string | string[]];

function captureSockets(): Ctor[] {
  const calls: Ctor[] = [];
  class FakeSocket {
    onopen: (() => void) | null = null;
    onmessage: ((event: MessageEvent<string>) => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: (() => void) | null = null;
    constructor(url: string, protocols?: string | string[]) {
      calls.push(protocols === undefined ? [url] : [url, protocols]);
    }
    close(): void {}
  }
  vi.stubGlobal("WebSocket", FakeSocket);
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("useEventStream handshake", () => {
  it("offers [sentinel, key] as subprotocols when an API key is configured", () => {
    vi.stubEnv("VITE_API_KEY", "abc123def456");
    const calls = captureSockets();

    renderHook(() => useEventStream(() => {}));

    expect(calls).toHaveLength(1);
    const call = calls[0];
    if (call === undefined) throw new Error("expected a socket to be constructed");
    const [url, protocols] = call;
    expect(url).toContain("/ws/events");
    expect(protocols).toEqual(["api-key", "abc123def456"]);
    // The key must never ride in the URL — it would land in access logs.
    expect(url).not.toContain("abc123def456");
  });

  it("passes no subprotocol argument when no API key is configured", () => {
    vi.stubEnv("VITE_API_KEY", "");
    const calls = captureSockets();

    renderHook(() => useEventStream(() => {}));

    expect(calls).toHaveLength(1);
    const call = calls[0];
    if (call === undefined) throw new Error("expected a socket to be constructed");
    expect(call).toHaveLength(1);
    expect(call[1]).toBeUndefined();
  });
});
