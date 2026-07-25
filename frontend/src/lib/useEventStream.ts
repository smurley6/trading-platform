// WebSocket client hook: subscribes to /ws/events and invokes a callback per live event.
// Auto-reconnects with backoff. The callback is held in a ref so re-renders don't churn
// the socket.

import { useEffect, useRef } from "react";
import { apiKey } from "./apiKey";
import type { StoredEventDto } from "./types";
import { wsBase } from "./wsBase";

function wsUrl(): string {
  return `${wsBase()}/ws/events`;
}

// The backend guards /ws/events with the API key, but a browser cannot set request headers on a
// WebSocket — so the key rides in the handshake's `Sec-WebSocket-Protocol` list as exactly
// [sentinel, key] (backend `interface/auth.py::authorize_websocket`). The server selects and
// echoes back only the sentinel, so the key never appears in the response header, the URL, or any
// access log. With no key configured we pass no subprotocol at all and the socket opens as today.
// NOTE: the key must be an HTTP token (hex is safe; base64 `+/=` is not) or the browser rejects
// the handshake — the backend warns about this at startup.
const WS_API_KEY_SUBPROTOCOL = "api-key";

function wsProtocols(): string[] | undefined {
  const key = apiKey();
  return key === null ? undefined : [WS_API_KEY_SUBPROTOCOL, key];
}

/** Tri-state connection status for the status bar's WS dot (§4: green ok / amber
 *  reconnecting / red down — replaces silent failure). "reconnecting" is entered on every
 *  close after the first successful open; "down" is the pre-first-connect / never-connected
 *  state. */
export type ConnectionStatus = "connected" | "reconnecting" | "down";

// Optional second argument: a callback invoked whenever the socket opens (true) or
// closes (false), so callers can surface connection state without owning the socket.
export function useEventStream(
  onEvent: (event: StoredEventDto) => void,
  onConnectionChange?: (connected: boolean) => void,
  onStatusChange?: (status: ConnectionStatus) => void,
): void {
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;
  const statusRef = useRef(onConnectionChange);
  statusRef.current = onConnectionChange;
  const statusDetailRef = useRef(onStatusChange);
  statusDetailRef.current = onStatusChange;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let closed = false;
    let backoffMs = 500;
    let everConnected = false;

    statusDetailRef.current?.("down");

    const connect = (): void => {
      const protocols = wsProtocols();
      socket = protocols === undefined ? new WebSocket(wsUrl()) : new WebSocket(wsUrl(), protocols);
      socket.onopen = () => {
        backoffMs = 500;
        everConnected = true;
        statusRef.current?.(true);
        statusDetailRef.current?.("connected");
      };
      socket.onmessage = (message: MessageEvent<string>) => {
        try {
          handlerRef.current(JSON.parse(message.data) as StoredEventDto);
        } catch {
          // Ignore malformed frames rather than tearing down the stream.
        }
      };
      socket.onclose = () => {
        if (closed) return;
        statusRef.current?.(false);
        statusDetailRef.current?.(everConnected ? "reconnecting" : "down");
        reconnectTimer = window.setTimeout(connect, backoffMs);
        backoffMs = Math.min(backoffMs * 2, 10_000);
      };
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);
}
