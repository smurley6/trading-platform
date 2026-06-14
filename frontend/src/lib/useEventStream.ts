// WebSocket client hook: subscribes to /ws/events and invokes a callback per live event.
// Auto-reconnects with backoff. The callback is held in a ref so re-renders don't churn
// the socket.

import { useEffect, useRef } from "react";
import type { StoredEventDto } from "./types";

function wsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws/events`;
}

export function useEventStream(onEvent: (event: StoredEventDto) => void): void {
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let closed = false;
    let backoffMs = 500;

    const connect = (): void => {
      socket = new WebSocket(wsUrl());
      socket.onopen = () => {
        backoffMs = 500;
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
