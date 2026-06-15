// Real-time numeric quote hook for the Markets watchlist and Dashboard live P&L.
//
// Prefers a WebSocket stream (`/ws/quotes`, proxied to the backend like `/ws/events`); the
// server announces feed availability with a `status` frame, then pushes `quote` frames as
// prices move. If the socket can't connect, the hook falls back to interval polling of the
// REST `/quotes` endpoint — but ONLY while the feed is actually enabled, so a disabled feed
// stays completely dormant (no sockets churning, no wasted polls): exactly today's behaviour.
//
// Numeric fields stay as strings (matching the `Quote` DTO); render with Number()/format().

import { useEffect, useState } from "react";
import { api } from "./api";
import type { Quote } from "./types";

const POLL_INTERVAL_MS = 4_000;
// Throttle the live stream's render cadence: apply buffered ticks to state twice a second.
const FLUSH_INTERVAL_MS = 500;
const MAX_BACKOFF_MS = 10_000;
const INITIAL_BACKOFF_MS = 500;

// --- Wire formats (server -> client). Narrow shapes; no `any`. ---------------------------

interface StatusFrame {
  type: "status";
  enabled: boolean;
}

interface QuoteFrame {
  type: "quote";
  symbol: string;
  bid: string;
  ask: string;
  last: string;
  as_of: string;
}

type ServerFrame = StatusFrame | QuoteFrame;

function parseFrame(data: string): ServerFrame | null {
  try {
    const raw = JSON.parse(data) as unknown;
    if (typeof raw !== "object" || raw === null) return null;
    const frame = raw as { type?: unknown };
    if (frame.type === "status") {
      const s = raw as { enabled?: unknown };
      if (typeof s.enabled === "boolean") return { type: "status", enabled: s.enabled };
      return null;
    }
    if (frame.type === "quote") {
      const q = raw as Record<string, unknown>;
      if (
        typeof q.symbol === "string" &&
        typeof q.bid === "string" &&
        typeof q.ask === "string" &&
        typeof q.last === "string" &&
        typeof q.as_of === "string"
      ) {
        return {
          type: "quote",
          symbol: q.symbol,
          bid: q.bid,
          ask: q.ask,
          last: q.last,
          as_of: q.as_of,
        };
      }
    }
    return null;
  } catch {
    return null;
  }
}

function quotesWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws/quotes`;
}

export interface QuoteStream {
  /** True only when a live feed (stream or REST) reports the quote feed is enabled. */
  enabled: boolean;
  /** Latest quote per symbol, updated live; empty while the feed is off. */
  quotes: Map<string, Quote>;
  /** True while the WebSocket is open (false when relying on the REST poll fallback). */
  connected: boolean;
}

// Tracks the latest quote per symbol, updated in real time from the stream (or poll fallback).
// Re-running on a changed symbol set tears down and rebuilds the socket/interval cleanly.
export function useQuoteStream(symbols: readonly string[]): QuoteStream {
  const [enabled, setEnabled] = useState(false);
  const [connected, setConnected] = useState(false);
  const [quotes, setQuotes] = useState<Map<string, Quote>>(() => new Map());

  // Stable key so the effect only re-runs when the actual symbol set changes (not on every
  // render that produces a new array identity).
  const symbolsKey = symbols.join(",");

  useEffect(() => {
    const symbolList = symbolsKey ? symbolsKey.split(",") : [];

    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let pollTimer: number | undefined;
    let backoffMs = INITIAL_BACKOFF_MS;
    // Tracks whether the live stream is the active source; while true we never poll, to avoid
    // double-sourcing the same quotes.
    let streaming = false;

    // Coalesce the high-frequency stream: buffer the newest quote per symbol and flush to React
    // state at ~1 Hz, so the watchlist updates smoothly once a second instead of flickering on
    // every raw tick. No flush happens in a second with no ticks (no wasted re-render).
    const pending = new Map<string, Quote>();
    let flushTimer: number | undefined;

    const flushPending = (): void => {
      if (disposed || pending.size === 0) return;
      setQuotes((prev) => {
        const next = new Map(prev);
        for (const [sym, q] of pending) next.set(sym, q);
        return next;
      });
      pending.clear();
    };
    const startFlush = (): void => {
      if (flushTimer === undefined) flushTimer = window.setInterval(flushPending, FLUSH_INTERVAL_MS);
    };
    const clearFlush = (): void => {
      if (flushTimer !== undefined) {
        window.clearInterval(flushTimer);
        flushTimer = undefined;
      }
      pending.clear();
    };

    const clearPoll = (): void => {
      if (pollTimer !== undefined) {
        window.clearInterval(pollTimer);
        pollTimer = undefined;
      }
    };

    // REST fallback: poll only while the feed is enabled. The very first response decides
    // whether we keep polling at all — a disabled feed stops immediately and stays dormant.
    const pollOnce = (): void => {
      if (disposed || streaming) return;
      void api.quotes(symbolList.slice()).then((result) => {
        if (disposed || streaming) return;
        if (!result.enabled) {
          setEnabled(false);
          setQuotes(new Map());
          clearPoll(); // Feed is off — go dormant, no further polling.
          return;
        }
        setEnabled(true);
        const next = new Map<string, Quote>();
        for (const q of result.quotes) next.set(q.symbol, q);
        setQuotes(next);
      });
    };

    const startPolling = (): void => {
      if (disposed || streaming || pollTimer !== undefined || symbolList.length === 0) return;
      pollOnce();
      pollTimer = window.setInterval(pollOnce, POLL_INTERVAL_MS);
    };

    const scheduleReconnect = (): void => {
      if (disposed) return;
      reconnectTimer = window.setTimeout(connect, backoffMs);
      backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS);
    };

    function connect(): void {
      if (disposed) return;
      let opened = false;
      try {
        socket = new WebSocket(quotesWsUrl());
      } catch {
        // Construction can throw on a bad URL; fall back to REST polling.
        setConnected(false);
        startPolling();
        return;
      }

      socket.onopen = () => {
        opened = true;
        backoffMs = INITIAL_BACKOFF_MS;
        setConnected(true);
        clearPoll(); // The socket is the source of truth now; stop any fallback polling.
        if (symbolList.length > 0) {
          socket?.send(JSON.stringify({ action: "subscribe", symbols: symbolList }));
        }
      };

      socket.onmessage = (message: MessageEvent<string>) => {
        const frame = parseFrame(message.data);
        if (frame === null) return;
        if (frame.type === "status") {
          setEnabled(frame.enabled);
          if (frame.enabled) {
            // Live stream is the active source: drop the poll, start the 1 Hz flush.
            streaming = true;
            clearPoll();
            startFlush();
          } else {
            // Feed reported off: stay dormant exactly like today (charts-only / FEED OFF).
            streaming = false;
            clearFlush();
            setQuotes(new Map());
          }
          return;
        }
        // frame.type === "quote": buffer the newest value per symbol; `flushPending` applies it
        // to state at ~1 Hz so the UI updates once a second, not on every raw tick.
        pending.set(frame.symbol, {
          symbol: frame.symbol,
          bid: frame.bid,
          ask: frame.ask,
          last: frame.last,
          as_of: frame.as_of,
        });
      };

      socket.onclose = () => {
        setConnected(false);
        streaming = false;
        clearFlush();
        socket = null;
        if (disposed) return;
        if (opened) {
          // We had a working socket; reconnect promptly without poking REST.
          scheduleReconnect();
        } else {
          // Never connected (no backend WS / feed unavailable): try REST polling, which will
          // itself go dormant if the feed is disabled, and keep retrying the socket slowly.
          startPolling();
          scheduleReconnect();
        }
      };

      socket.onerror = () => {
        // onclose handles teardown/reconnect; nothing to do here.
      };
    }

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      clearPoll();
      clearFlush();
      if (socket) {
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
      }
      // Reset for the next symbol set so stale quotes never flash.
      setConnected(false);
    };
  }, [symbolsKey]);

  return { enabled, quotes, connected };
}
