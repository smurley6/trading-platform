"""Real-time quote stream over the Alpaca market-data WebSocket
(`wss://stream.data.alpaca.markets/v2/iex`).

Pushes live NBBO updates behind the `/ws/quotes` endpoint, fanning out to per-connection
subscriber queues exactly like `BroadcastingEventStore` does for events. Built in the REST
adapter's style: the same `key`/`secret` auth, `last` mapped to the bid/ask midpoint (so it
agrees with `AlpacaMarketData`), and the wire protocol kept in pure helpers so it is fully
unit-testable against a FAKE websocket — no network.

Protocol (Alpaca v2):
  connect -> send {"action":"auth","key":..,"secret":..}
          -> receive [{"T":"success","msg":"authenticated"}]   (or "error" on bad creds)
          -> send {"action":"subscribe","quotes":[<symbols>]}
          -> receive arrays of frames; quote frames are {"T":"q","S":sym,"bp":bid,"ap":ask,"t":ts}

DORMANT BY DEFAULT: importing this module opens nothing. A socket is opened only when `run()`
is awaited (which `main.py` does on startup ONLY when the feed is enabled and creds resolve).
On disconnect/auth-recoverable failure it reconnects with capped exponential backoff. `run()`
is cancel-safe: cancelling the background task closes the socket and returns cleanly.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from trading.application.ports.market_data import Quote

logger = logging.getLogger(__name__)

_SUBSCRIBER_QUEUE_SIZE = 1000
_STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"


class AlpacaQuoteStreamError(Exception):
    """Non-recoverable stream failure (e.g. authentication rejected by the venue)."""


class _WebSocket(Protocol):
    """The slice of a `websockets` connection this adapter uses (also what the fake implements)."""

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


# A connect callable yields a connected `_WebSocket`. Defaults to `websockets.connect`; tests
# inject a fake so no socket is ever opened.
ConnectFn = Callable[[str], Awaitable[_WebSocket]]


async def _default_connect(url: str) -> _WebSocket:
    # Imported lazily so importing this module never pulls a live connection into existence.
    import websockets

    connection: _WebSocket = await websockets.connect(url)
    return connection


class AlpacaQuoteStream:
    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        symbols: Sequence[str] = (),
        url: str = _STREAM_URL,
        connect: ConnectFn = _default_connect,
        max_backoff_seconds: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._key_id = key_id
        self._secret_key = secret_key
        self._url = url
        self._connect = connect
        self._max_backoff = max_backoff_seconds
        self._sleep = sleep
        self._symbols: set[str] = {s.strip().upper() for s in symbols if s.strip()}
        self._subscribers: set[asyncio.Queue[Quote]] = set()
        self._ws: _WebSocket | None = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return True

    def subscribe(self) -> "asyncio.Queue[Quote]":
        queue: asyncio.Queue[Quote] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[Quote]") -> None:
        self._subscribers.discard(queue)

    async def add_symbols(self, symbols: Sequence[str]) -> None:
        new = {s.strip().upper() for s in symbols if s.strip()} - self._symbols
        if not new:
            return
        self._symbols |= new
        async with self._lock:
            ws = self._ws
            if ws is not None:
                await ws.send(_subscribe_payload(sorted(new)))

    async def run(self) -> None:
        """Connect → auth → subscribe → consume, reconnecting with backoff until cancelled.

        The ONLY method that opens a socket. Safe to cancel: the active connection is closed
        and the coroutine returns. A rejected auth is fatal (no point retrying bad creds);
        transport errors are transient and retried with capped exponential backoff.
        """
        backoff = 1.0
        try:
            while True:
                try:
                    await self._connect_and_consume()
                    backoff = 1.0  # a clean end-of-stream resets the backoff
                except asyncio.CancelledError:
                    raise
                except AlpacaQuoteStreamError:
                    logger.exception("quote stream authentication failed; not retrying")
                    return
                except Exception:
                    logger.warning(
                        "quote stream disconnected; reconnecting in %.1fs", backoff, exc_info=True
                    )
                    await self._sleep(backoff)
                    backoff = min(backoff * 2, self._max_backoff)
        finally:
            await self._close()

    async def _connect_and_consume(self) -> None:
        ws = await self._connect(self._url)
        async with self._lock:
            self._ws = ws
        try:
            await ws.send(_auth_payload(self._key_id, self._secret_key))
            async for quote in self._read_quotes(ws):
                self._publish(quote)
        finally:
            async with self._lock:
                if self._ws is ws:
                    self._ws = None
            await _safe_close(ws)

    async def _read_quotes(self, ws: _WebSocket) -> AsyncIterator[Quote]:
        """Drive the wire protocol, yielding mapped `Quote`s. Sends the subscribe frame once
        auth is acknowledged. Raises `AlpacaQuoteStreamError` on an auth/error frame."""
        while True:
            raw = await ws.recv()
            for frame in _decode_frames(raw):
                kind = frame.get("T")
                if kind == "error":
                    raise AlpacaQuoteStreamError(f"stream error frame: {frame.get('msg', frame)}")
                if kind == "success" and frame.get("msg") == "authenticated":
                    if self._symbols:
                        await ws.send(_subscribe_payload(sorted(self._symbols)))
                    continue
                if kind == "q":
                    quote = _frame_to_quote(frame)
                    if quote is not None:
                        yield quote
                # subscription acks ("T":"subscription") and other control frames: ignore.

    def _publish(self, quote: Quote) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(quote)
            except asyncio.QueueFull:
                logger.warning("dropping quote for a slow websocket subscriber")

    async def _close(self) -> None:
        async with self._lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            await _safe_close(ws)


async def _safe_close(ws: _WebSocket) -> None:
    try:
        await ws.close()
    except Exception:  # pragma: no cover - close best-effort
        logger.debug("error closing quote stream socket", exc_info=True)


def _auth_payload(key_id: str, secret_key: str) -> str:
    return json.dumps({"action": "auth", "key": key_id, "secret": secret_key})


def _subscribe_payload(symbols: Sequence[str]) -> str:
    return json.dumps({"action": "subscribe", "quotes": list(symbols)})


def _decode_frames(raw: str | bytes) -> list[dict[str, Any]]:
    """Alpaca sends each message as a JSON ARRAY of frames; tolerate a bare object too."""
    text = raw.decode() if isinstance(raw, bytes) else raw
    payload = json.loads(text)
    if isinstance(payload, list):
        return [f for f in payload if isinstance(f, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _frame_to_quote(frame: dict[str, Any]) -> Quote | None:
    symbol = frame.get("S")
    if not symbol:
        return None
    bid = _decimal(frame.get("bp"))
    ask = _decimal(frame.get("ap"))
    return Quote(
        symbol=str(symbol),
        bid=bid,
        ask=ask,
        last=_mid(bid, ask),
        as_of=_parse_time(frame.get("t")),
    )


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


def _mid(bid: Decimal, ask: Decimal) -> Decimal:
    if bid > 0 and ask > 0:
        return ((bid + ask) / 2).quantize(Decimal("0.0001"))
    return ask if ask > 0 else bid


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    # Alpaca timestamps are RFC-3339 with nanoseconds; trim to microseconds for fromisoformat.
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        frac = tail
        offset = ""
        for marker in ("+", "-"):
            if marker in tail:
                frac, _, rest = tail.partition(marker)
                offset = marker + rest
                break
        text = f"{head}.{frac[:6]}{offset}"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
