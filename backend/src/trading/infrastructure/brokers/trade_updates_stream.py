"""Alpaca trade-updates WebSocket adapter (FR-REC, FR-EXE, spec §11).

Real-time account stream for the paper venue. Connects to the paper trading WebSocket
(``wss://paper-api.alpaca.markets/stream``), authenticates, listens to ``trade_updates`` and
dispatches each decoded :class:`TradeUpdate` to a handler the moment the broker emits it —
the push complement to the 30s polling order-sync.

Built in the quote-stream adapter's style: the same ``_WebSocket`` slice, a lazily-imported
``_default_connect`` so importing this module opens nothing, capped exponential-backoff
reconnects, and a pure wire-protocol layer kept fully unit-testable against a fake socket.

Protocol (Alpaca v2 account/trade stream):
  connect -> send {"action":"authenticate","data":{"key_id":..,"secret_key":..}}
          -> receive {"stream":"authorization","data":{"status":"authorized"}}  (else fatal)
          -> send {"action":"listen","data":{"streams":["trade_updates"]}}
          -> receive {"stream":"trade_updates","data":{"event":..,"order":{..},..}}

DORMANT BY DEFAULT: a socket is opened only when ``run()`` is awaited, which ``main.py`` does
on startup ONLY when the trade-updates stream is enabled and paper creds resolve. ``run()`` is
cancel-safe: cancelling the background task closes the socket and returns cleanly.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from trading.application.ports.trade_updates import TradeUpdate, TradeUpdateHandler
from trading.domain.orders import OrderSide, OrderStatus

logger = logging.getLogger(__name__)

# Alpaca trade-update event -> our lifecycle status. Mirrors the REST status map's intent.
_EVENT_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.ACKNOWLEDGED,
    "pending_new": OrderStatus.SUBMITTED,
    "partial_fill": OrderStatus.PARTIALLY_FILLED,
    "fill": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "done_for_day": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "rejected": OrderStatus.REJECTED,
    "stopped": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
}
_FILL_EVENTS = frozenset({"fill", "partial_fill"})


class AlpacaTradeUpdatesError(Exception):
    """Non-recoverable stream failure (e.g. authentication rejected by the venue)."""


class _WebSocket(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


ConnectFn = Callable[[str], Awaitable[_WebSocket]]


async def _default_connect(url: str) -> _WebSocket:
    # Lazy import so importing this module never pulls a live connection into existence.
    import websockets

    connection: _WebSocket = await websockets.connect(url)
    return connection


class AlpacaTradeUpdatesStream:
    def __init__(
        self,
        *,
        key_id: str,
        secret_key: str,
        url: str,
        sleeve_id: str,
        connect: ConnectFn = _default_connect,
        max_backoff_seconds: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._key_id = key_id
        self._secret_key = secret_key
        self._url = url
        self._sleeve_id = sleeve_id
        self._connect = connect
        self._max_backoff = max_backoff_seconds
        self._sleep = sleep
        self._ws: _WebSocket | None = None

    @property
    def enabled(self) -> bool:
        return True

    async def run(self, handler: TradeUpdateHandler) -> None:
        """Connect → auth → listen → dispatch, reconnecting with backoff until cancelled.

        The ONLY method that opens a socket. Safe to cancel. A rejected auth is fatal; transport
        errors are transient and retried with capped exponential backoff.
        """
        backoff = 1.0
        try:
            while True:
                try:
                    await self._connect_and_consume(handler)
                    backoff = 1.0
                except asyncio.CancelledError:
                    raise
                except AlpacaTradeUpdatesError:
                    logger.exception("trade-updates stream authentication failed; not retrying")
                    return
                except Exception:
                    logger.warning(
                        "trade-updates stream disconnected; reconnecting in %.1fs",
                        backoff,
                        exc_info=True,
                    )
                    await self._sleep(backoff)
                    backoff = min(backoff * 2, self._max_backoff)
        finally:
            await self._close()

    async def _connect_and_consume(self, handler: TradeUpdateHandler) -> None:
        ws = await self._connect(self._url)
        self._ws = ws
        try:
            await ws.send(_auth_payload(self._key_id, self._secret_key))
            await self._consume(ws, handler)
        finally:
            if self._ws is ws:
                self._ws = None
            await _safe_close(ws)

    async def _consume(self, ws: _WebSocket, handler: TradeUpdateHandler) -> None:
        while True:
            raw = await ws.recv()
            for frame in _decode_frames(raw):
                stream = frame.get("stream")
                data = frame.get("data")
                if stream == "authorization":
                    _assert_authorized(data)
                    await ws.send(_listen_payload())
                    continue
                if stream == "trade_updates" and isinstance(data, dict):
                    update = _frame_to_update(data)
                    if update is not None:
                        await handler(update, sleeve_id=self._sleeve_id)
                # listening acks and other control frames: ignore.

    async def _close(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            await _safe_close(ws)


async def _safe_close(ws: _WebSocket) -> None:
    try:
        await ws.close()
    except Exception:  # pragma: no cover - close best-effort
        logger.debug("error closing trade-updates socket", exc_info=True)


def _auth_payload(key_id: str, secret_key: str) -> str:
    return json.dumps(
        {"action": "authenticate", "data": {"key_id": key_id, "secret_key": secret_key}}
    )


def _listen_payload() -> str:
    return json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}})


def _assert_authorized(data: Any) -> None:
    status = data.get("status") if isinstance(data, dict) else None
    if status != "authorized":
        raise AlpacaTradeUpdatesError(f"trade-updates auth rejected: {data}")


def _decode_frames(raw: str | bytes) -> list[dict[str, Any]]:
    """Alpaca account-stream messages are single JSON objects; tolerate an array too."""
    text = raw.decode() if isinstance(raw, bytes) else raw
    payload = json.loads(text)
    if isinstance(payload, list):
        return [f for f in payload if isinstance(f, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _frame_to_update(data: dict[str, Any]) -> TradeUpdate | None:
    event = str(data.get("event", ""))
    order = data.get("order")
    if not isinstance(order, dict):
        return None
    client_order_id = order.get("client_order_id")
    if not client_order_id:
        return None
    status = _EVENT_STATUS_MAP.get(event)
    if status is None:
        return None  # an event we do not record (e.g. order_replace_rejected)
    side = _side(order.get("side"))
    if side is None:
        return None
    occurred_at = _parse_time(data.get("timestamp"))
    fill_price: Decimal | None = None
    fill_quantity: Decimal | None = None
    filled_at: datetime | None = None
    if event in _FILL_EVENTS:
        # Cumulative quantity at this execution; the use case records only the new increment.
        fill_quantity = _decimal(order.get("filled_qty") or data.get("position_qty"))
        fill_price = _decimal(data.get("price") or order.get("filled_avg_price"))
        filled_at = _parse_time(order.get("filled_at") or data.get("timestamp"))
        if fill_price is None or fill_quantity is None or fill_quantity <= 0:
            # A fill event without usable execution detail: record the status, not a bad fill.
            fill_price = None
            fill_quantity = None
            filled_at = None
    return TradeUpdate(
        client_order_id=str(client_order_id),
        broker_order_id=str(order.get("id", "")),
        event=event,
        status=status,
        symbol=str(order.get("symbol", "")),
        side=side,
        occurred_at=occurred_at,
        fill_price=fill_price,
        fill_quantity=fill_quantity,
        filled_at=filled_at,
    )


def _side(value: Any) -> OrderSide | None:
    try:
        return OrderSide(str(value))
    except ValueError:
        return None


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
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
