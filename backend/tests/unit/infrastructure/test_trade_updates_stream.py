"""AlpacaTradeUpdatesStream: wire protocol driven against a fake socket (no network)."""

import asyncio
import json
from decimal import Decimal

import pytest

from trading.application.ports.trade_updates import TradeUpdate
from trading.domain.orders import OrderSide, OrderStatus
from trading.infrastructure.brokers.trade_updates_stream import (
    AlpacaTradeUpdatesError,
    AlpacaTradeUpdatesStream,
)


class FakeSocket:
    """Yields a scripted sequence of inbound frames, then blocks (until cancelled)."""

    def __init__(self, inbound: list[dict[str, object] | list[object]]) -> None:
        self._inbound = [json.dumps(f) for f in inbound]
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self._inbound:
            return self._inbound.pop(0)
        await asyncio.Event().wait()  # block forever; cancellation ends the run
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closed = True


def _order(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "brk-1",
        "client_order_id": "order-abc",
        "symbol": "AAPL",
        "side": "buy",
        "filled_qty": "10",
        "filled_avg_price": "101.5",
    }
    base.update(over)
    return base


async def _collect(stream: AlpacaTradeUpdatesStream, count: int) -> list[TradeUpdate]:
    received: list[TradeUpdate] = []
    done = asyncio.Event()

    async def handler(update: TradeUpdate, *, sleeve_id: str) -> None:
        received.append(update)
        if len(received) >= count:
            done.set()

    task = asyncio.ensure_future(stream.run(handler))
    try:
        await asyncio.wait_for(done.wait(), timeout=1.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return received


async def test_auth_listen_and_dispatch_fill() -> None:
    socket = FakeSocket(
        [
            {"stream": "authorization", "data": {"status": "authorized"}},
            {"stream": "listening", "data": {"streams": ["trade_updates"]}},
            {
                "stream": "trade_updates",
                "data": {
                    "event": "fill",
                    "timestamp": "2026-06-12T14:30:00Z",
                    "price": "101.5",
                    "order": _order(),
                },
            },
        ]
    )

    async def connect(_url: str) -> FakeSocket:
        return socket

    stream = AlpacaTradeUpdatesStream(
        key_id="K",
        secret_key="S",
        url="wss://test/stream",
        sleeve_id="sleeve-political",
        connect=connect,
    )
    updates = await _collect(stream, 1)

    # Sent auth then listen, in order.
    assert json.loads(socket.sent[0])["action"] == "authenticate"
    assert json.loads(socket.sent[1])["action"] == "listen"
    assert len(updates) == 1
    u = updates[0]
    assert u.client_order_id == "order-abc"
    assert u.status is OrderStatus.FILLED
    assert u.side is OrderSide.BUY
    assert u.is_fill
    assert u.fill_price == Decimal("101.5")
    assert u.fill_quantity == Decimal("10")


async def test_rejected_auth_is_fatal_and_does_not_retry() -> None:
    socket = FakeSocket([{"stream": "authorization", "data": {"status": "unauthorized"}}])
    connects = 0

    async def connect(_url: str) -> FakeSocket:
        nonlocal connects
        connects += 1
        return socket

    stream = AlpacaTradeUpdatesStream(
        key_id="K", secret_key="S", url="wss://test/stream", sleeve_id="s", connect=connect
    )

    async def handler(update: TradeUpdate, *, sleeve_id: str) -> None:  # pragma: no cover
        raise AssertionError("no updates expected on rejected auth")

    # run() catches the fatal error internally and returns without retrying.
    await asyncio.wait_for(stream.run(handler), timeout=1.0)
    assert connects == 1
    assert socket.closed is True


async def test_status_only_event_has_no_fill() -> None:
    socket = FakeSocket(
        [
            {"stream": "authorization", "data": {"status": "authorized"}},
            {
                "stream": "trade_updates",
                "data": {
                    "event": "canceled",
                    "timestamp": "2026-06-12T14:30:00Z",
                    "order": _order(filled_qty="0", filled_avg_price=None),
                },
            },
        ]
    )

    async def connect(_url: str) -> FakeSocket:
        return socket

    stream = AlpacaTradeUpdatesStream(
        key_id="K", secret_key="S", url="wss://test/stream", sleeve_id="s", connect=connect
    )
    updates = await _collect(stream, 1)
    assert updates[0].status is OrderStatus.CANCELED
    assert updates[0].is_fill is False


def test_auth_assertion_raises() -> None:
    from trading.infrastructure.brokers.trade_updates_stream import _assert_authorized

    with pytest.raises(AlpacaTradeUpdatesError):
        _assert_authorized({"status": "denied"})
