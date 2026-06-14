"""AlpacaQuoteStream against a FAKE websocket — wire protocol + quote parsing, no network.

The fake yields queued messages (auth ack, subscription ack, `T:"q"` arrays) and records the
payloads the stream sends, so we can assert correct auth/subscribe frames and the mapped
`Quote`s without opening a socket. Covers the auth-failure and reconnect-with-backoff paths.
"""

import asyncio
import json
from decimal import Decimal

import pytest

from trading.infrastructure.market_data.alpaca_stream import (
    AlpacaQuoteStream,
    AlpacaQuoteStreamError,
)

_AUTH_OK = json.dumps([{"T": "success", "msg": "authenticated"}])
_SUB_ACK = json.dumps([{"T": "subscription", "quotes": ["AAPL"]}])
_QUOTE = json.dumps(
    [{"T": "q", "S": "AAPL", "bp": 232.10, "ap": 232.14, "t": "2026-06-13T20:00:00.123456789Z"}]
)


class FakeWebSocket:
    """Stand-in for a `websockets` connection: returns queued inbound messages and records
    everything sent. `recv()` blocks forever once the queue drains (a real idle socket)."""

    def __init__(self, inbound: list[str]) -> None:
        self._inbound = asyncio.Queue[str]()
        for message in inbound:
            self._inbound.put_nowait(message)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        return await self._inbound.get()

    async def close(self) -> None:
        self.closed = True


def _stream(ws: FakeWebSocket) -> AlpacaQuoteStream:
    async def connect(_url: str) -> FakeWebSocket:
        return ws

    return AlpacaQuoteStream(key_id="k", secret_key="s", symbols=["AAPL"], connect=connect)


async def test_auth_and_subscribe_payloads_then_emits_quote() -> None:
    ws = FakeWebSocket([_AUTH_OK, _SUB_ACK, _QUOTE])
    stream = _stream(ws)
    queue = stream.subscribe()

    task = asyncio.ensure_future(stream.run())
    quote = await asyncio.wait_for(queue.get(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # First frame is auth; second is the subscribe sent after the auth ack.
    assert json.loads(ws.sent[0]) == {"action": "auth", "key": "k", "secret": "s"}
    assert json.loads(ws.sent[1]) == {"action": "subscribe", "quotes": ["AAPL"]}
    # Quote mapped: midpoint last, string-safe Decimals, tz-aware timestamp.
    assert quote.symbol == "AAPL"
    assert quote.bid == Decimal("232.10")
    assert quote.ask == Decimal("232.14")
    assert quote.last == Decimal("232.12")
    assert quote.as_of.tzinfo is not None
    assert ws.closed is True  # cancelling run() closes the socket


async def test_one_sided_quote_falls_back_to_present_side() -> None:
    frame = json.dumps([{"T": "q", "S": "AAPL", "bp": 0, "ap": 232.14}])
    ws = FakeWebSocket([_AUTH_OK, frame])
    stream = _stream(ws)
    queue = stream.subscribe()

    task = asyncio.ensure_future(stream.run())
    quote = await asyncio.wait_for(queue.get(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert quote.last == Decimal("232.14")


async def test_quote_without_symbol_is_skipped() -> None:
    bad = json.dumps([{"T": "q", "bp": 1, "ap": 2}])
    ws = FakeWebSocket([_AUTH_OK, bad, _QUOTE])
    stream = _stream(ws)
    queue = stream.subscribe()

    task = asyncio.ensure_future(stream.run())
    quote = await asyncio.wait_for(queue.get(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert quote.symbol == "AAPL"  # the malformed frame was skipped, the next one delivered


async def test_add_symbols_sends_subscribe_when_connected() -> None:
    ws = FakeWebSocket([_AUTH_OK])
    stream = _stream(ws)

    task = asyncio.ensure_future(stream.run())
    # Wait until the auth frame is sent (i.e. we are connected).
    for _ in range(100):
        if ws.sent:
            break
        await asyncio.sleep(0.01)
    await stream.add_symbols(["lmt", "AAPL"])  # AAPL already subscribed -> only LMT is new
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    subscribes = [json.loads(s) for s in ws.sent if "subscribe" in s]
    assert {"action": "subscribe", "quotes": ["LMT"]} in subscribes


async def test_auth_failure_is_fatal_no_reconnect() -> None:
    error_frame = json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}])
    connects = 0

    async def connect(_url: str) -> FakeWebSocket:
        nonlocal connects
        connects += 1
        return FakeWebSocket([error_frame])

    stream = AlpacaQuoteStream(key_id="k", secret_key="s", symbols=["AAPL"], connect=connect)
    # run() must return (not raise, not loop) on a fatal auth error, after a single connect.
    await asyncio.wait_for(stream.run(), timeout=1)
    assert connects == 1


async def test_reconnects_with_backoff_after_disconnect() -> None:
    connects = 0

    class DroppingWebSocket(FakeWebSocket):
        async def recv(self) -> str:
            # Authenticates (first message), then the socket dies mid-stream.
            if not self._inbound.empty():
                return await super().recv()
            raise ConnectionError("dropped")

    async def connect(_url: str) -> FakeWebSocket:
        nonlocal connects
        connects += 1
        if connects == 1:
            return DroppingWebSocket([_AUTH_OK])
        return FakeWebSocket([_AUTH_OK, _QUOTE])  # second socket delivers a quote

    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    stream = AlpacaQuoteStream(
        key_id="k",
        secret_key="s",
        symbols=["AAPL"],
        connect=connect,
        sleep=record_sleep,
    )
    queue = stream.subscribe()

    task = asyncio.ensure_future(stream.run())
    quote = await asyncio.wait_for(queue.get(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert connects == 2  # it reconnected after the drop
    assert sleeps and sleeps[0] == 1.0  # backed off before retrying
    assert quote.symbol == "AAPL"


async def test_enabled_is_true() -> None:
    stream = AlpacaQuoteStream(key_id="k", secret_key="s")
    assert stream.enabled is True


async def test_unsubscribe_stops_delivery() -> None:
    ws = FakeWebSocket([_AUTH_OK, _QUOTE])
    stream = _stream(ws)
    queue = stream.subscribe()
    stream.unsubscribe(queue)

    task = asyncio.ensure_future(stream.run())
    await asyncio.sleep(0.05)  # let it process the quote frame
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert queue.empty()


async def test_error_frame_raises_from_reader() -> None:
    # The fatal path surfaces as AlpacaQuoteStreamError inside _connect_and_consume.
    error_frame = json.dumps([{"T": "error", "msg": "bad"}])
    ws = FakeWebSocket([error_frame])
    stream = _stream(ws)
    with pytest.raises(AlpacaQuoteStreamError):
        await stream._connect_and_consume()
