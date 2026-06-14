"""The /ws/quotes WebSocket streams live quotes (or politely reports disabled).

Drives the app through a FAKE QuoteStreamPort injected via AppDeps — no network — covering
the status frame in both enabled and disabled states, that pushed quotes arrive as
{"type":"quote",...}, and that a client {"action":"subscribe",...} reaches the stream.
Mirrors tests/e2e/test_websocket.py + tests/e2e/test_quotes_api.py.
"""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from starlette.testclient import TestClient

from tests.conftest import FixedClock
from trading.application.ports.market_data import Quote
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.interface.app import AppDeps, create_app

_AS_OF = datetime(2026, 6, 13, 20, 0, tzinfo=UTC)


class FakeQuoteStream:
    """In-process QuoteStreamPort: push() fans a quote to every subscriber, exactly like the
    real adapter, but never opens a socket. Records the symbols clients subscribe to."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._subscribers: set[asyncio.Queue[Quote]] = set()
        self.added: list[list[str]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def subscribe(self) -> "asyncio.Queue[Quote]":
        queue: asyncio.Queue[Quote] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[Quote]") -> None:
        self._subscribers.discard(queue)

    async def add_symbols(self, symbols: Sequence[str]) -> None:
        self.added.append([s.upper() for s in symbols])

    def push(self, quote: Quote) -> None:
        for queue in self._subscribers:
            queue.put_nowait(quote)


def _deps(clock: FixedClock, stream: FakeQuoteStream | None, watchlist: list[str]) -> AppDeps:
    return AppDeps(
        event_store=InMemoryEventStore(clock),
        kill_switch=KillSwitch(),
        sleeve_registry=SleeveRegistry([]),
        evaluators={},
        default_sleeve_id="sleeve-political",
        reconcilers=[],
        clock=clock,
        conviction_thresholds=ConvictionThresholds(),
        conviction_window_days=21,
        quote_stream=stream,
        default_watchlist=watchlist,
    )


def test_quotes_ws_disabled_sends_status_and_no_quotes() -> None:
    app = create_app(_deps(FixedClock(), stream=None, watchlist=["AAPL"]))
    with TestClient(app) as client, client.websocket_connect("/ws/quotes") as ws:
        status = ws.receive_json()
        assert status == {"type": "status", "enabled": False}


def test_quotes_ws_enabled_pushes_quote_frames() -> None:
    stream = FakeQuoteStream(enabled=True)
    app = create_app(_deps(FixedClock(), stream=stream, watchlist=["AAPL", "MSFT"]))
    with TestClient(app) as client, client.websocket_connect("/ws/quotes") as ws:
        status = ws.receive_json()
        assert status == {"type": "status", "enabled": True}

        stream.push(
            Quote(
                symbol="AAPL",
                bid=Decimal("232.10"),
                ask=Decimal("232.14"),
                last=Decimal("232.12"),
                as_of=_AS_OF,
            )
        )
        frame = ws.receive_json()
        assert frame == {
            "type": "quote",
            "symbol": "AAPL",
            "bid": "232.10",
            "ask": "232.14",
            "last": "232.12",
            "as_of": _AS_OF.isoformat(),
        }
    # On connect the endpoint seeds the upstream subscription with the tracked/watchlist set.
    assert ["AAPL", "MSFT"] in stream.added


def test_quotes_ws_client_subscribe_reaches_stream() -> None:
    stream = FakeQuoteStream(enabled=True)
    app = create_app(_deps(FixedClock(), stream=stream, watchlist=["AAPL"]))
    with TestClient(app) as client, client.websocket_connect("/ws/quotes") as ws:
        ws.receive_json()  # status
        ws.send_json({"action": "subscribe", "symbols": ["lmt", "nvda"]})
        # Round-trip a pushed quote to be sure the subscribe was processed first.
        stream.push(
            Quote(
                symbol="LMT",
                bid=Decimal("450.00"),
                ask=Decimal("450.20"),
                last=Decimal("450.10"),
                as_of=_AS_OF,
            )
        )
        frame = ws.receive_json()
        assert frame["symbol"] == "LMT"
    assert ["LMT", "NVDA"] in stream.added
