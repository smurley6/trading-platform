"""Default-off guardrails for the live market-data stream (and feed).

With a bare `Settings()` (only the required risk caps set, no data credentials, ENABLE_LIVE_
QUOTES unset) the composition root must inject the DISABLED stream/feed, register NO background
task, and — when the app starts — open no socket and make no network call. This is the
contract that keeps `python -m trading.main` and the test suite network-silent.
"""

import asyncio

import pytest
from starlette.testclient import TestClient

from trading.config import Settings
from trading.infrastructure.market_data.disabled import DisabledQuoteFeed, DisabledQuoteStream
from trading.interface.app import create_app
from trading.main import build_dependencies

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}


def _default_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    # Only the required risk caps; everything market-data stays at its OFF default.
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    # Ensure no ambient credentials/flags leak in from a developer .env or shell.
    for key in (
        "ENABLE_LIVE_QUOTES",
        "ALPACA_DATA_KEY_ID",
        "ALPACA_DATA_SECRET_KEY",
        "ALPACA_LIVE_KEY_ID",
        "ALPACA_LIVE_SECRET_KEY",
        "ENABLE_ALPACA_PAPER",
        "ALPACA_PAPER_KEY_ID",
        "ALPACA_PAPER_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


def test_default_settings_inject_disabled_stream_and_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = build_dependencies(_default_settings(monkeypatch))
    assert isinstance(deps.quote_stream, DisabledQuoteStream)
    assert isinstance(deps.quote_feed, DisabledQuoteFeed)
    assert deps.quote_stream.enabled is False
    # Nothing to start on startup => no connect/consume task, no socket.
    assert deps.background_tasks == []


def test_startup_opens_no_socket_when_default(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = build_dependencies(_default_settings(monkeypatch))
    app = create_app(deps)

    # If anything tried to dial a websocket, this stand-in would be hit and fail the test.
    async def explode(_url: str) -> object:
        raise AssertionError("a socket was opened under default (disabled) settings")

    import trading.infrastructure.market_data.alpaca_stream as stream_mod

    monkeypatch.setattr(stream_mod, "_default_connect", explode)

    # Full lifespan: startup + shutdown. With no background tasks this must be inert.
    with TestClient(app):
        pass

    # The disabled stream hands back a real (empty) queue but never produces.
    stream = deps.quote_stream
    assert stream is not None
    queue = stream.subscribe()
    assert isinstance(queue, asyncio.Queue)
    assert queue.empty()
