"""The /ws/events WebSocket streams appended events live (FR-API-1)."""

import logging
from typing import Any

import pytest
from starlette.testclient import TestClient
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocketDisconnect

from tests.conftest import FixedClock
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.events import EventType
from trading.infrastructure.persistence.broadcasting import BroadcastingEventStore
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.interface.app import AppDeps, create_app


def _deps(*, api_key: str | None = None) -> AppDeps:
    clock = FixedClock()
    read_model = InMemoryReadModel()
    event_bus = BroadcastingEventStore(ProjectingEventStore(InMemoryEventStore(clock), read_model))
    return AppDeps(
        event_store=event_bus,
        read_model=read_model,
        kill_switch=KillSwitch(),
        sleeve_registry=SleeveRegistry([]),
        evaluators={},
        default_sleeve_id="sleeve-political",
        reconcilers=[],
        order_syncs=[],
        clock=clock,
        conviction_thresholds=ConvictionThresholds(),
        conviction_window_days=21,
        event_bus=event_bus,
        api_key=api_key,
    )


def test_websocket_streams_events_live() -> None:
    app = create_app(_deps())

    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        # A user action appends events, which must arrive over the socket.
        client.post("/kill-switch", json={"engaged": True})
        first = ws.receive_json()
        assert first["event_type"] in {
            EventType.USER_ACTION,
            EventType.KILL_SWITCH_CHANGED,
        }
        assert first["correlation_id"]


def test_ws_events_rejects_missing_key_when_secured() -> None:
    # The event firehose is a sensitive read (review F24): with a key configured, an unauthenticated
    # socket is closed (1008), never accepted.
    app = create_app(_deps(api_key="s3cret"))
    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect("/ws/events"),
    ):
        pass
    assert exc.value.code == 1008


def test_ws_events_accepts_valid_key_when_secured() -> None:
    app = create_app(_deps(api_key="s3cret"))
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/events", headers={"X-API-Key": "s3cret"}) as ws,
    ):
        client.post("/kill-switch", json={"engaged": True}, headers={"X-API-Key": "s3cret"})
        first = ws.receive_json()
        assert first["correlation_id"]


# --- browser path: the key rides the Sec-WebSocket-Protocol offer (task-22) --------------------
# The browser WebSocket constructor cannot set request headers, so the console connects with
# `new WebSocket(url, ["api-key", key])`. TestClient's `subprotocols=` produces exactly that
# handshake (one comma-joined Sec-WebSocket-Protocol header + the split scope list).


def _accepted_subprotocol(ws: Any) -> object:
    # TestClient types this attribute as None (it is only ever reassigned from an untyped
    # message), so read it through `object` to keep the comparison meaningful under mypy.
    accepted: object = ws.accepted_subprotocol
    return accepted


def test_ws_events_accepts_browser_subprotocol_key_when_secured() -> None:
    app = create_app(_deps(api_key="s3cret"))
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/events", subprotocols=["api-key", "s3cret"]) as ws,
    ):
        # The server must select one of the OFFERED subprotocols or a real browser fails the
        # handshake — and it must select the sentinel, not the key.
        assert _accepted_subprotocol(ws) == "api-key"
        client.post("/kill-switch", json={"engaged": True}, headers={"X-API-Key": "s3cret"})
        first = ws.receive_json()
        assert first["correlation_id"]


@pytest.mark.parametrize(
    "offered",
    [
        ["api-key", "wrong"],  # wrong key
        ["api-key"],  # sentinel, no key
        ["api-key", "s3cret", "extra"],  # ambiguous offer => rejected, never guessed
        ["extra", "api-key", "s3cret"],  # sentinel not first
        ["s3cret"],  # bare key without the sentinel
    ],
)
def test_ws_events_rejects_bad_browser_offer_without_sending_events(offered: list[str]) -> None:
    app = create_app(_deps(api_key="s3cret"))
    with TestClient(app) as client:
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/ws/events", subprotocols=offered),
        ):
            pass
        assert exc.value.code == 1008
        # An event appended while the socket was refused must not have reached it: the close
        # happens BEFORE accept, so there is no accepted socket to receive a frame at all.
        client.post("/kill-switch", json={"engaged": True}, headers={"X-API-Key": "s3cret"})


def test_ws_events_open_mode_still_echoes_the_browser_offer() -> None:
    # No API_KEY: open exactly as before. The offer is still answered, so the console's
    # browser-shaped connect works in dev too.
    app = create_app(_deps())
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/events", subprotocols=["api-key", "anything"]) as ws,
    ):
        assert _accepted_subprotocol(ws) == "api-key"
        client.post("/kill-switch", json={"engaged": True})
        assert ws.receive_json()["correlation_id"]


def test_ws_events_browser_key_never_reaches_a_url_or_a_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The whole point of the subprotocol mechanism over a query parameter (review F24): the key
    # must not land in the access log, the request path, or any proxy-visible query string.
    key = "s3cret-browser-key"
    app = create_app(_deps(api_key=key))
    scopes: list[dict[str, Any]] = []

    async def recording(scope: Scope, receive: Receive, send: Send) -> None:
        scopes.append(dict(scope))
        await app(scope, receive, send)

    with caplog.at_level(logging.DEBUG):
        with (
            TestClient(recording) as client,
            client.websocket_connect("/ws/events", subprotocols=["api-key", key]) as ws,
        ):
            client.post("/kill-switch", json={"engaged": True}, headers={"X-API-Key": key})
            assert ws.receive_json()["correlation_id"]
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws/events", subprotocols=["api-key", "wrong"]),
        ):
            pass

    sockets = [scope for scope in scopes if scope["type"] == "websocket"]
    assert sockets, "expected the recorded websocket handshakes"
    for scope in sockets:
        assert scope["query_string"] == b""
        assert key not in scope["path"]
        assert key not in scope["raw_path"].decode()
    assert caplog.records, "expected the access log to have run"
    assert key not in caplog.text
    for record in caplog.records:
        assert key not in str(record.__dict__)
