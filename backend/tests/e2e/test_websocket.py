"""The /ws/events WebSocket streams appended events live (FR-API-1)."""

import pytest
from starlette.testclient import TestClient
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
