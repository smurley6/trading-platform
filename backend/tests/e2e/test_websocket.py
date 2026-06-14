"""The /ws/events WebSocket streams appended events live (FR-API-1)."""

from starlette.testclient import TestClient

from tests.conftest import FixedClock
from trading.application.risk import KillSwitch
from trading.application.sleeve_registry import SleeveRegistry
from trading.domain.conviction import ConvictionThresholds
from trading.domain.events import EventType
from trading.infrastructure.persistence.broadcasting import BroadcastingEventStore
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.interface.app import AppDeps, create_app


def test_websocket_streams_events_live() -> None:
    clock = FixedClock()
    event_bus = BroadcastingEventStore(InMemoryEventStore(clock))
    deps = AppDeps(
        event_store=event_bus,
        kill_switch=KillSwitch(),
        sleeve_registry=SleeveRegistry([]),
        evaluators={},
        default_sleeve_id="sleeve-political",
        reconcilers=[],
        clock=clock,
        conviction_thresholds=ConvictionThresholds(),
        conviction_window_days=21,
        event_bus=event_bus,
    )
    app = create_app(deps)

    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        # A user action appends events, which must arrive over the socket.
        client.post("/kill-switch", json={"engaged": True})
        first = ws.receive_json()
        assert first["event_type"] in {
            EventType.USER_ACTION,
            EventType.KILL_SWITCH_CHANGED,
        }
        assert first["correlation_id"]
