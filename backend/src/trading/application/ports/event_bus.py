"""Event broadcaster port: live fan-out of appended events to subscribers (FR-API-1).

The WebSocket layer subscribes here to stream signals, order updates, and P&L in real time
without polling. Implemented by a broadcasting wrapper around the event store.
"""

import asyncio
from typing import Protocol

from trading.domain.events import StoredEvent


class EventBroadcasterPort(Protocol):
    def subscribe(self) -> "asyncio.Queue[StoredEvent]":
        """Register a subscriber and return its queue of newly appended events."""
        ...

    def unsubscribe(self, queue: "asyncio.Queue[StoredEvent]") -> None: ...
