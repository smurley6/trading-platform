"""Clock port: all time flows through here so tests are deterministic (spec section 14)."""

from datetime import datetime
from typing import Protocol


class ClockPort(Protocol):
    def now(self) -> datetime:
        """Return the current time as a timezone-aware datetime (UTC)."""
        ...


class MutableClockPort(ClockPort, Protocol):
    """A clock whose time can be set — used to drive deterministic backtest replay."""

    def set(self, when: datetime) -> None: ...
