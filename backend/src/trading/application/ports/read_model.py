"""Read-model port: maintained current-state projections (CQRS read side).

The event log is the system of record; these projections are rebuildable views over it
(spec section 9). Hot read endpoints query this port instead of replaying the whole log per
request. Implementations keep their projections current as events are appended and can be
rebuilt from scratch by replaying the log.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from trading.domain.positions import HeldPosition
from trading.domain.signals import Signal


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """A recorded signal as the feed/conviction views need it.

    Carries the reconstructed domain `Signal` plus the event's `occurred_at` (the moment it
    was recorded), which the signal feed surfaces separately from the signal's `created_at`.
    """

    signal: Signal
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class PortfolioSnapshotRecord:
    """A point on a sleeve's equity curve, persisted as time-series data."""

    sleeve_id: str
    taken_at: datetime
    cash: Decimal
    equity: Decimal


class ReadModelPort(Protocol):
    """Current-state queries backing the hot read endpoints (CQRS read side)."""

    async def held_positions(self) -> Sequence[HeldPosition]:
        """Open positions with cost basis, grouped per sleeve (closed lots excluded)."""
        ...

    async def order_statuses(self, *, sleeve_id: str | None = None) -> dict[str, str]:
        """Latest known local status per client_order_id, optionally scoped to one sleeve."""
        ...

    async def signals_in_window(self, *, now: datetime, window_days: int) -> Sequence[Signal]:
        """Signals created within the rolling corroboration window (FR-POL-4)."""
        ...

    async def recent_signals(self, *, limit: int) -> Sequence[SignalRecord]:
        """The most recently recorded signals, oldest-first, for the signal feed."""
        ...

    async def all_signals(self) -> Sequence[Signal]:
        """Every recorded signal, in recording order (e.g. for backtests)."""
        ...

    async def latest_portfolio_snapshots(self) -> Sequence[PortfolioSnapshotRecord]:
        """The newest equity-curve point per sleeve."""
        ...
