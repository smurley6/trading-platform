"""In-memory read model: maintained current-state projections (CQRS read side).

Default (no-DB) path. Projections are materialized incrementally as events are appended —
not replayed per request — so a POST then GET sees its own write (read-after-write
consistency, which the e2e tests rely on). Rebuildable from the log via `rebuild`.

The projection math is the single source in `application.projections`; this class only
threads each appended event through it and stores the resulting current state.
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading.application.ports.read_model import PortfolioSnapshotRecord, SignalRecord
from trading.application.projections import _signal_from_payload
from trading.domain.events import EventType, StoredEvent
from trading.domain.orders import Fill, OrderSide
from trading.domain.positions import HeldPosition, Portfolio
from trading.domain.signals import Signal


class InMemoryReadModel:
    """Materialized projections kept current on each appended event."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._portfolios: dict[str, Portfolio] = {}  # sleeve_id -> replayed fills
        self._order_statuses: dict[str, tuple[str, str]] = {}  # client_order_id -> (sleeve, status)
        self._signals: list[SignalRecord] = []  # recording order
        self._snapshots: dict[str, PortfolioSnapshotRecord] = {}  # sleeve_id -> latest

    async def apply(self, stored: StoredEvent) -> None:
        """Fold one newly appended event into the projections."""
        async with self._lock:
            self._apply(stored)

    async def rebuild(self, events: Sequence[StoredEvent]) -> None:
        """Rebuild every projection from scratch by replaying the log (spec section 9)."""
        async with self._lock:
            self._portfolios.clear()
            self._order_statuses.clear()
            self._signals.clear()
            self._snapshots.clear()
            for stored in events:
                self._apply(stored)

    def _apply(self, stored: StoredEvent) -> None:
        event_type = stored.event.event_type
        payload = stored.event.payload
        if event_type == EventType.SIGNAL_RECEIVED:
            self._signals.append(
                SignalRecord(
                    signal=_signal_from_payload(payload),
                    occurred_at=stored.event.occurred_at,
                )
            )
        elif event_type == EventType.ORDER_FILLED:
            sleeve_id = str(payload.get("sleeve_id", ""))
            fill = Fill(
                client_order_id=str(payload.get("client_order_id", "")),
                symbol=str(payload["symbol"]),
                side=OrderSide(str(payload["side"])),
                price=Decimal(str(payload["price"])),
                quantity=Decimal(str(payload["quantity"])),
                fees=Decimal(str(payload.get("fees", "0"))),
                filled_at=datetime.fromisoformat(str(payload["filled_at"])),
            )
            portfolio = self._portfolios.get(sleeve_id, Portfolio(cash=Decimal("0")))
            self._portfolios[sleeve_id] = portfolio.apply_fill(fill)
        elif event_type == EventType.PORTFOLIO_SNAPSHOT:
            sleeve_id = str(payload["sleeve_id"])
            self._snapshots[sleeve_id] = PortfolioSnapshotRecord(
                sleeve_id=sleeve_id,
                taken_at=stored.event.occurred_at,
                cash=Decimal(str(payload["cash"])),
                equity=Decimal(str(payload["equity"])),
            )
        else:
            self._apply_order_status(event_type, payload)

    def _apply_order_status(self, event_type: str, payload: dict[str, object]) -> None:
        status = {
            EventType.ORDER_INTENT: "intent",
            EventType.ORDER_SUBMITTED: "submitted",
            EventType.ORDER_SUBMIT_FAILED: "submit_failed",
        }.get(event_type)
        if event_type == EventType.ORDER_STATUS_CHANGED:
            status = str(payload["status"])
        if status is None:
            return
        client_order_id = str(payload["client_order_id"])
        sleeve_id = str(payload.get("sleeve_id", ""))
        self._order_statuses[client_order_id] = (sleeve_id, status)

    async def held_positions(self) -> Sequence[HeldPosition]:
        async with self._lock:
            held: list[HeldPosition] = []
            for sleeve_id, portfolio in self._portfolios.items():
                for position in portfolio.positions.values():
                    held.append(
                        HeldPosition(
                            symbol=position.symbol,
                            sleeve_id=sleeve_id,
                            quantity=position.quantity,
                            avg_price=position.avg_entry_price,
                        )
                    )
            return held

    async def order_statuses(self, *, sleeve_id: str | None = None) -> dict[str, str]:
        async with self._lock:
            return {
                order_id: status
                for order_id, (sleeve, status) in self._order_statuses.items()
                if sleeve_id is None or sleeve == sleeve_id
            }

    async def signals_in_window(self, *, now: datetime, window_days: int) -> Sequence[Signal]:
        cutoff = now - timedelta(days=window_days)
        async with self._lock:
            return [r.signal for r in self._signals if r.signal.created_at >= cutoff]

    async def recent_signals(self, *, limit: int) -> Sequence[SignalRecord]:
        async with self._lock:
            return list(self._signals[-limit:])

    async def all_signals(self) -> Sequence[Signal]:
        async with self._lock:
            return [r.signal for r in self._signals]

    async def latest_portfolio_snapshots(self) -> Sequence[PortfolioSnapshotRecord]:
        async with self._lock:
            return list(self._snapshots.values())
