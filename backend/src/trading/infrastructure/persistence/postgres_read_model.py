"""Postgres read model: current-state projections backed by `rm_*` tables and the
`portfolio_snapshots` TimescaleDB hypertable (CQRS read side).

Projections are updated inside the event-append transaction (`apply`), so the read side is
always consistent with the log and a write is visible to the next read. `rebuild` replays the
whole log to repopulate the tables from scratch (spec section 9 — projections are derivable).

`apply` reuses the domain math (`Portfolio.apply_fill`) for cost basis, so the persisted
positions agree with the live engine and the in-memory projection.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import Row, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from trading.application.ports.read_model import PortfolioSnapshotRecord, SignalRecord
from trading.application.projections import _signal_from_payload
from trading.domain.events import EventType, StoredEvent
from trading.domain.orders import Fill, OrderSide
from trading.domain.positions import HeldPosition, Portfolio, Position
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.tables import (
    portfolio_snapshots_table,
    rm_order_statuses_table,
    rm_positions_table,
    rm_signals_table,
)


class PostgresReadModel:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # --- write side: fold an appended event into the projection tables ---

    async def apply(self, conn: AsyncConnection, stored: StoredEvent, recorded_id: int) -> None:
        event_type = stored.event.event_type
        payload = stored.event.payload
        if event_type == EventType.SIGNAL_RECEIVED:
            await self._apply_signal(conn, stored, recorded_id)
        elif event_type == EventType.ORDER_FILLED:
            await self._apply_fill(conn, payload)
        elif event_type == EventType.PORTFOLIO_SNAPSHOT:
            await self._apply_snapshot(conn, stored)
        else:
            await self._apply_order_status(conn, event_type, payload)

    async def _apply_signal(
        self, conn: AsyncConnection, stored: StoredEvent, recorded_id: int
    ) -> None:
        signal = _signal_from_payload(stored.event.payload)
        stmt = pg_insert(rm_signals_table).values(
            signal_id=signal.id,
            created_at=signal.created_at,
            occurred_at=stored.event.occurred_at,
            recorded_id=recorded_id,
            source_type=signal.source_type.value,
            ticker=signal.ticker,
            politician_id=signal.politician_id,
            sentiment=signal.sentiment,
            politician_weight=signal.politician_weight,
            source_reliability=signal.source_reliability,
            raw_payload=signal.raw_payload,
        )
        # A signal id is recorded once; ignore a replayed duplicate so rebuild is idempotent.
        await conn.execute(stmt.on_conflict_do_nothing(index_elements=["signal_id"]))

    async def _apply_fill(self, conn: AsyncConnection, payload: dict[str, object]) -> None:
        sleeve_id = str(payload.get("sleeve_id", ""))
        symbol = str(payload["symbol"])
        fill = Fill(
            client_order_id=str(payload.get("client_order_id", "")),
            symbol=symbol,
            side=OrderSide(str(payload["side"])),
            price=Decimal(str(payload["price"])),
            quantity=Decimal(str(payload["quantity"])),
            fees=Decimal(str(payload.get("fees", "0"))),
            filled_at=datetime.fromisoformat(str(payload["filled_at"])),
        )
        existing = (
            await conn.execute(
                select(rm_positions_table.c.quantity, rm_positions_table.c.avg_price).where(
                    rm_positions_table.c.sleeve_id == sleeve_id,
                    rm_positions_table.c.symbol == symbol,
                )
            )
        ).one_or_none()
        positions: dict[str, Position] = {}
        if existing is not None:
            positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(existing.quantity)),
                avg_entry_price=Decimal(str(existing.avg_price)),
            )
        # Reuse the domain cost-basis math; cash is irrelevant to the held-position view.
        updated = Portfolio(cash=Decimal("0"), positions=positions).apply_fill(fill)
        position = updated.positions.get(symbol)
        if position is None:
            await conn.execute(
                delete(rm_positions_table).where(
                    rm_positions_table.c.sleeve_id == sleeve_id,
                    rm_positions_table.c.symbol == symbol,
                )
            )
            return
        stmt = pg_insert(rm_positions_table).values(
            sleeve_id=sleeve_id,
            symbol=symbol,
            quantity=position.quantity,
            avg_price=position.avg_entry_price,
        )
        await conn.execute(
            stmt.on_conflict_do_update(
                index_elements=["sleeve_id", "symbol"],
                set_={"quantity": position.quantity, "avg_price": position.avg_entry_price},
            )
        )

    async def _apply_snapshot(self, conn: AsyncConnection, stored: StoredEvent) -> None:
        payload = stored.event.payload
        await conn.execute(
            pg_insert(portfolio_snapshots_table).values(
                sleeve_id=str(payload["sleeve_id"]),
                taken_at=stored.event.occurred_at,
                cash=Decimal(str(payload["cash"])),
                equity=Decimal(str(payload["equity"])),
            )
        )

    async def _apply_order_status(
        self, conn: AsyncConnection, event_type: str, payload: dict[str, object]
    ) -> None:
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
        stmt = pg_insert(rm_order_statuses_table).values(
            client_order_id=client_order_id, sleeve_id=sleeve_id, status=status
        )
        await conn.execute(
            stmt.on_conflict_do_update(
                index_elements=["client_order_id"],
                set_={"status": status, "sleeve_id": sleeve_id},
            )
        )

    # --- rebuild from the log (spec section 9: projections are derivable) ---

    async def rebuild(self, events: Sequence[StoredEvent]) -> None:
        """Truncate the read-model tables and replay the log to repopulate them."""
        async with self._engine.begin() as conn:
            for table in (
                rm_signals_table,
                rm_positions_table,
                rm_order_statuses_table,
                portfolio_snapshots_table,
            ):
                await conn.execute(delete(table))
            # `events.id` is the global recording order; rebuild keeps feed ordering stable.
            for recorded_id, stored in enumerate(events, start=1):
                await self.apply(conn, stored, recorded_id)

    # --- read side: queries backing the hot endpoints ---

    async def held_positions(self) -> Sequence[HeldPosition]:
        query = select(rm_positions_table).order_by(
            rm_positions_table.c.sleeve_id, rm_positions_table.c.symbol
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [
            HeldPosition(
                symbol=row.symbol,
                sleeve_id=row.sleeve_id,
                quantity=Decimal(str(row.quantity)),
                avg_price=Decimal(str(row.avg_price)),
            )
            for row in rows
        ]

    async def order_statuses(self, *, sleeve_id: str | None = None) -> dict[str, str]:
        query = select(rm_order_statuses_table.c.client_order_id, rm_order_statuses_table.c.status)
        if sleeve_id is not None:
            query = query.where(rm_order_statuses_table.c.sleeve_id == sleeve_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return {row.client_order_id: row.status for row in rows}

    async def signals_in_window(self, *, now: datetime, window_days: int) -> Sequence[Signal]:
        cutoff = now - timedelta(days=window_days)
        query = (
            select(rm_signals_table)
            .where(rm_signals_table.c.created_at >= cutoff)
            .order_by(rm_signals_table.c.recorded_id)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [_row_to_signal(row) for row in rows]

    async def recent_signals(self, *, limit: int) -> Sequence[SignalRecord]:
        query = (
            select(rm_signals_table).order_by(rm_signals_table.c.recorded_id.desc()).limit(limit)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        records = [
            SignalRecord(signal=_row_to_signal(row), occurred_at=row.occurred_at) for row in rows
        ]
        records.reverse()  # oldest-first, matching the in-memory feed
        return records

    async def all_signals(self) -> Sequence[Signal]:
        query = select(rm_signals_table).order_by(rm_signals_table.c.recorded_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [_row_to_signal(row) for row in rows]

    async def latest_portfolio_snapshots(self) -> Sequence[PortfolioSnapshotRecord]:
        # DISTINCT ON keeps the newest point per sleeve from the equity-curve hypertable.
        query = (
            select(portfolio_snapshots_table)
            .distinct(portfolio_snapshots_table.c.sleeve_id)
            .order_by(
                portfolio_snapshots_table.c.sleeve_id,
                portfolio_snapshots_table.c.taken_at.desc(),
            )
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [
            PortfolioSnapshotRecord(
                sleeve_id=row.sleeve_id,
                taken_at=row.taken_at,
                cash=Decimal(str(row.cash)),
                equity=Decimal(str(row.equity)),
            )
            for row in rows
        ]


def _row_to_signal(row: Row[Any]) -> Signal:
    return Signal(
        id=row.signal_id,
        created_at=row.created_at,
        source_type=SignalSource(row.source_type),
        ticker=row.ticker,
        politician_id=row.politician_id,
        sentiment=float(row.sentiment),
        politician_weight=float(row.politician_weight),
        source_reliability=float(row.source_reliability),
        raw_payload=dict(row.raw_payload),
    )
