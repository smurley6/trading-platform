"""Postgres read model + TimescaleDB hypertable integration tests.

Require TEST_DATABASE_URL pointing at a Timescale-enabled Postgres with migrations applied
(the CI integration job runs `alembic upgrade head` against timescale/timescaledb:pg16).
"""

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from trading.domain.events import DomainEvent, EventType

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL not set",
    ),
]

_WHEN = datetime(2026, 6, 13, 14, 0, tzinfo=UTC)


def _signal_event(*, signal_id: str, ticker: str, created_at: datetime) -> DomainEvent:
    return DomainEvent(
        stream_id=f"signal-{signal_id}",
        event_type=EventType.SIGNAL_RECEIVED,
        payload={
            "signal_id": signal_id,
            "created_at": created_at.isoformat(),
            "source_type": "statement",
            "ticker": ticker,
            "politician_id": "pol-1",
            "sentiment": 0.8,
            "politician_weight": 1.0,
            "source_reliability": 1.0,
            "raw_payload": {"k": "v"},
        },
        occurred_at=created_at,
        correlation_id="c1",
    )


def _fill_event(*, sleeve_id: str, symbol: str, side: str, price: str, qty: str) -> DomainEvent:
    return DomainEvent(
        stream_id=f"sleeve-{sleeve_id}",
        event_type=EventType.ORDER_FILLED,
        payload={
            "client_order_id": f"ord-{uuid4()}",
            "sleeve_id": sleeve_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": qty,
            "fees": "0",
            "filled_at": _WHEN.isoformat(),
        },
        occurred_at=_WHEN,
        correlation_id="c1",
    )


def _snapshot_event(*, sleeve_id: str, cash: str, equity: str, taken_at: datetime) -> DomainEvent:
    return DomainEvent(
        stream_id=f"sleeve-{sleeve_id}",
        event_type=EventType.PORTFOLIO_SNAPSHOT,
        payload={"sleeve_id": sleeve_id, "cash": cash, "equity": equity, "positions": {}},
        occurred_at=taken_at,
        correlation_id="c1",
    )


async def test_append_keeps_read_model_current() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres import PostgresEventStore
    from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    read_model = PostgresReadModel(engine)
    store = PostgresEventStore(engine, on_append=read_model.apply)
    sleeve = f"sleeve-{uuid4()}"
    sig = f"s-{uuid4()}"
    try:
        # Reset the read-model tables so assertions are isolated from prior runs.
        await read_model.rebuild([])

        await store.append(_signal_event(signal_id=sig, ticker="AAPL", created_at=_WHEN))
        await store.append(
            _fill_event(sleeve_id=sleeve, symbol="AAPL", side="buy", price="100", qty="10")
        )
        await store.append(
            _fill_event(sleeve_id=sleeve, symbol="AAPL", side="buy", price="120", qty="10")
        )

        ids = {s.id for s in await read_model.all_signals()}
        assert sig in ids

        held = {(p.sleeve_id, p.symbol): p for p in await read_model.held_positions()}
        position = held[(sleeve, "AAPL")]
        assert position.quantity == Decimal("20")
        assert position.avg_price == Decimal("110")

        # Selling everything removes the row.
        await store.append(
            _fill_event(sleeve_id=sleeve, symbol="AAPL", side="sell", price="130", qty="20")
        )
        held_keys = {(p.sleeve_id, p.symbol) for p in await read_model.held_positions()}
        assert (sleeve, "AAPL") not in held_keys
    finally:
        await engine.dispose()


async def test_signals_in_window_is_an_indexed_range() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres import PostgresEventStore
    from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    read_model = PostgresReadModel(engine)
    store = PostgresEventStore(engine, on_append=read_model.apply)
    try:
        await read_model.rebuild([])
        old_when = _WHEN - timedelta(days=40)
        await store.append(
            _signal_event(signal_id=f"old-{uuid4()}", ticker="AAPL", created_at=old_when)
        )
        recent_id = f"new-{uuid4()}"
        await store.append(_signal_event(signal_id=recent_id, ticker="MSFT", created_at=_WHEN))

        window = await read_model.signals_in_window(now=_WHEN, window_days=21)
        assert [s.id for s in window] == [recent_id]
    finally:
        await engine.dispose()


async def test_portfolio_snapshots_hypertable_latest_per_sleeve() -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres import PostgresEventStore
    from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    read_model = PostgresReadModel(engine)
    store = PostgresEventStore(engine, on_append=read_model.apply)
    sleeve = f"sleeve-{uuid4()}"
    try:
        await read_model.rebuild([])
        await store.append(
            _snapshot_event(sleeve_id=sleeve, cash="900", equity="1000", taken_at=_WHEN)
        )
        later = _WHEN + timedelta(hours=1)
        await store.append(
            _snapshot_event(sleeve_id=sleeve, cash="800", equity="1100", taken_at=later)
        )

        latest = {s.sleeve_id: s for s in await read_model.latest_portfolio_snapshots()}
        assert latest[sleeve].equity == Decimal("1100")
        assert latest[sleeve].taken_at == later

        # portfolio_snapshots is a TimescaleDB hypertable (time-series storage, spec mandate).
        async with engine.connect() as conn:
            count = await conn.scalar(
                text(
                    "SELECT count(*) FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = 'portfolio_snapshots'"
                )
            )
        assert count == 1
    finally:
        await engine.dispose()


async def test_rebuild_repopulates_from_the_log() -> None:
    from datetime import datetime as _dt

    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.domain.events import StoredEvent
    from trading.infrastructure.persistence.postgres_read_model import PostgresReadModel

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    read_model = PostgresReadModel(engine)
    sig = f"s-{uuid4()}"
    sleeve = f"sleeve-{uuid4()}"
    try:
        recorded_at = _dt.now(tz=UTC)
        events = [
            StoredEvent(
                event=_signal_event(signal_id=sig, ticker="AAPL", created_at=_WHEN),
                seq=1,
                recorded_at=recorded_at,
            ),
            StoredEvent(
                event=_fill_event(
                    sleeve_id=sleeve, symbol="AAPL", side="buy", price="100", qty="5"
                ),
                seq=1,
                recorded_at=recorded_at,
            ),
        ]
        await read_model.rebuild(events)
        assert sig in {s.id for s in await read_model.all_signals()}
        held = {(p.sleeve_id, p.symbol): p for p in await read_model.held_positions()}
        assert held[(sleeve, "AAPL")].quantity == Decimal("5")

        await read_model.rebuild([])
        assert await read_model.all_signals() == []
    finally:
        await engine.dispose()
