"""In-memory read model + projecting store: current-state projections stay consistent as
events are appended (CQRS read side), and are rebuildable from the log.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.conftest import FixedClock
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore

_WHEN = datetime(2026, 6, 13, 14, 0, tzinfo=UTC)


def _signal_event(*, signal_id: str, ticker: str, created_at: datetime) -> StoredEvent:
    event = DomainEvent(
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
    return StoredEvent(event=event, seq=1, recorded_at=created_at)


def _fill_event(*, sleeve_id: str, symbol: str, side: str, price: str, qty: str) -> StoredEvent:
    event = DomainEvent(
        stream_id=f"sleeve-{sleeve_id}",
        event_type=EventType.ORDER_FILLED,
        payload={
            "client_order_id": f"ord-{symbol}",
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
    return StoredEvent(event=event, seq=1, recorded_at=_WHEN)


def _order_event(
    *, event_type: str, client_order_id: str, sleeve_id: str, status: str
) -> StoredEvent:
    payload: dict[str, object] = {"client_order_id": client_order_id, "sleeve_id": sleeve_id}
    if event_type == EventType.ORDER_STATUS_CHANGED:
        payload["status"] = status
    event = DomainEvent(
        stream_id=f"order-{client_order_id}",
        event_type=event_type,
        payload=payload,
        occurred_at=_WHEN,
        correlation_id="c1",
    )
    return StoredEvent(event=event, seq=1, recorded_at=_WHEN)


def _snapshot_event(*, sleeve_id: str, cash: str, equity: str, taken_at: datetime) -> StoredEvent:
    event = DomainEvent(
        stream_id=f"sleeve-{sleeve_id}",
        event_type=EventType.PORTFOLIO_SNAPSHOT,
        payload={"sleeve_id": sleeve_id, "cash": cash, "equity": equity, "positions": {}},
        occurred_at=taken_at,
        correlation_id="c1",
    )
    return StoredEvent(event=event, seq=1, recorded_at=taken_at)


async def test_signals_feed_window_and_all() -> None:
    rm = InMemoryReadModel()
    old = _WHEN - timedelta(days=40)
    await rm.apply(_signal_event(signal_id="s-old", ticker="AAPL", created_at=old))
    await rm.apply(_signal_event(signal_id="s-new", ticker="MSFT", created_at=_WHEN))

    feed = await rm.recent_signals(limit=10)
    assert [r.signal.id for r in feed] == ["s-old", "s-new"]  # oldest-first
    assert feed[0].occurred_at == old

    window = await rm.signals_in_window(now=_WHEN, window_days=21)
    assert [s.id for s in window] == ["s-new"]  # the 40-day-old one is outside the window

    assert {s.id for s in await rm.all_signals()} == {"s-old", "s-new"}


async def test_recent_signals_respects_limit() -> None:
    rm = InMemoryReadModel()
    for i in range(5):
        await rm.apply(_signal_event(signal_id=f"s{i}", ticker="AAPL", created_at=_WHEN))
    feed = await rm.recent_signals(limit=2)
    assert [r.signal.id for r in feed] == ["s3", "s4"]


async def test_held_positions_average_cost_and_close_out() -> None:
    rm = InMemoryReadModel()
    await rm.apply(_fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="100", qty="10"))
    await rm.apply(_fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="120", qty="10"))
    [held] = await rm.held_positions()
    assert (held.symbol, held.quantity, held.avg_price) == ("AAPL", Decimal("20"), Decimal("110"))

    # Selling the whole lot drops the position from the view.
    await rm.apply(_fill_event(sleeve_id="s1", symbol="AAPL", side="sell", price="130", qty="20"))
    assert await rm.held_positions() == []


async def test_held_positions_scoped_per_sleeve() -> None:
    rm = InMemoryReadModel()
    await rm.apply(_fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="100", qty="5"))
    await rm.apply(_fill_event(sleeve_id="s2", symbol="AAPL", side="buy", price="200", qty="3"))
    held = sorted(await rm.held_positions(), key=lambda p: p.sleeve_id)
    assert [(h.sleeve_id, h.quantity, h.avg_price) for h in held] == [
        ("s1", Decimal("5"), Decimal("100")),
        ("s2", Decimal("3"), Decimal("200")),
    ]


async def test_order_statuses_latest_wins_and_sleeve_scoped() -> None:
    rm = InMemoryReadModel()
    await rm.apply(
        _order_event(
            event_type=EventType.ORDER_INTENT, client_order_id="o1", sleeve_id="s1", status=""
        )
    )
    await rm.apply(
        _order_event(
            event_type=EventType.ORDER_STATUS_CHANGED,
            client_order_id="o1",
            sleeve_id="s1",
            status="filled",
        )
    )
    await rm.apply(
        _order_event(
            event_type=EventType.ORDER_SUBMITTED, client_order_id="o2", sleeve_id="s2", status=""
        )
    )
    assert await rm.order_statuses() == {"o1": "filled", "o2": "submitted"}
    assert await rm.order_statuses(sleeve_id="s1") == {"o1": "filled"}


async def test_latest_snapshot_per_sleeve() -> None:
    rm = InMemoryReadModel()
    await rm.apply(
        _snapshot_event(sleeve_id="s1", cash="900", equity="1000", taken_at=_WHEN)
    )
    later = _WHEN + timedelta(hours=1)
    await rm.apply(
        _snapshot_event(sleeve_id="s1", cash="800", equity="1100", taken_at=later)
    )
    [snap] = await rm.latest_portfolio_snapshots()
    assert (snap.taken_at, snap.cash, snap.equity) == (later, Decimal("800"), Decimal("1100"))


async def test_projecting_store_is_read_after_write(clock: FixedClock) -> None:
    rm = InMemoryReadModel()
    store = ProjectingEventStore(InMemoryEventStore(clock), rm)
    await store.append(_signal_event(signal_id="s1", ticker="AAPL", created_at=clock.now()).event)
    # The read model reflects the write immediately after append returns.
    assert [s.id for s in await rm.all_signals()] == ["s1"]
    # And the inner store still persisted it.
    assert len(await store.read_all()) == 1


async def test_rebuild_repopulates_from_the_log() -> None:
    rm = InMemoryReadModel()
    events = [
        _signal_event(signal_id="s1", ticker="AAPL", created_at=_WHEN),
        _fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="100", qty="5"),
    ]
    await rm.rebuild(events)
    assert [s.id for s in await rm.all_signals()] == ["s1"]
    [held] = await rm.held_positions()
    assert held.quantity == Decimal("5")

    # Rebuilding again from an empty log clears every projection.
    await rm.rebuild([])
    assert await rm.all_signals() == []
    assert await rm.held_positions() == []
