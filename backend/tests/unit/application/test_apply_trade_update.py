"""ApplyTradeUpdate: real-time fills/status recorded as events, idempotently."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tests.conftest import FixedClock
from trading.application.apply_trade_update import ApplyTradeUpdate
from trading.application.ports.trade_updates import TradeUpdate
from trading.application.projections import (
    held_positions_from_events,
    order_statuses_from_events,
)
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide, OrderStatus
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore

_SLEEVE = "sleeve-political"
_COID = "order-abc"


@pytest.fixture
def store(clock: FixedClock) -> ProjectingEventStore:
    return ProjectingEventStore(InMemoryEventStore(clock), InMemoryReadModel())


async def _seed_submitted(store: ProjectingEventStore, clock: FixedClock) -> None:
    """Make the order known to the sleeve's local log (as PlaceOrder would)."""
    for event_type in (EventType.ORDER_INTENT, EventType.ORDER_SUBMITTED):
        await store.append(
            DomainEvent(
                stream_id=f"order-{_COID}",
                event_type=event_type,
                payload={"client_order_id": _COID, "sleeve_id": _SLEEVE},
                occurred_at=clock.now(),
                correlation_id="corr-seed",
            )
        )


def _fill_update(qty: str = "10", price: str = "100") -> TradeUpdate:
    return TradeUpdate(
        client_order_id=_COID,
        broker_order_id="brk-1",
        event="fill",
        status=OrderStatus.FILLED,
        symbol="AAPL",
        side=OrderSide.BUY,
        occurred_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
        fill_price=Decimal(price),
        fill_quantity=Decimal(qty),
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )


async def test_records_fill_and_status_for_known_order(
    store: ProjectingEventStore, clock: FixedClock
) -> None:
    await _seed_submitted(store, clock)
    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)

    result = await apply.execute(_fill_update())
    assert result.recorded_fill is True
    assert result.recorded_status_change is True

    events = await store.read_all()
    held = held_positions_from_events(events)
    assert len(held) == 1
    assert held[0].symbol == "AAPL"
    assert held[0].quantity == Decimal("10")
    assert order_statuses_from_events(events, sleeve_id=_SLEEVE)[_COID] == "filled"


async def test_unknown_order_is_skipped(store: ProjectingEventStore, clock: FixedClock) -> None:
    # No seed: the order was never placed by this platform.
    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)
    result = await apply.execute(_fill_update())
    assert result.skipped is True
    assert held_positions_from_events(await store.read_all()) == []


async def test_redelivered_fill_is_idempotent(
    store: ProjectingEventStore, clock: FixedClock
) -> None:
    await _seed_submitted(store, clock)
    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)

    first = await apply.execute(_fill_update())
    second = await apply.execute(_fill_update())  # same cumulative fill replayed

    assert first.recorded_fill is True
    assert second.recorded_fill is False
    held = held_positions_from_events(await store.read_all())
    assert held[0].quantity == Decimal("10")  # not doubled


async def test_partial_then_full_fill_records_only_the_increment(
    store: ProjectingEventStore, clock: FixedClock
) -> None:
    await _seed_submitted(store, clock)
    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)

    partial = TradeUpdate(
        client_order_id=_COID,
        broker_order_id="brk-1",
        event="partial_fill",
        status=OrderStatus.PARTIALLY_FILLED,
        symbol="AAPL",
        side=OrderSide.BUY,
        occurred_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
        fill_price=Decimal("100"),
        fill_quantity=Decimal("4"),  # cumulative so far
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )
    await apply.execute(partial)
    await apply.execute(_fill_update(qty="10"))  # cumulative 10 -> increment 6

    held = held_positions_from_events(await store.read_all())
    assert held[0].quantity == Decimal("10")


async def test_call_handler_entrypoint_applies_update(
    store: ProjectingEventStore, clock: FixedClock
) -> None:
    # The TradeUpdateHandler entrypoint (__call__) drives execute() and ignores the passed
    # sleeve_id in favour of the bound one.
    await _seed_submitted(store, clock)
    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)
    await apply(_fill_update(), sleeve_id="ignored")  # __call__ entrypoint returns None
    held = held_positions_from_events(await store.read_all())
    assert held[0].quantity == Decimal("10")


async def test_status_only_update_records_no_fill(
    store: ProjectingEventStore, clock: FixedClock
) -> None:
    await _seed_submitted(store, clock)
    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)

    canceled = TradeUpdate(
        client_order_id=_COID,
        broker_order_id="brk-1",
        event="canceled",
        status=OrderStatus.CANCELED,
        symbol="AAPL",
        side=OrderSide.BUY,
        occurred_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )
    result = await apply.execute(canceled)
    assert result.recorded_fill is False
    assert result.recorded_status_change is True
    assert (
        order_statuses_from_events(await store.read_all(), sleeve_id=_SLEEVE)[_COID] == "canceled"
    )
