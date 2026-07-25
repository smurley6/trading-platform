"""SyncOrders: ingest the broker's async fills/status changes the local log is missing.

The Alpaca venue fills AFTER submission, so PlaceOrder never sees the fill. SyncOrders pulls
the broker's orders and appends the missing `order.filled` / `order.status_changed` events,
idempotently (rebuilds the local picture from the log every run), and only for orders the
local log already knows about.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tests.conftest import FixedClock
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.projections import (
    held_positions_from_events,
    order_statuses_from_events,
    positions_from_events,
)
from trading.application.sync_orders import SyncOrders
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType
from trading.infrastructure.persistence.memory import InMemoryEventStore

_SLEEVE = "sleeve-test"


def _fill(symbol: str = "AAPL", quantity: str = "5") -> Fill:
    return Fill(
        client_order_id="ord-1",
        symbol=symbol,
        side=OrderSide.BUY,
        price=Decimal("100"),
        quantity=Decimal(quantity),
        fees=Decimal("0"),
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )


def _broker_order(
    *,
    client_order_id: str = "ord-1",
    status: OrderStatus,
    fills: tuple[Fill, ...] = (),
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="b-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        order_type=OrderType.MARKET,
        status=status,
        fills=fills,
    )


class _FakeBroker:
    """Minimal BrokerPort returning a fixed list of orders."""

    def __init__(self, orders: Sequence[BrokerOrder]) -> None:
        self._orders = list(orders)

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return self._orders

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return []

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"))


async def _seed_submitted(store: InMemoryEventStore, clock: FixedClock) -> None:
    """Make the local log aware of `ord-1` at status `submitted` (as PlaceOrder would)."""
    await store.append(
        DomainEvent(
            stream_id="order-ord-1",
            event_type=EventType.ORDER_SUBMITTED,
            payload={"client_order_id": "ord-1", "sleeve_id": _SLEEVE},
            occurred_at=clock.now(),
            correlation_id="corr-seed",
        )
    )


def _sync(broker: _FakeBroker, store: InMemoryEventStore, clock: FixedClock) -> SyncOrders:
    return SyncOrders(broker=broker, event_store=store, clock=clock, sleeve_id=_SLEEVE)


async def _order_events(store: InMemoryEventStore) -> list[str]:
    stored = await store.read_stream("order-ord-1")
    return [s.event.event_type for s in stored]


async def test_emits_missing_fill(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    broker = _FakeBroker([_broker_order(status=OrderStatus.FILLED, fills=(_fill(),))])

    report = await _sync(broker, store, clock).execute()

    assert report.fills_emitted == 1
    # Filled is also a status change vs the local `submitted`.
    assert report.status_changes_emitted == 1
    assert report.checked_orders == 1
    types = await _order_events(store)
    assert types.count(EventType.ORDER_FILLED) == 1
    assert types.count(EventType.ORDER_STATUS_CHANGED) == 1
    # The fill carries the place_order payload shape (used by the positions projection).
    fill_event = next(
        s
        for s in await store.read_stream("order-ord-1")
        if s.event.event_type == EventType.ORDER_FILLED
    )
    payload = fill_event.event.payload
    assert payload["sleeve_id"] == _SLEEVE
    assert payload["symbol"] == "AAPL"
    assert payload["quantity"] == "5"
    assert payload["price"] == "100"


async def test_idempotent_second_run_emits_nothing(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    broker = _FakeBroker([_broker_order(status=OrderStatus.FILLED, fills=(_fill(),))])
    sync = _sync(broker, store, clock)

    first = await sync.execute()
    assert not first.emitted_nothing

    second = await sync.execute()

    assert second.emitted_nothing
    assert second.fills_emitted == 0
    assert second.status_changes_emitted == 0
    # Still exactly one fill + one status change recorded — no duplicates.
    types = await _order_events(store)
    assert types.count(EventType.ORDER_FILLED) == 1
    assert types.count(EventType.ORDER_STATUS_CHANGED) == 1


async def test_status_only_change_is_emitted(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    # Canceled at the broker, no fills.
    broker = _FakeBroker([_broker_order(status=OrderStatus.CANCELED)])

    report = await _sync(broker, store, clock).execute()

    assert report.fills_emitted == 0
    assert report.status_changes_emitted == 1
    types = await _order_events(store)
    assert EventType.ORDER_FILLED not in types
    assert types.count(EventType.ORDER_STATUS_CHANGED) == 1
    status_event = next(
        s
        for s in await store.read_stream("order-ord-1")
        if s.event.event_type == EventType.ORDER_STATUS_CHANGED
    )
    assert status_event.event.payload["status"] == OrderStatus.CANCELED.value


async def test_unknown_order_is_ignored(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    # Local log knows nothing about this order; the broker reports a filled account order.
    broker = _FakeBroker(
        [
            _broker_order(
                client_order_id="ord-foreign",
                status=OrderStatus.FILLED,
                fills=(_fill(),),
            )
        ]
    )

    report = await _sync(broker, store, clock).execute()

    assert report.emitted_nothing
    assert report.checked_orders == 0
    assert list(await store.read_all()) == []


async def test_status_matching_local_is_not_re_emitted(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)
    # Broker still reports `submitted` — the baseline PlaceOrder already recorded.
    broker = _FakeBroker([_broker_order(status=OrderStatus.SUBMITTED)])

    report = await _sync(broker, store, clock).execute()

    assert report.checked_orders == 1
    assert report.emitted_nothing
    assert await _order_events(store) == [EventType.ORDER_SUBMITTED]


def _fill_qty(quantity: str, price: str = "100") -> Fill:
    return Fill(
        client_order_id="ord-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        price=Decimal(price),
        quantity=Decimal(quantity),
        fees=Decimal("0"),
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )


async def test_partial_then_complete_records_the_remaining_increment(clock: FixedClock) -> None:
    # Review F8: the broker reports a CUMULATIVE filled quantity. A 100-share order polled at 40
    # filled must record 40; when it completes to 100 the remaining 60 must still be recorded
    # (the old boolean gate dropped it forever). Final position == 100, status FILLED.
    store = InMemoryEventStore(clock)
    await _seed_submitted(store, clock)

    partial_broker = _FakeBroker(
        [_broker_order(status=OrderStatus.PARTIALLY_FILLED, fills=(_fill_qty("40"),))]
    )
    first = await _sync(partial_broker, store, clock).execute()
    assert first.fills_emitted == 1
    assert positions_from_events(await store.read_all(), sleeve_id=_SLEEVE) == {
        "AAPL": Decimal("40")
    }

    complete_broker = _FakeBroker(
        [_broker_order(status=OrderStatus.FILLED, fills=(_fill_qty("100"),))]
    )
    second = await _sync(complete_broker, store, clock).execute()
    assert second.fills_emitted == 1  # only the 60-share increment

    events = await store.read_all()
    assert positions_from_events(events, sleeve_id=_SLEEVE) == {"AAPL": Decimal("100")}
    fill_qtys = [
        s.event.payload["quantity"] for s in events if s.event.event_type == EventType.ORDER_FILLED
    ]
    assert fill_qtys == ["40", "60"]
    assert order_statuses_from_events(events, sleeve_id=_SLEEVE)["ord-1"] == "filled"

    # A third run with the same completed order is a pure no-op — nothing left to record.
    third = await _sync(complete_broker, store, clock).execute()
    assert third.emitted_nothing


async def _seed_submitted_order(
    store: InMemoryEventStore, clock: FixedClock, client_order_id: str
) -> None:
    await store.append(
        DomainEvent(
            stream_id=f"order-{client_order_id}",
            event_type=EventType.ORDER_SUBMITTED,
            payload={"client_order_id": client_order_id, "sleeve_id": _SLEEVE},
            occurred_at=clock.now(),
            correlation_id="corr-seed",
        )
    )


def _order(
    *,
    client_order_id: str,
    side: OrderSide,
    fill_qty: str,
    submitted_at: datetime,
    status: OrderStatus = OrderStatus.FILLED,
) -> BrokerOrder:
    fill = Fill(
        client_order_id=client_order_id,
        symbol="AAPL",
        side=side,
        price=Decimal("100"),
        quantity=Decimal(fill_qty),
        fees=Decimal("0"),
        filled_at=submitted_at,
    )
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id=f"b-{client_order_id}",
        symbol="AAPL",
        side=side,
        quantity=Decimal(fill_qty),
        order_type=OrderType.MARKET,
        status=status,
        fills=(fill,),
        submitted_at=submitted_at,
    )


async def test_sell_before_buy_newest_first_is_applied_oldest_first(clock: FixedClock) -> None:
    # Review F7: Alpaca returns orders newest-first. A SELL processed before its BUY used to trip
    # the long-only guard and wedge sync permanently. SyncOrders now sorts by submitted_at
    # ascending, so the BUY is applied first and the projection is correct (holds the net 40).
    store = InMemoryEventStore(clock)
    await _seed_submitted_order(store, clock, "ord-buy")
    await _seed_submitted_order(store, clock, "ord-sell")

    early = datetime(2026, 6, 12, 14, 30, tzinfo=UTC)
    late = datetime(2026, 6, 12, 15, 30, tzinfo=UTC)
    # Broker list is NEWEST-FIRST: the later SELL comes before the earlier BUY.
    broker = _FakeBroker(
        [
            _order(
                client_order_id="ord-sell", side=OrderSide.SELL, fill_qty="60", submitted_at=late
            ),
            _order(
                client_order_id="ord-buy", side=OrderSide.BUY, fill_qty="100", submitted_at=early
            ),
        ]
    )

    report = await SyncOrders(
        broker=broker, event_store=store, clock=clock, sleeve_id=_SLEEVE
    ).execute()
    assert report.failed_orders == 0
    assert report.fills_emitted == 2

    events = await store.read_all()
    # Applied oldest-first (BUY then SELL): 100 bought, 60 sold, 40 held at the BUY cost basis.
    held = {h.symbol: h.quantity for h in held_positions_from_events(events)}
    assert held == {"AAPL": Decimal("40")}

    # A later run is a pure no-op, not a repeated crash.
    second = await SyncOrders(
        broker=broker, event_store=store, clock=clock, sleeve_id=_SLEEVE
    ).execute()
    assert second.emitted_nothing
    assert second.failed_orders == 0


class _FlakyAppendStore:
    """Wraps an InMemoryEventStore and raises on any append to one order's stream, to prove one
    bad order cannot wedge the whole sync loop (review F7 per-order isolation)."""

    def __init__(self, inner: InMemoryEventStore, poison_stream: str) -> None:
        self._inner = inner
        self._poison = poison_stream

    async def append(self, event: DomainEvent, *, expected_version: int | None = None) -> Any:
        if event.stream_id == self._poison:
            raise RuntimeError("simulated append failure")
        return await self._inner.append(event, expected_version=expected_version)

    async def read_stream(self, stream_id: str) -> Any:
        return await self._inner.read_stream(stream_id)

    async def read_by_correlation(self, correlation_id: str) -> Any:
        return await self._inner.read_by_correlation(correlation_id)

    async def read_all(self, limit: int | None = None) -> Any:
        return await self._inner.read_all(limit=limit)


async def test_one_failing_order_does_not_wedge_the_loop(clock: FixedClock) -> None:
    inner = InMemoryEventStore(clock)
    await _seed_submitted_order(inner, clock, "ord-bad")
    await _seed_submitted_order(inner, clock, "ord-good")
    store = _FlakyAppendStore(inner, poison_stream="order-ord-bad")

    early = datetime(2026, 6, 12, 14, 30, tzinfo=UTC)
    late = datetime(2026, 6, 12, 15, 30, tzinfo=UTC)
    broker = _FakeBroker(
        [
            _order(
                client_order_id="ord-bad", side=OrderSide.BUY, fill_qty="10", submitted_at=early
            ),
            _order(client_order_id="ord-good", side=OrderSide.BUY, fill_qty="7", submitted_at=late),
        ]
    )

    report = await SyncOrders(
        broker=broker,
        event_store=store,
        clock=clock,
        sleeve_id=_SLEEVE,
    ).execute()

    assert report.failed_orders == 1  # ord-bad raised
    assert report.checked_orders == 2
    # The good order behind the bad one was still synced.
    assert positions_from_events(await inner.read_all(), sleeve_id=_SLEEVE) == {
        "AAPL": Decimal("7")
    }
