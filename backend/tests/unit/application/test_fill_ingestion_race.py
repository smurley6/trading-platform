"""Poller (SyncOrders) + real-time stream (ApplyTradeUpdate) recording the SAME fill must
record it EXACTLY ONCE — no position doubling (review F9).

Both paths are read-log -> compute increment -> append. The compare-and-set append makes the
loser of a race conflict on the order stream's version; it then re-reads, sees the winner's
fill, computes a zero increment, and no-ops. `_RaceStore` makes that race deterministic: it
serves a frozen (pre-fill) snapshot to the second writer until its conflicting append thaws it,
exactly modelling two writers that both read before either wrote.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.apply_trade_update import ApplyTradeUpdate
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.ports.event_store import ExpectedVersionError
from trading.application.ports.trade_updates import TradeUpdate
from trading.application.projections import held_positions_from_events
from trading.application.sync_orders import SyncOrders
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType
from trading.infrastructure.persistence.memory import InMemoryEventStore

_SLEEVE = "sleeve-political"
_COID = "ord-race"


class _RaceStore:
    """Serves a frozen snapshot while `stale`, thaws itself when a CAS append conflicts."""

    def __init__(self, inner: InMemoryEventStore) -> None:
        self._inner = inner
        self._frozen: list[StoredEvent] | None = None

    def freeze(self, snapshot: Sequence[StoredEvent]) -> None:
        self._frozen = list(snapshot)

    async def append(
        self, event: DomainEvent, *, expected_version: int | None = None
    ) -> StoredEvent:
        try:
            return await self._inner.append(event, expected_version=expected_version)
        except ExpectedVersionError:
            self._frozen = None  # the loser's next read now sees the winner's fill
            raise

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_stream(stream_id)

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_by_correlation(correlation_id)

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        if self._frozen is not None:
            return list(self._frozen)
        return await self._inner.read_all(limit=limit)


class _FilledBroker:
    def __init__(self, order: BrokerOrder) -> None:
        self._order = order

    async def submit_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return [self._order]

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return []

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"))


async def _seed_submitted(store: InMemoryEventStore, clock: FixedClock) -> None:
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


def _fill_update() -> TradeUpdate:
    return TradeUpdate(
        client_order_id=_COID,
        broker_order_id="brk-1",
        event="fill",
        status=OrderStatus.FILLED,
        symbol="AAPL",
        side=OrderSide.BUY,
        occurred_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
        fill_price=Decimal("100"),
        fill_quantity=Decimal("10"),
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )


def _filled_order() -> BrokerOrder:
    fill = Fill(
        client_order_id=_COID,
        symbol="AAPL",
        side=OrderSide.BUY,
        price=Decimal("100"),
        quantity=Decimal("10"),  # cumulative broker total
        fees=Decimal("0"),
        filled_at=datetime(2026, 6, 12, 14, 30, tzinfo=UTC),
    )
    return BrokerOrder(
        client_order_id=_COID,
        broker_order_id="brk-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        fills=(fill,),
    )


async def test_stream_then_poller_same_fill_records_once(clock: FixedClock) -> None:
    # The stream wins the race (records the fill); the poller then reads a STALE (pre-fill)
    # snapshot, computes the same 10-share increment, and its CAS append conflicts on the order
    # stream version. It re-reads the now-live log, sees increment 0, and records nothing.
    inner = InMemoryEventStore(clock)
    await _seed_submitted(inner, clock)
    store = _RaceStore(inner)

    pre_fill_snapshot = await inner.read_all()

    apply = ApplyTradeUpdate(event_store=store, clock=clock, sleeve_id=_SLEEVE)
    result = await apply.execute(_fill_update())
    assert result.recorded_fill is True

    # Now the poller runs against the SAME state the stream started from (both read pre-fill).
    store.freeze(pre_fill_snapshot)
    sync = SyncOrders(
        broker=_FilledBroker(_filled_order()),
        event_store=store,
        clock=clock,
        sleeve_id=_SLEEVE,
    )
    report = await sync.execute()
    assert report.fills_emitted == 0  # conflict -> re-read -> increment 0 -> no double

    held = held_positions_from_events(await inner.read_all())
    assert len(held) == 1
    assert held[0].quantity == Decimal("10")  # recorded exactly once, not 20


async def test_sequential_cross_path_dedupe_is_cumulative(clock: FixedClock) -> None:
    # Even without a forced race, the cumulative model dedupes across paths: the poller records
    # 10, the stream re-delivers the same cumulative 10 and records nothing.
    inner = InMemoryEventStore(clock)
    await _seed_submitted(inner, clock)

    sync = SyncOrders(
        broker=_FilledBroker(_filled_order()),
        event_store=inner,
        clock=clock,
        sleeve_id=_SLEEVE,
    )
    assert (await sync.execute()).fills_emitted == 1

    apply = ApplyTradeUpdate(event_store=inner, clock=clock, sleeve_id=_SLEEVE)
    assert (await apply.execute(_fill_update())).recorded_fill is False

    held = held_positions_from_events(await inner.read_all())
    assert held[0].quantity == Decimal("10")
