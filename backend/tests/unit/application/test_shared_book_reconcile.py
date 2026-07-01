"""Shared-book (one Alpaca account, many algos) reconciliation.

When both algos trade the ONE shared book, reconciliation MUST be pool-level: the local
projection summed across ALL sleeves (sleeve_id=None) is what the broker's single account
reflects. A per-sleeve reconciler on the shared book would flag every OTHER algo's holdings as
spurious drift — the exact bug the pool-level reconciler avoids.
"""

from collections.abc import Sequence
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.reconcile import Reconcile
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide, OrderType
from trading.infrastructure.persistence.memory import InMemoryEventStore


class SharedBookBroker:
    """One account holding the COMBINED positions of every algo (AAPL 5 + AAPL 3 = 8)."""

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
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [BrokerPosition("AAPL", Decimal("8"), Decimal("100"))]

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("800"), buying_power=Decimal("800"))


async def _record_fill(
    store: InMemoryEventStore, clock: FixedClock, *, sleeve_id: str, coid: str, qty: str
) -> None:
    await store.append(
        DomainEvent(
            stream_id=f"order-{coid}",
            event_type=EventType.ORDER_FILLED,
            payload={
                "client_order_id": coid,
                "sleeve_id": sleeve_id,
                "symbol": "AAPL",
                "side": OrderSide.BUY.value,
                "price": "100",
                "quantity": qty,
                "fees": "0",
                "filled_at": clock.now().isoformat(),
            },
            occurred_at=clock.now(),
            correlation_id="corr-shared",
        )
    )


async def test_pool_reconcile_is_clean_across_algos(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    # Two algos, each with a fill in the ONE shared book.
    await _record_fill(store, clock, sleeve_id="sleeve-political", coid="ord-a", qty="5")
    await _record_fill(store, clock, sleeve_id="sleeve-testing", coid="ord-b", qty="3")

    pool = Reconcile(broker=SharedBookBroker(), event_store=store, clock=clock, sleeve_id=None)
    report = await pool.execute()

    # Summed local (5 + 3 = 8) matches the broker's single account (8) => no drift.
    assert report.clean
    assert report.sleeve_id is None


async def test_per_sleeve_reconcile_would_falsely_drift_on_shared_book(clock: FixedClock) -> None:
    store = InMemoryEventStore(clock)
    await _record_fill(store, clock, sleeve_id="sleeve-political", coid="ord-a", qty="5")
    await _record_fill(store, clock, sleeve_id="sleeve-testing", coid="ord-b", qty="3")

    # A per-sleeve reconciler sees only its own 5 shares but the shared account holds 8 — this is
    # the spurious drift the pool-level reconciler is designed to prevent.
    per_sleeve = Reconcile(
        broker=SharedBookBroker(), event_store=store, clock=clock, sleeve_id="sleeve-political"
    )
    report = await per_sleeve.execute()

    assert not report.clean
    assert ("position", "AAPL") in {(d.kind, d.key) for d in report.drifts}
