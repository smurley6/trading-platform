"""Per-venue reconciliation split by sleeve mode (review F15).

Once a sleeve goes LIVE, its fills enter the projection but the OLD single reconciler diffed the
WHOLE projection against the PAPER book — so every live position read back as spurious drift and
the live account was never reconciled. The per-venue reconcilers split the local projection by
mode: the PAPER reconciler sums paper-mode sleeves vs the paper book, the LIVE reconciler sums
live-mode sleeves vs the live book. Both come out clean; neither manufactures false drift.
"""

from collections.abc import Sequence
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.ports.broker import BrokerAccount, BrokerOrder, BrokerPosition
from trading.application.reconcile import Reconcile
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import OrderSide
from trading.infrastructure.persistence.memory import InMemoryEventStore


class _Book:
    """A broker book holding a fixed set of positions."""

    def __init__(self, positions: dict[str, Decimal]) -> None:
        self._positions = positions

    async def submit_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, **_: object) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [BrokerPosition(sym, qty, Decimal("100")) for sym, qty in self._positions.items()]

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(cash=Decimal("0"), equity=Decimal("0"), buying_power=Decimal("0"))


async def _fill(
    store: InMemoryEventStore,
    clock: FixedClock,
    *,
    sleeve_id: str,
    coid: str,
    symbol: str,
    qty: str,
) -> None:
    await store.append(
        DomainEvent(
            stream_id=f"order-{coid}",
            event_type=EventType.ORDER_FILLED,
            payload={
                "client_order_id": coid,
                "sleeve_id": sleeve_id,
                "symbol": symbol,
                "side": OrderSide.BUY.value,
                "price": "100",
                "quantity": qty,
                "fees": "0",
                "filled_at": clock.now().isoformat(),
            },
            occurred_at=clock.now(),
            correlation_id="c",
        )
    )


async def test_live_sleeve_reconciles_against_the_live_book_no_false_paper_drift(
    clock: FixedClock,
) -> None:
    store = InMemoryEventStore(clock)
    # sleeve-political is PAPER (holds AAPL 5 in the paper book); sleeve-testing is LIVE (holds
    # MSFT 3 in the live book). The projection carries BOTH sleeves' fills.
    await _fill(store, clock, sleeve_id="sleeve-political", coid="p-1", symbol="AAPL", qty="5")
    await _fill(store, clock, sleeve_id="sleeve-testing", coid="l-1", symbol="MSFT", qty="3")

    modes = {"sleeve-political": "paper", "sleeve-testing": "live"}

    paper_reconcile = Reconcile(
        broker=_Book({"AAPL": Decimal("5")}),
        event_store=store,
        clock=clock,
        sleeve_ids=lambda: [s for s, m in modes.items() if m == "paper"],
        venue=None,
    )
    live_reconcile = Reconcile(
        broker=_Book({"MSFT": Decimal("3")}),
        event_store=store,
        clock=clock,
        sleeve_ids=lambda: [s for s, m in modes.items() if m == "live"],
        venue="live",
    )

    paper_report = await paper_reconcile.execute()
    live_report = await live_reconcile.execute()

    # Each venue only reconciles its own sleeves against its own book -> both clean.
    assert paper_report.clean, paper_report.drifts
    assert live_report.clean, live_report.drifts
    # The live venue's records are tagged so its drift history is independent of the paper venue.
    assert live_report.sleeve_id == "live"
    assert paper_report.sleeve_id is None


async def test_old_pooled_reconcile_would_falsely_drift_the_live_position(
    clock: FixedClock,
) -> None:
    """Regression contrast: diffing the WHOLE projection against the paper book flags the live
    position as drift — the exact bug the per-venue split fixes."""
    store = InMemoryEventStore(clock)
    await _fill(store, clock, sleeve_id="sleeve-political", coid="p-1", symbol="AAPL", qty="5")
    await _fill(store, clock, sleeve_id="sleeve-testing", coid="l-1", symbol="MSFT", qty="3")

    pooled = Reconcile(
        broker=_Book({"AAPL": Decimal("5")}),  # paper book only
        event_store=store,
        clock=clock,
        sleeve_id=None,  # sums ALL sleeves (the old behaviour)
    )
    report = await pooled.execute()

    assert not report.clean
    assert ("position", "MSFT") in {(d.kind, d.key) for d in report.drifts}
