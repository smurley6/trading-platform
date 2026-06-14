"""held_positions_from_events: open positions with cost basis, grouped per sleeve."""

from datetime import UTC, datetime
from decimal import Decimal

from trading.application.projections import held_positions_from_events
from trading.domain.events import DomainEvent, EventType, StoredEvent

_WHEN = datetime(2026, 6, 13, 14, 0, tzinfo=UTC)


def _fill_event(
    *, sleeve_id: str, symbol: str, side: str, price: str, quantity: str, seq: int
) -> StoredEvent:
    event = DomainEvent(
        stream_id=f"order-{seq}",
        event_type=EventType.ORDER_FILLED,
        payload={
            "client_order_id": f"ord-{seq}",
            "sleeve_id": sleeve_id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "fees": "0",
            "filled_at": _WHEN.isoformat(),
        },
        occurred_at=_WHEN,
        correlation_id="c1",
    )
    return StoredEvent(event=event, seq=seq, recorded_at=_WHEN)


def test_averages_cost_basis_across_buys() -> None:
    events = [
        _fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="100", quantity="10", seq=1),
        _fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="120", quantity="10", seq=2),
    ]
    [held] = held_positions_from_events(events)
    assert held.symbol == "AAPL"
    assert held.sleeve_id == "s1"
    assert held.quantity == Decimal("20")
    assert held.avg_price == Decimal("110")


def test_sells_reduce_and_closed_positions_drop_out() -> None:
    events = [
        _fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="100", quantity="10", seq=1),
        _fill_event(sleeve_id="s1", symbol="AAPL", side="sell", price="130", quantity="10", seq=2),
    ]
    assert held_positions_from_events(events) == []


def test_positions_are_scoped_per_sleeve() -> None:
    events = [
        _fill_event(sleeve_id="s1", symbol="AAPL", side="buy", price="100", quantity="5", seq=1),
        _fill_event(sleeve_id="s2", symbol="AAPL", side="buy", price="200", quantity="3", seq=2),
    ]
    held = sorted(held_positions_from_events(events), key=lambda p: p.sleeve_id)
    assert [(h.sleeve_id, h.quantity, h.avg_price) for h in held] == [
        ("s1", Decimal("5"), Decimal("100")),
        ("s2", Decimal("3"), Decimal("200")),
    ]


def test_ignores_non_fill_events() -> None:
    other = StoredEvent(
        event=DomainEvent(
            stream_id="x",
            event_type=EventType.ORDER_INTENT,
            payload={"client_order_id": "o", "symbol": "AAPL"},
            occurred_at=_WHEN,
            correlation_id="c1",
        ),
        seq=1,
        recorded_at=_WHEN,
    )
    assert held_positions_from_events([other]) == []
