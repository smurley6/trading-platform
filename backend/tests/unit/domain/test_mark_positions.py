"""Pure position-marking math: market value + unrealized P&L, with graceful fallback."""

from decimal import Decimal

from trading.domain.positions import HeldPosition, mark_positions


def _held(symbol: str = "AAPL", qty: str = "10", avg: str = "230.00") -> HeldPosition:
    return HeldPosition(
        symbol=symbol, sleeve_id="sleeve-political", quantity=Decimal(qty), avg_price=Decimal(avg)
    )


def test_marks_value_and_unrealized_pnl() -> None:
    [marked] = mark_positions([_held()], {"AAPL": Decimal("232.12")})

    assert marked.symbol == "AAPL"
    assert marked.sleeve_id == "sleeve-political"
    assert marked.last == Decimal("232.12")
    assert marked.market_value == Decimal("2321.20")
    assert marked.unrealized_pnl == Decimal("21.20")


def test_loss_is_negative() -> None:
    [marked] = mark_positions([_held(avg="240.00")], {"AAPL": Decimal("232.00")})
    assert marked.unrealized_pnl == Decimal("-80.00")


def test_missing_mark_falls_back_to_avg_price_zero_pnl() -> None:
    [marked] = mark_positions([_held()], {})  # no quote available
    assert marked.last == Decimal("230.00")
    assert marked.market_value == Decimal("2300.00")
    assert marked.unrealized_pnl == Decimal("0")


def test_empty_positions() -> None:
    assert mark_positions([], {"AAPL": Decimal("1")}) == []
