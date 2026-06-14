from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.domain.orders import Fill, OrderSide
from trading.domain.positions import Portfolio

_NOW = datetime(2026, 6, 12, tzinfo=UTC)


def _fill(side: OrderSide, price: str, quantity: str) -> Fill:
    return Fill(
        client_order_id="ord-1",
        symbol="AAPL",
        side=side,
        price=Decimal(price),
        quantity=Decimal(quantity),
        fees=Decimal("0"),
        filled_at=_NOW,
    )


def test_buy_creates_position_and_reduces_cash() -> None:
    portfolio = Portfolio(cash=Decimal("1000"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "100", "5"))
    assert portfolio.cash == Decimal("500")
    assert portfolio.quantity("AAPL") == Decimal("5")
    assert portfolio.positions["AAPL"].avg_entry_price == Decimal("100")


def test_second_buy_averages_entry_price() -> None:
    portfolio = Portfolio(cash=Decimal("2000"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "100", "5"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "200", "5"))
    assert portfolio.positions["AAPL"].avg_entry_price == Decimal("150")
    assert portfolio.quantity("AAPL") == Decimal("10")


def test_full_sell_removes_position() -> None:
    portfolio = Portfolio(cash=Decimal("1000"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "100", "5"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.SELL, "110", "5"))
    assert portfolio.cash == Decimal("1050")
    assert "AAPL" not in portfolio.positions


def test_overselling_raises() -> None:
    portfolio = Portfolio(cash=Decimal("1000"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "100", "5"))
    with pytest.raises(ValueError, match="long-only"):
        portfolio.apply_fill(_fill(OrderSide.SELL, "100", "6"))


def test_equity_marks_positions_to_market() -> None:
    portfolio = Portfolio(cash=Decimal("500"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "100", "5"))
    assert portfolio.equity({"AAPL": Decimal("120")}) == Decimal("600")
