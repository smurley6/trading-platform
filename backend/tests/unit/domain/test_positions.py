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


def test_apply_fill_tolerant_matches_strict_for_a_valid_sell() -> None:
    # For BUYs and valid SELLs the tolerant variant is identical to strict apply_fill and never
    # flags a clamp (review F7).
    portfolio = Portfolio(cash=Decimal("1000"))
    portfolio, clamped = portfolio.apply_fill_tolerant(_fill(OrderSide.BUY, "100", "5"))
    assert clamped is False
    portfolio, clamped = portfolio.apply_fill_tolerant(_fill(OrderSide.SELL, "110", "5"))
    assert clamped is False
    assert "AAPL" not in portfolio.positions
    assert portfolio.cash == Decimal("1050")


def test_apply_fill_tolerant_clamps_an_oversell_instead_of_raising() -> None:
    # A SELL exceeding held (or with no held position at all) must NOT raise in the projection —
    # it is clamped to flat and flagged, so an out-of-order fill can be recorded without aborting
    # the append (review F7). Only the shares actually held are removed.
    portfolio = Portfolio(cash=Decimal("1000"))
    portfolio, _ = portfolio.apply_fill_tolerant(_fill(OrderSide.BUY, "100", "5"))
    portfolio, clamped = portfolio.apply_fill_tolerant(_fill(OrderSide.SELL, "110", "8"))
    assert clamped is True
    assert "AAPL" not in portfolio.positions
    # Cash reflects only the 5 shares actually held and sold at 110, not the phantom 8.
    assert portfolio.cash == Decimal("500") + Decimal("110") * Decimal("5")


def test_apply_fill_tolerant_sell_with_no_position_is_a_flagged_noop() -> None:
    portfolio = Portfolio(cash=Decimal("1000"))
    portfolio, clamped = portfolio.apply_fill_tolerant(_fill(OrderSide.SELL, "110", "3"))
    assert clamped is True
    assert portfolio.positions == {}
    assert portfolio.cash == Decimal("1000")  # nothing held -> nothing sold


def test_equity_marks_positions_to_market() -> None:
    portfolio = Portfolio(cash=Decimal("500"))
    portfolio = portfolio.apply_fill(_fill(OrderSide.BUY, "100", "5"))
    assert portfolio.equity({"AAPL": Decimal("120")}) == Decimal("600")
