from datetime import UTC, datetime
from decimal import Decimal

from trading.domain.metrics import max_drawdown_pct, total_return_pct
from trading.domain.prices import PriceSeries


def test_total_return_positive() -> None:
    assert total_return_pct(Decimal("100"), Decimal("110")) == 10.0


def test_total_return_negative() -> None:
    assert total_return_pct(Decimal("100"), Decimal("80")) == -20.0


def test_total_return_zero_start_is_zero() -> None:
    assert total_return_pct(Decimal("0"), Decimal("50")) == 0.0


def test_max_drawdown_simple_dip() -> None:
    # Peak 120, trough 90 -> 25% drawdown.
    values = [Decimal(v) for v in ("100", "120", "90", "110")]
    assert max_drawdown_pct(values) == 25.0


def test_max_drawdown_monotonic_up_is_zero() -> None:
    values = [Decimal(v) for v in ("100", "101", "105")]
    assert max_drawdown_pct(values) == 0.0


def _ts(day: int) -> datetime:
    return datetime(2026, 6, day, tzinfo=UTC)


def test_price_series_lookup() -> None:
    series = PriceSeries(series={"LMT": ((_ts(1), Decimal("100")), (_ts(5), Decimal("120")))})
    assert series.price_at("LMT", _ts(3)) == Decimal("100")  # latest at/before
    assert series.price_at("LMT", _ts(5)) == Decimal("120")
    assert series.price_at("LMT", _ts(1)) == Decimal("100")


def test_price_series_before_first_returns_earliest() -> None:
    series = PriceSeries(series={"LMT": ((_ts(5), Decimal("120")),)})
    assert series.price_at("LMT", _ts(1)) == Decimal("120")


def test_price_series_unknown_symbol_is_none() -> None:
    assert PriceSeries(series={}).price_at("ZZZ", _ts(1)) is None
