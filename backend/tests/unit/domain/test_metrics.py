from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading.domain.metrics import (
    EquityPoint,
    benchmark_total_return_pct,
    buy_and_hold_curve,
    cagr_pct,
    max_drawdown_pct,
    period_returns,
    return_volatility,
    round_trips,
    sharpe_ratio,
    sortino_ratio,
    total_return_pct,
    trade_stats,
)
from trading.domain.prices import PriceSeries


@dataclass(frozen=True, slots=True)
class _Fill:
    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    filled_at: datetime


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


# --- period returns -------------------------------------------------------------------


def test_period_returns_basic() -> None:
    values = [Decimal("100"), Decimal("110"), Decimal("99")]
    returns = period_returns(values)
    assert returns == pytest.approx((0.1, -0.1))


def test_period_returns_empty_and_single() -> None:
    assert period_returns([]) == ()
    assert period_returns([Decimal("100")]) == ()


def test_period_returns_guards_zero_prior() -> None:
    assert period_returns([Decimal("0"), Decimal("50")]) == (0.0,)


# --- volatility / sharpe / sortino ----------------------------------------------------


def test_return_volatility_zero_for_flat_returns() -> None:
    assert return_volatility((0.01, 0.01, 0.01)) == 0.0


def test_return_volatility_empty_is_zero() -> None:
    assert return_volatility(()) == 0.0


def test_return_volatility_scales_with_annualization() -> None:
    base = return_volatility((0.02, -0.01, 0.03), annualization=1.0)
    annual = return_volatility((0.02, -0.01, 0.03), annualization=252.0)
    assert base > 0.0
    assert annual == pytest.approx(base * (252.0**0.5))


def test_sharpe_positive_for_steady_gains() -> None:
    assert sharpe_ratio((0.01, 0.02, 0.015, 0.01)) > 0.0


def test_sharpe_zero_when_no_dispersion() -> None:
    assert sharpe_ratio((0.01, 0.01, 0.01)) == 0.0


def test_sharpe_zero_for_empty() -> None:
    assert sharpe_ratio(()) == 0.0


def test_sharpe_unannualized_is_mean_over_std() -> None:
    returns = (0.02, -0.01, 0.03, 0.0)
    assert sharpe_ratio(returns, annualization=1.0) == pytest.approx(0.6324555, rel=1e-4)


def test_sortino_only_penalizes_downside() -> None:
    returns = (0.05, -0.01, 0.04, -0.02)
    # Sortino's downside-only denominator is smaller than Sharpe's full stdev -> larger ratio.
    assert sortino_ratio(returns) > sharpe_ratio(returns) > 0.0


def test_sortino_zero_when_no_downside() -> None:
    assert sortino_ratio((0.01, 0.02, 0.0)) == 0.0


def test_sortino_zero_for_empty() -> None:
    assert sortino_ratio(()) == 0.0


# --- CAGR -----------------------------------------------------------------------------


def test_cagr_one_year_doubling() -> None:
    curve = [
        EquityPoint(datetime(2025, 1, 1, tzinfo=UTC), Decimal("100")),
        EquityPoint(datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=365), Decimal("200")),
    ]
    # ~365 days of 2x -> ~100% CAGR (365.25/365 power barely above 1).
    assert cagr_pct(curve) == pytest.approx(100.0, abs=0.5)


def test_cagr_short_span_is_zero() -> None:
    curve = [
        EquityPoint(datetime(2025, 1, 1, 9, tzinfo=UTC), Decimal("100")),
        EquityPoint(datetime(2025, 1, 1, 15, tzinfo=UTC), Decimal("110")),
    ]
    assert cagr_pct(curve) == 0.0


def test_cagr_too_few_points_is_zero() -> None:
    assert cagr_pct([]) == 0.0
    assert cagr_pct([EquityPoint(_ts(1), Decimal("100"))]) == 0.0


def test_cagr_nonpositive_start_is_zero() -> None:
    curve = [
        EquityPoint(datetime(2025, 1, 1, tzinfo=UTC), Decimal("0")),
        EquityPoint(datetime(2025, 6, 1, tzinfo=UTC), Decimal("100")),
    ]
    assert cagr_pct(curve) == 0.0


# --- round-trip pairing + trade stats -------------------------------------------------


def _fill(symbol: str, side: str, price: str, qty: str, day: int) -> _Fill:
    return _Fill(symbol, side, Decimal(price), Decimal(qty), _ts(day))


def test_round_trips_fifo_single_pair() -> None:
    fills = [_fill("LMT", "buy", "100", "10", 1), _fill("LMT", "sell", "120", "10", 5)]
    trips = round_trips(fills)
    assert len(trips) == 1
    assert trips[0].pnl == Decimal("200")  # (120-100)*10
    assert (trips[0].opened_at, trips[0].closed_at) == (_ts(1), _ts(5))


def test_round_trips_fifo_partial_and_multi_lot() -> None:
    # Two buy lots (10 @ 100, then 5 @ 110); sell 12 closes the first lot fully and 2 of next.
    fills = [
        _fill("LMT", "buy", "100", "10", 1),
        _fill("LMT", "buy", "110", "5", 2),
        _fill("LMT", "sell", "130", "12", 3),
    ]
    trips = round_trips(fills)
    assert len(trips) == 2
    assert trips[0].quantity == Decimal("10") and trips[0].buy_price == Decimal("100")
    assert trips[1].quantity == Decimal("2") and trips[1].buy_price == Decimal("110")
    assert sum((t.pnl for t in trips), Decimal("0")) == Decimal("340")  # 300 + 40


def test_round_trips_open_lot_unrealized_ignored() -> None:
    # Buy 10, sell only 4 -> one round-trip of 4; remaining 6 stays open (no trip).
    fills = [_fill("AAA", "buy", "50", "10", 1), _fill("AAA", "sell", "60", "4", 2)]
    trips = round_trips(fills)
    assert len(trips) == 1
    assert trips[0].quantity == Decimal("4")


def test_round_trips_is_case_insensitive_and_per_symbol() -> None:
    fills = [
        _fill("AAA", "BUY", "10", "1", 1),
        _fill("BBB", "buy", "20", "1", 1),
        _fill("AAA", "Sell", "15", "1", 2),
    ]
    trips = round_trips(fills)
    assert len(trips) == 1 and trips[0].symbol == "AAA"


def test_trade_stats_win_loss_profit_factor() -> None:
    fills = [
        _fill("AAA", "buy", "100", "1", 1),
        _fill("AAA", "sell", "120", "1", 2),  # +20 win
        _fill("BBB", "buy", "100", "1", 1),
        _fill("BBB", "sell", "90", "1", 2),  # -10 loss
        _fill("CCC", "buy", "100", "1", 1),
        _fill("CCC", "sell", "110", "1", 2),  # +10 win
    ]
    stats = trade_stats(round_trips(fills))
    assert (stats.round_trips, stats.wins, stats.losses) == (3, 2, 1)
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.average_win == Decimal("15")  # (20+10)/2
    assert stats.average_loss == Decimal("10")
    assert stats.profit_factor == pytest.approx(3.0)  # 30 / 10
    assert stats.largest_win == Decimal("20")
    assert stats.largest_loss == Decimal("-10")


def test_trade_stats_no_loss_profit_factor_none() -> None:
    fills = [_fill("AAA", "buy", "100", "1", 1), _fill("AAA", "sell", "110", "1", 2)]
    stats = trade_stats(round_trips(fills))
    assert stats.profit_factor is None
    assert stats.losses == 0


def test_trade_stats_empty() -> None:
    stats = trade_stats(())
    assert stats.round_trips == 0 and stats.win_rate == 0.0
    assert stats.average_win == Decimal("0") and stats.profit_factor is None


# --- buy-and-hold benchmark -----------------------------------------------------------


def test_buy_and_hold_single_symbol_doubles() -> None:
    prices = PriceSeries(series={"LMT": ((_ts(1), Decimal("100")), (_ts(5), Decimal("200")))})
    curve = buy_and_hold_curve(["LMT"], prices, [_ts(1), _ts(5)], Decimal("1000"))
    assert curve[0].equity == Decimal("1000")  # 10 shares @ 100
    assert curve[-1].equity == Decimal("2000")  # 10 shares @ 200
    assert benchmark_total_return_pct(curve, Decimal("1000")) == 100.0


def test_buy_and_hold_equal_weight_two_symbols() -> None:
    prices = PriceSeries(
        series={
            "AAA": ((_ts(1), Decimal("100")), (_ts(2), Decimal("110"))),
            "BBB": ((_ts(1), Decimal("50")), (_ts(2), Decimal("50"))),
        }
    )
    curve = buy_and_hold_curve(["AAA", "BBB"], prices, [_ts(1), _ts(2)], Decimal("1000"))
    # 500 each: 5 AAA + 10 BBB. Day 2: 5*110 + 10*50 = 1050.
    assert curve[0].equity == Decimal("1000")
    assert curve[-1].equity == Decimal("1050")


def test_buy_and_hold_skips_unpriceable_symbol() -> None:
    prices = PriceSeries(series={"AAA": ((_ts(1), Decimal("100")), (_ts(2), Decimal("120")))})
    # ZZZ has no price -> only AAA is held; full 1000 goes into AAA.
    curve = buy_and_hold_curve(["AAA", "ZZZ"], prices, [_ts(1), _ts(2)], Decimal("1000"))
    assert curve[-1].equity == Decimal("1200")


def test_buy_and_hold_no_priceable_symbols_holds_cash() -> None:
    curve = buy_and_hold_curve(["ZZZ"], PriceSeries(series={}), [_ts(1), _ts(2)], Decimal("1000"))
    assert [p.equity for p in curve] == [Decimal("1000"), Decimal("1000")]


def test_buy_and_hold_empty_timeline() -> None:
    prices = PriceSeries(series={"AAA": ((_ts(1), Decimal("100")),)})
    assert buy_and_hold_curve(["AAA"], prices, [], Decimal("1000")) == ()


def test_benchmark_total_return_empty_curve_is_zero() -> None:
    assert benchmark_total_return_pct((), Decimal("1000")) == 0.0
