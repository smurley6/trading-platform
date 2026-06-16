"""Backtest performance metrics (FR-BT-2): equity curve, return, drawdown, risk-adjusted
ratios, trade statistics, and a passive buy-and-hold benchmark.

PURE: every function here is deterministic and side-effect free — no I/O, no wall-clock, no
randomness — so the same numbers are produced live and in backtest replay (FR-BT-1). The
inputs are the equity curve and realised fills the harness already records; nothing reaches
back out to a broker, store, or network.

Conventions and assumptions (documented per the design brief):

* **Period returns** are simple per-step returns of the equity curve,
  ``r_i = equity_i / equity_{i-1} - 1``. A curve of ``n`` points yields ``n - 1`` returns.
* **Risk-free rate is 0** for Sharpe/Sortino (the brief fixes this; subtract it from each
  period return to generalise).
* **Annualization** multiplies the mean by ``periods_per_year`` and the stdev by its square
  root. ``periods_per_year`` is configurable; the default ``252`` assumes one equity point
  per trading day (daily bars). Callers replaying at another cadence should pass the matching
  factor.
* **Volatility / stdev** uses the *population* standard deviation (divide by ``n``), which is
  well defined for a single return and avoids a sample-size discontinuity; with many points
  the difference from the sample stdev is negligible.
* **CAGR** compounds the total return over the elapsed wall-clock span of the curve:
  ``(end / start) ** (365.25 / days) - 1``. With under a day of span (or a non-positive
  start) it falls back to ``0.0`` rather than exploding.
* **Round-trip P&L pairing** is FIFO per symbol: each ``sell`` is matched against the oldest
  open ``buy`` lots of the same symbol, realising ``(sell_price - buy_price) * matched_qty``.
  Unmatched buys (open at the end) and unmatched sells (shorts, which this long-only platform
  does not produce) realise nothing. Wins/losses are counted per matched lot.
* **Buy-and-hold benchmark** invests the starting cash equally across the traded universe at
  each symbol's first available price, then marks that fixed share count at every equity-curve
  timestamp via :class:`~trading.domain.prices.PriceSeries`. It is the passive alternative to
  the strategy over the identical timeline.

Divide-by-zero and degenerate inputs are guarded throughout: ratios with zero risk return
``0.0``, and empty inputs return neutral zeros/empties rather than raising.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from typing import Protocol

from trading.domain.prices import PriceSeries

# One equity point per trading day -> 252 trading days a year (FR-BT-2 default).
DEFAULT_ANNUALIZATION = 252.0
_DAYS_PER_YEAR = Decimal("365.25")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    when: datetime
    equity: Decimal


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """One closed lot: a buy quantity later sold, with the realised P&L (FR-BT-2)."""

    symbol: str
    quantity: Decimal
    buy_price: Decimal
    sell_price: Decimal
    opened_at: datetime
    closed_at: datetime

    @property
    def pnl(self) -> Decimal:
        return (self.sell_price - self.buy_price) * self.quantity


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Aggregate statistics over realised round-trips (FR-BT-2)."""

    round_trips: int
    wins: int
    losses: int
    win_rate: float
    average_win: Decimal
    average_loss: Decimal
    profit_factor: float | None
    largest_win: Decimal
    largest_loss: Decimal
    gross_profit: Decimal
    gross_loss: Decimal


class Fill(Protocol):
    """Structural shape :func:`round_trips` pairs over.

    The application-layer ``Trade`` already matches this, so the domain stays free of any
    application import while still pairing the harness's real fills (FR-BT-1).
    """

    @property
    def symbol(self) -> str: ...
    @property
    def side(self) -> str: ...
    @property
    def price(self) -> Decimal: ...
    @property
    def quantity(self) -> Decimal: ...
    @property
    def filled_at(self) -> datetime: ...


def total_return_pct(start: Decimal, end: Decimal) -> float:
    if start <= 0:
        return 0.0
    return float((end - start) / start * 100)


def max_drawdown_pct(values: Sequence[Decimal]) -> float:
    """Largest peak-to-trough decline over the series, as a positive percentage."""
    peak: Decimal | None = None
    worst = Decimal("0")
    for value in values:
        if peak is None or value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak * 100
            if drawdown > worst:
                worst = drawdown
    return float(worst)


def period_returns(equity: Sequence[Decimal]) -> tuple[float, ...]:
    """Simple per-step returns ``equity_i / equity_{i-1} - 1`` (FR-BT-2).

    A curve of ``n`` points yields ``n - 1`` returns. Steps where the prior equity is
    non-positive contribute ``0.0`` (a flat step) rather than dividing by zero.
    """
    returns: list[float] = []
    for prev, curr in pairwise(equity):
        if prev <= 0:
            returns.append(0.0)
        else:
            returns.append(float((curr - prev) / prev))
    return tuple(returns)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _population_stdev(values: Sequence[float], *, mean: float | None = None) -> float:
    """Population standard deviation (divide by ``n``). ``0.0`` for fewer than one value."""
    if not values:
        return 0.0
    mu = _mean(values) if mean is None else mean
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return float(variance**0.5)


def return_volatility(
    returns: Sequence[float], *, annualization: float = DEFAULT_ANNUALIZATION
) -> float:
    """Annualized stdev of period returns (FR-BT-2). ``0.0`` when there is no dispersion."""
    if not returns:
        return 0.0
    return float(_population_stdev(returns) * (annualization**0.5))


def sharpe_ratio(
    returns: Sequence[float], *, annualization: float = DEFAULT_ANNUALIZATION
) -> float:
    """Annualized Sharpe ratio with risk-free rate 0 (FR-BT-2).

    ``sqrt(annualization) * mean(returns) / stdev(returns)``. Returns ``0.0`` when there are
    no returns or the returns have zero dispersion (guards divide-by-zero).
    """
    if not returns:
        return 0.0
    std = _population_stdev(returns)
    if std == 0.0:
        return 0.0
    return float((_mean(returns) / std) * (annualization**0.5))


def sortino_ratio(
    returns: Sequence[float], *, annualization: float = DEFAULT_ANNUALIZATION
) -> float:
    """Annualized Sortino ratio with risk-free rate 0 (FR-BT-2).

    Like Sharpe but the denominator is downside deviation — the root-mean-square of the
    *negative* returns only (zeros for non-negative steps), so upside volatility is not
    penalised. Returns ``0.0`` when there are no returns or no downside dispersion.
    """
    if not returns:
        return 0.0
    downside = [min(0.0, r) for r in returns]
    # RMS of the downside series (mean fixed at 0: deviation below the 0 target).
    downside_dev = (sum(d * d for d in downside) / len(downside)) ** 0.5
    if downside_dev == 0.0:
        return 0.0
    return float((_mean(returns) / downside_dev) * (annualization**0.5))


def cagr_pct(curve: Sequence[EquityPoint]) -> float:
    """Compound annual growth rate of the equity curve, as a percentage (FR-BT-2).

    ``((end / start) ** (365.25 / elapsed_days) - 1) * 100`` over the curve's wall-clock span.
    Falls back to ``0.0`` for a non-positive start, fewer than two points, or under a day of
    span (where annualizing a sub-daily move is meaningless).
    """
    if len(curve) < 2:
        return 0.0
    start = curve[0].equity
    end = curve[-1].equity
    if start <= 0 or end <= 0:
        return 0.0
    span_seconds = (curve[-1].when - curve[0].when).total_seconds()
    elapsed_days = Decimal(str(span_seconds)) / Decimal("86400")
    if elapsed_days < Decimal("1"):
        return 0.0
    years = float(_DAYS_PER_YEAR / elapsed_days)
    ratio = float(end / start)
    return float((ratio**years - 1.0) * 100.0)


@dataclass(slots=True)
class _OpenLot:
    """A still-open buy lot awaiting a sell, used by the FIFO pairing."""

    quantity: Decimal
    price: Decimal
    opened_at: datetime


def round_trips(fills: Sequence[Fill]) -> tuple[RoundTrip, ...]:
    """Pair fills into realised round-trips, FIFO per symbol (FR-BT-2).

    Buys open lots; each later sell of the same symbol consumes the oldest open lots first,
    realising ``(sell_price - buy_price) * matched_qty`` per matched lot. Sides are matched
    case-insensitively against ``"buy"``/``"sell"``. Open lots at the end and any sell
    quantity that exceeds the open inventory (not produced by this long-only platform) are
    dropped. Fills are processed in ``filled_at`` order so the FIFO queue is chronological.
    """
    ordered = sorted(fills, key=lambda f: f.filled_at)
    open_lots: dict[str, list[_OpenLot]] = {}
    closed: list[RoundTrip] = []
    for fill in ordered:
        side = fill.side.lower()
        if side == "buy":
            open_lots.setdefault(fill.symbol, []).append(
                _OpenLot(quantity=fill.quantity, price=fill.price, opened_at=fill.filled_at)
            )
        elif side == "sell":
            remaining = fill.quantity
            lots = open_lots.get(fill.symbol, [])
            while remaining > 0 and lots:
                lot = lots[0]
                matched = min(lot.quantity, remaining)
                closed.append(
                    RoundTrip(
                        symbol=fill.symbol,
                        quantity=matched,
                        buy_price=lot.price,
                        sell_price=fill.price,
                        opened_at=lot.opened_at,
                        closed_at=fill.filled_at,
                    )
                )
                if matched == lot.quantity:
                    lots.pop(0)
                else:
                    lot.quantity -= matched
                remaining -= matched
    return tuple(closed)


def trade_stats(trips: Sequence[RoundTrip]) -> TradeStats:
    """Aggregate win/loss statistics over realised round-trips (FR-BT-2).

    Wins are round-trips with positive P&L, losses with negative P&L (break-even trades count
    toward neither). ``profit_factor`` is gross profit / gross loss, ``None`` when there is no
    loss to divide by (no finite ratio exists). Averages are ``0`` when there are none.
    """
    wins = [t.pnl for t in trips if t.pnl > 0]
    losses = [t.pnl for t in trips if t.pnl < 0]
    gross_profit = sum(wins, Decimal("0"))
    gross_loss = -sum(losses, Decimal("0"))  # reported as a positive magnitude
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided if decided else 0.0
    average_win = gross_profit / len(wins) if wins else Decimal("0")
    average_loss = (gross_loss / len(losses)) if losses else Decimal("0")
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else None
    largest_win = max(wins) if wins else Decimal("0")
    largest_loss = min(losses) if losses else Decimal("0")
    return TradeStats(
        round_trips=len(trips),
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        average_win=average_win,
        average_loss=average_loss,
        profit_factor=profit_factor,
        largest_win=largest_win,
        largest_loss=largest_loss,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )


def buy_and_hold_curve(
    symbols: Sequence[str],
    prices: PriceSeries,
    timeline: Sequence[datetime],
    starting_cash: Decimal,
) -> tuple[EquityPoint, ...]:
    """Passive buy-and-hold benchmark valued along ``timeline`` (FR-BT-2).

    Allocates ``starting_cash`` equally across the priceable symbols, buying a fractional
    share count at each symbol's first available price (the price at ``timeline[0]``), then
    marks that fixed basket at every timestamp in ``timeline`` via
    :meth:`PriceSeries.price_at`. Symbols with no price are skipped and their slice of cash
    is held flat. Returns an empty tuple for an empty timeline.
    """
    if not timeline:
        return ()
    priceable = [s for s in symbols if prices.price_at(s, timeline[0]) is not None]
    n = len(priceable)
    if n == 0:
        # Nothing to hold: the benchmark just parks the cash, flat across the timeline.
        return tuple(EquityPoint(when, starting_cash) for when in timeline)

    per_symbol_cash = starting_cash / Decimal(n)
    shares: dict[str, Decimal] = {}
    for symbol in priceable:
        entry_price = prices.price_at(symbol, timeline[0])
        assert entry_price is not None  # priceable by construction
        shares[symbol] = per_symbol_cash / entry_price

    curve: list[EquityPoint] = []
    for when in timeline:
        value = Decimal("0")
        for symbol in priceable:
            mark = prices.price_at(symbol, when)
            assert mark is not None  # first price exists, so all later marks do too
            value += shares[symbol] * mark
        curve.append(EquityPoint(when, value))
    return tuple(curve)


def benchmark_total_return_pct(curve: Sequence[EquityPoint], starting_cash: Decimal) -> float:
    """Total return of a benchmark equity curve vs. the starting cash, as a percentage."""
    if not curve:
        return 0.0
    return total_return_pct(starting_cash, curve[-1].equity)
