"""Backtest end-to-end (FR-BT): recorded signals replay through the live strategy + risk +
execution path in a synthetic environment; known inputs produce an expected trade list and
metrics. Deterministic — no real clock, no real prices."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from trading.application.backtest import Backtest, BacktestRequest
from trading.application.risk import RiskLimits
from trading.domain.prices import PriceSeries
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory
from trading.strategies.political_signal import PoliticalSignalStrategy

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("50"),
    max_position_concentration_pct=Decimal("90"),
    max_drawdown_pct=Decimal("90"),
)


def _ts(day: int) -> datetime:
    return datetime(2026, 6, day, 14, 0, tzinfo=UTC)


def _signal(ticker: str, sentiment: float, source: SignalSource, when: datetime) -> Signal:
    return Signal(
        id=str(uuid4()),
        created_at=when,
        source_type=source,
        ticker=ticker,
        politician_id="pol-1",
        sentiment=sentiment,
        politician_weight=1.5,
        source_reliability=1.0,
        raw_payload={},
    )


def _backtest() -> Backtest:
    return Backtest(env_factory=InMemoryBacktestEnvFactory())


async def test_uptrend_backtest_buys_and_profits() -> None:
    # Day 1: corroborated strong LMT signals -> conviction trade -> buy 10 @ 100.
    # Day 5: another LMT signal triggers a fresh snapshot while the price has risen to 120.
    signals = [
        _signal("LMT", 0.9, SignalSource.STATEMENT, _ts(1)),
        _signal("LMT", 0.8, SignalSource.CONTRACT, _ts(1)),
        _signal("LMT", 0.7, SignalSource.STATEMENT, _ts(5)),
    ]
    prices = PriceSeries(series={"LMT": ((_ts(1), Decimal("100")), (_ts(5), Decimal("120")))})

    result = await _backtest().run(
        BacktestRequest(
            strategy=PoliticalSignalStrategy(window_days=21, position_size=Decimal("10")),
            signals=signals,
            risk_limits=_LIMITS,
            prices=prices,
            starting_cash=Decimal("100000"),
            window_days=21,
        )
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert (trade.symbol, trade.side, trade.quantity, trade.price) == (
        "LMT",
        "buy",
        Decimal("10"),
        Decimal("100"),
    )
    # Bought 10 @ 100 (cash 99,000), marked at 120 on day 5 -> equity 100,200.
    assert result.ending_equity == Decimal("100200")
    assert result.total_return_pct > 0
    assert result.max_drawdown_pct == 0.0
    assert len(result.equity_curve) == 2


async def test_downtrend_backtest_shows_drawdown() -> None:
    signals = [
        _signal("LMT", 0.9, SignalSource.STATEMENT, _ts(1)),
        _signal("LMT", 0.8, SignalSource.CONTRACT, _ts(1)),
        _signal("LMT", 0.7, SignalSource.STATEMENT, _ts(5)),
    ]
    prices = PriceSeries(series={"LMT": ((_ts(1), Decimal("100")), (_ts(5), Decimal("80")))})

    result = await _backtest().run(
        BacktestRequest(
            strategy=PoliticalSignalStrategy(window_days=21, position_size=Decimal("10")),
            signals=signals,
            risk_limits=_LIMITS,
            prices=prices,
            starting_cash=Decimal("100000"),
            window_days=21,
        )
    )

    # Bought 10 @ 100, marked at 80 -> equity 99,800: a loss and a drawdown.
    assert result.ending_equity == Decimal("99800")
    assert result.total_return_pct < 0
    assert result.max_drawdown_pct > 0


async def test_empty_signals_is_flat() -> None:
    result = await _backtest().run(
        BacktestRequest(
            strategy=PoliticalSignalStrategy(),
            signals=[],
            risk_limits=_LIMITS,
            prices=PriceSeries(series={}),
            starting_cash=Decimal("100000"),
            window_days=21,
        )
    )
    assert result.trades == ()
    assert result.ending_equity == Decimal("100000")
    assert result.total_return_pct == 0.0
