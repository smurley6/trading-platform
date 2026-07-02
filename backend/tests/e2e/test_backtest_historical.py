"""Backtest with the injected historical-prices provider (FR-BT).

Shows the backtest pricing off whatever `HistoricalPricesPort` is wired: the offline
synthetic default still produces an equity curve, and a stub provider returning KNOWN bars
makes the backtest price off those bars. The wiring mirrors `main.run_backtest`.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from trading.application.backtest import Backtest, BacktestRequest
from trading.application.risk import RiskLimits
from trading.domain.prices import PriceSeries
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory
from trading.infrastructure.market_data.historical import SyntheticHistoricalPrices
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


class _StubHistoricalPrices:
    """A `HistoricalPricesPort` returning a fixed, known PriceSeries (the spirit of a real
    provider stubbed for the test) and recording how it was called."""

    def __init__(self, series: PriceSeries) -> None:
        self._series = series
        self.calls: list[tuple[tuple[str, ...], datetime, datetime]] = []

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        self.calls.append((tuple(symbols), start, end))
        return self._series


async def _run(provider: object, signals: list[Signal]) -> object:
    # Mirror main.run_backtest: derive symbols + date range, ask the provider, run the backtest.
    symbols = sorted({s.ticker for s in signals if s.ticker is not None})
    times = sorted({s.created_at for s in signals})
    end = times[-1] if times else _ts(1)
    start = min(times[0], end - timedelta(days=180)) if times else end - timedelta(days=180)
    prices = await provider.price_series(symbols, start, end)  # type: ignore[attr-defined]
    backtest = Backtest(env_factory=InMemoryBacktestEnvFactory())
    return await backtest.run(
        BacktestRequest(
            strategy=PoliticalSignalStrategy(window_days=21, position_size=Decimal("10")),
            signals=signals,
            risk_limits=_LIMITS,
            prices=prices,
            starting_cash=Decimal("100000"),
            window_days=21,
        )
    )


async def test_default_synthetic_provider_produces_equity_curve() -> None:
    signals = [
        _signal("LMT", 0.9, SignalSource.STATEMENT, _ts(1)),
        _signal("LMT", 0.8, SignalSource.CONTRACT, _ts(1)),
        _signal("LMT", 0.7, SignalSource.STATEMENT, _ts(5)),
    ]
    provider = SyntheticHistoricalPrices({"LMT": Decimal("450")})

    result = await _run(provider, signals)

    assert len(result.equity_curve) >= 1  # type: ignore[attr-defined]
    assert len(result.trades) == 1  # type: ignore[attr-defined]
    assert result.trades[0].symbol == "LMT"  # type: ignore[attr-defined]


async def test_stub_provider_prices_off_known_bars() -> None:
    signals = [
        _signal("LMT", 0.9, SignalSource.STATEMENT, _ts(1)),
        _signal("LMT", 0.8, SignalSource.CONTRACT, _ts(1)),
        _signal("LMT", 0.7, SignalSource.STATEMENT, _ts(5)),
    ]
    known = PriceSeries(series={"LMT": ((_ts(1), Decimal("100")), (_ts(5), Decimal("120")))})
    provider = _StubHistoricalPrices(known)

    result = await _run(provider, signals)

    # The backtest priced strictly off the stub's bars: buy 10 @ 100, marked @ 120 -> 100,200.
    assert provider.calls[0][0] == ("LMT",)
    assert result.trades[0].price == Decimal("100")  # type: ignore[attr-defined]
    assert result.ending_equity == Decimal("100200")  # type: ignore[attr-defined]


async def test_empty_signals_is_flat_with_synthetic() -> None:
    provider = SyntheticHistoricalPrices({})
    result = await _run(provider, [])
    assert result.trades == ()  # type: ignore[attr-defined]
    assert result.ending_equity == Decimal("100000")  # type: ignore[attr-defined]
