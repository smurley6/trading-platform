"""Unit tests for the real-data backtest seeding helper (FR-BT-1/2).

The seeding helper builds a `BacktestRequest` from an injected `HistoricalPricesPort` plus a
deterministic seeded signal stream. These tests use a stub port (no network) and assert
reproducibility, the burst/corroboration scenario, and that a seeded run actually trades.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading.application.backtest import Backtest
from trading.application.backtest_seeding import (
    SeededScenario,
    build_seeded_request,
    seeded_signals,
)
from trading.application.risk import RiskLimits
from trading.domain.prices import PriceSeries
from trading.domain.signals import SignalSource
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory
from trading.strategies.political_signal import PoliticalSignalStrategy

_START = datetime(2026, 1, 1, tzinfo=UTC)
_END = _START + timedelta(days=21)
_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("50"),
    max_position_concentration_pct=Decimal("90"),
    max_drawdown_pct=Decimal("90"),
)


class _StubHistoricalPrices:
    def __init__(self, series: PriceSeries) -> None:
        self._series = series
        self.calls: list[tuple[tuple[str, ...], datetime, datetime]] = []

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        self.calls.append((tuple(symbols), start, end))
        return self._series


def _rising_series(symbols: Sequence[str]) -> PriceSeries:
    series: dict[str, tuple[tuple[datetime, Decimal], ...]] = {}
    for symbol in symbols:
        points: list[tuple[datetime, Decimal]] = []
        price = Decimal("100")
        when = _START
        while when <= _END:
            points.append((when, price))
            price += Decimal("2")
            when += timedelta(days=1)
        series[symbol] = tuple(points)
    return PriceSeries(series=series)


def test_seeded_signals_are_reproducible() -> None:
    a = seeded_signals(["AAPL", "MSFT"], _START, _END, scenario=SeededScenario(seed=7))
    b = seeded_signals(["AAPL", "MSFT"], _START, _END, scenario=SeededScenario(seed=7))
    assert a == b  # identical ids, timestamps, sentiments


def test_seeded_signals_seed_changes_sentiment() -> None:
    a = seeded_signals(["AAPL"], _START, _END, scenario=SeededScenario(seed=1))
    b = seeded_signals(["AAPL"], _START, _END, scenario=SeededScenario(seed=2))
    assert [s.sentiment for s in a] != [s.sentiment for s in b]


def test_seeded_signals_bursts_corroborate_two_sources() -> None:
    signals = seeded_signals(["AAPL"], _START, _END, scenario=SeededScenario(sources_per_burst=2))
    first_burst = [s for s in signals if s.created_at == _START]
    assert {s.source_type for s in first_burst} == {
        SignalSource.STATEMENT,
        SignalSource.CONTRACT,
    }


def test_seeded_signals_normalizes_and_dedupes_tickers() -> None:
    signals = seeded_signals([" aapl ", "AAPL", ""], _START, _START, scenario=SeededScenario())
    assert {s.ticker for s in signals} == {"AAPL"}


def test_scenario_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="burst_interval_days"):
        SeededScenario(burst_interval_days=0)
    with pytest.raises(ValueError, match="sources_per_burst"):
        SeededScenario(sources_per_burst=9)
    with pytest.raises(ValueError, match="sentiment"):
        SeededScenario(min_sentiment=0.9, max_sentiment=0.5)


async def test_build_seeded_request_uses_injected_prices() -> None:
    stub = _StubHistoricalPrices(_rising_series(["AAPL", "MSFT"]))
    seeded = await build_seeded_request(
        historical_prices=stub,
        strategy=PoliticalSignalStrategy(window_days=21),
        tickers=["msft", "AAPL"],
        start=_START,
        end=_END,
        risk_limits=_LIMITS,
        starting_cash=Decimal("100000"),
        window_days=21,
    )
    assert stub.calls[0][0] == ("AAPL", "MSFT")  # normalized + sorted
    assert seeded.symbols == ("AAPL", "MSFT")
    assert seeded.signal_count == len(seeded.request.signals)
    assert seeded.request.prices is stub._series


async def test_seeded_request_runs_and_trades() -> None:
    stub = _StubHistoricalPrices(_rising_series(["AAPL"]))
    seeded = await build_seeded_request(
        historical_prices=stub,
        strategy=PoliticalSignalStrategy(window_days=21, position_size=Decimal("5")),
        tickers=["AAPL"],
        start=_START,
        end=_END,
        risk_limits=_LIMITS,
        starting_cash=Decimal("100000"),
        window_days=21,
    )
    result = await Backtest(env_factory=InMemoryBacktestEnvFactory()).run(seeded.request)
    # Corroborated bullish bursts on a rising market -> at least one fill and a benchmark.
    assert len(result.trades) >= 1
    assert result.benchmark_total_return_pct > 0.0
    assert result.trade_stats is not None
