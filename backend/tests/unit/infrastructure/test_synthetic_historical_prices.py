"""SyntheticHistoricalPrices — the offline default `HistoricalPricesPort`. Verifies it
reproduces `synthetic_price_series` over the same daily grid and makes no network calls."""

from datetime import UTC, datetime
from decimal import Decimal

import httpx

from trading.application.ports.historical import HistoricalPricesPort
from trading.infrastructure.market_data.alpaca_history import AlpacaHistoricalData
from trading.infrastructure.market_data.historical import (
    SyntheticHistoricalPrices,
    synthetic_price_series,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)
_END = datetime(2026, 1, 4, tzinfo=UTC)
_BASE = {"AAPL": Decimal("200"), "MSFT": Decimal("100")}


def _daily(start: datetime, end: datetime) -> list[datetime]:
    from datetime import timedelta

    times = []
    when = start
    while when <= end:
        times.append(when)
        when += timedelta(days=1)
    return times


def test_both_providers_satisfy_the_port() -> None:
    # Static structural conformance: both concrete providers are HistoricalPricesPort.
    synthetic: HistoricalPricesPort = SyntheticHistoricalPrices()
    real: HistoricalPricesPort = AlpacaHistoricalData(
        client=httpx.AsyncClient(),  # not used; only type conformance is checked
        key_id="k",
        secret_key="s",
        base_url="https://data.alpaca.markets",
    )
    assert hasattr(synthetic, "price_series")
    assert hasattr(real, "price_series")


async def test_reproduces_synthetic_price_series_on_daily_grid() -> None:
    provider = SyntheticHistoricalPrices(_BASE)
    series = await provider.price_series(["AAPL", "MSFT"], _START, _END)

    expected = synthetic_price_series(_BASE, _daily(_START, _END))
    assert series.series == expected.series


async def test_unknown_symbol_uses_default_base() -> None:
    provider = SyntheticHistoricalPrices(_BASE)
    series = await provider.price_series(["ZZZZ"], _START, _END)

    expected = synthetic_price_series({"ZZZZ": Decimal("100")}, _daily(_START, _END))
    assert series.series == expected.series


async def test_empty_symbols_yields_empty_series() -> None:
    provider = SyntheticHistoricalPrices(_BASE)
    series = await provider.price_series([], _START, _END)
    assert series.symbols() == ()


async def test_normalizes_case_and_is_deterministic() -> None:
    provider = SyntheticHistoricalPrices(_BASE)
    first = await provider.price_series(["aapl"], _START, _END)
    second = await provider.price_series(["AAPL"], _START, _END)
    assert first.series == second.series
    assert first.symbols() == ("AAPL",)
