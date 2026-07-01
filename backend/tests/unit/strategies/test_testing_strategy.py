"""The `testing` strategy is a registered, inert drop-in slot for a second algorithm.

It must be creatable via the registry (so it wires into the shared pool like any algo) and emit
NO proposals (so the placeholder never trades until a real thesis replaces it).
"""

from decimal import Decimal

from trading.domain.conviction import ConvictionThresholds
from trading.domain.positions import Portfolio
from trading.domain.signals import Signal, SignalSource
from trading.strategies import StrategyConfig, registry
from trading.strategies.testing import PlaceholderStrategy


def _config() -> StrategyConfig:
    return StrategyConfig(thresholds=ConvictionThresholds(trade=65.0, watch=30.0), window_days=14)


def _signal(ticker: str, sentiment: float) -> Signal:
    from datetime import UTC, datetime

    return Signal(
        id=f"sig-{ticker}",
        created_at=datetime(2026, 6, 12, tzinfo=UTC),
        source_type=SignalSource.STATEMENT,
        ticker=ticker,
        politician_id="pol-1",
        sentiment=sentiment,
        politician_weight=1.0,
        source_reliability=1.0,
        raw_payload={},
    )


def test_testing_strategy_is_registered() -> None:
    strategy = registry.create("testing", _config())
    assert isinstance(strategy, PlaceholderStrategy)
    assert strategy.strategy_id == "testing"


def test_testing_strategy_emits_no_proposals() -> None:
    strategy = registry.create("testing", _config())
    # Even a strongly positive signal produces nothing — the slot is inert until replaced.
    proposals = strategy.evaluate([_signal("AAPL", 0.9)], Portfolio(cash=Decimal("100000")))
    assert list(proposals) == []
