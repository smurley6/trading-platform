from decimal import Decimal

from tests.conftest import SignalFactory
from trading.domain.conviction import ConvictionThresholds
from trading.domain.positions import Portfolio, Position
from trading.domain.signals import SignalSource
from trading.strategies.political_signal import PoliticalSignalStrategy

THRESHOLDS = ConvictionThresholds(trade=70.0, watch=40.0)


def _strategy() -> PoliticalSignalStrategy:
    return PoliticalSignalStrategy(
        thresholds=THRESHOLDS, window_days=21, position_size=Decimal("10")
    )


def test_high_conviction_proposes_entry(make_signal: SignalFactory) -> None:
    # Corroboration across two distinct source types lifts conviction over the threshold.
    signals = [
        make_signal(
            ticker="AAPL",
            sentiment=0.9,
            politician_weight=1.5,
            source_type=SignalSource.STATEMENT,
        ),
        make_signal(
            ticker="AAPL",
            sentiment=0.8,
            politician_weight=1.5,
            source_type=SignalSource.CONTRACT,
        ),
    ]
    proposals = _strategy().evaluate(signals, Portfolio(cash=Decimal("100000")))
    assert len(proposals) == 1
    assert proposals[0].target.symbol == "AAPL"
    assert proposals[0].target.quantity == Decimal("10")
    assert proposals[0].conviction_score is not None
    assert len(proposals[0].contributing_signal_ids) == 2


def test_low_conviction_with_no_position_proposes_nothing(make_signal: SignalFactory) -> None:
    proposals = _strategy().evaluate(
        [make_signal(ticker="AAPL", sentiment=0.05)], Portfolio(cash=Decimal("100000"))
    )
    assert proposals == []


def test_collapsed_conviction_exits_existing_position(make_signal: SignalFactory) -> None:
    portfolio = Portfolio(
        cash=Decimal("100000"),
        positions={"AAPL": Position("AAPL", Decimal("10"), Decimal("100"))},
    )
    # A weak/negative signal puts conviction below the watch threshold -> exit.
    proposals = _strategy().evaluate(
        [make_signal(ticker="AAPL", sentiment=-0.9, politician_weight=1.5)], portfolio
    )
    assert len(proposals) == 1
    assert proposals[0].target.quantity == Decimal("0")


def test_already_at_target_proposes_nothing(make_signal: SignalFactory) -> None:
    portfolio = Portfolio(
        cash=Decimal("100000"),
        positions={"AAPL": Position("AAPL", Decimal("10"), Decimal("100"))},
    )
    signals = [
        make_signal(
            ticker="AAPL",
            sentiment=0.9,
            politician_weight=1.5,
            source_type=SignalSource.STATEMENT,
        ),
        make_signal(
            ticker="AAPL",
            sentiment=0.8,
            politician_weight=1.5,
            source_type=SignalSource.CONTRACT,
        ),
    ]
    # Conviction is TRADE, but we already hold the target size, so no order is proposed.
    assert _strategy().evaluate(signals, portfolio) == []
