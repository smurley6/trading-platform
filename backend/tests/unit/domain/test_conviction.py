import pytest

from tests.conftest import SignalFactory
from trading.domain.conviction import (
    ConvictionAction,
    ConvictionThresholds,
    compute_conviction,
)
from trading.domain.politicians import Politician, PoliticianRole
from trading.domain.signals import SignalSource

THRESHOLDS = ConvictionThresholds(trade=70.0, watch=40.0)


def test_weak_signal_is_ignored(make_signal: SignalFactory) -> None:
    result = compute_conviction(
        "AAPL", [make_signal(sentiment=0.1)], window_days=21, thresholds=THRESHOLDS
    )
    assert result.action is ConvictionAction.IGNORE
    assert result.score < 40


def test_strong_corroborated_signals_trade(make_signal: SignalFactory) -> None:
    signals = [
        make_signal(sentiment=0.9, politician_weight=1.5, source_type=SignalSource.STATEMENT),
        make_signal(sentiment=0.8, politician_weight=1.3, source_type=SignalSource.CONTRACT),
    ]
    result = compute_conviction("AAPL", signals, window_days=21, thresholds=THRESHOLDS)
    assert result.action is ConvictionAction.TRADE
    assert len(result.contributing_signal_ids) == 2


def test_corroboration_across_source_types_raises_score(make_signal: SignalFactory) -> None:
    single_source = [
        make_signal(sentiment=0.5, source_type=SignalSource.STATEMENT),
        make_signal(sentiment=0.5, source_type=SignalSource.STATEMENT),
    ]
    multi_source = [
        make_signal(sentiment=0.5, source_type=SignalSource.STATEMENT),
        make_signal(sentiment=0.5, source_type=SignalSource.CONTRACT),
    ]
    one = compute_conviction("AAPL", single_source, window_days=21, thresholds=THRESHOLDS)
    two = compute_conviction("AAPL", multi_source, window_days=21, thresholds=THRESHOLDS)
    assert two.score > one.score


def test_other_tickers_do_not_contribute(make_signal: SignalFactory) -> None:
    result = compute_conviction(
        "MSFT", [make_signal(ticker="AAPL", sentiment=1.0)], window_days=21, thresholds=THRESHOLDS
    )
    assert result.score == 0
    assert result.contributing_signal_ids == ()


def test_score_is_clamped_to_100(make_signal: SignalFactory) -> None:
    signals = [make_signal(sentiment=1.0, politician_weight=10.0) for _ in range(5)]
    result = compute_conviction("AAPL", signals, window_days=21, thresholds=THRESHOLDS)
    assert result.score == 100.0


def test_thresholds_validate_ordering() -> None:
    with pytest.raises(ValueError, match="watch threshold"):
        ConvictionThresholds(trade=40.0, watch=70.0)


def test_politician_influence_weight_is_a_product() -> None:
    politician = Politician(
        id="pol-1",
        name="Test Chair",
        role=PoliticianRole.COMMITTEE_CHAIR,
        committees=("armed_services",),
        party="I",
        base_weight=1.3,
        track_record_score=0.8,
    )
    assert politician.influence_weight(topic_relevance=0.5) == pytest.approx(0.52)
    with pytest.raises(ValueError, match="topic_relevance"):
        politician.influence_weight(topic_relevance=1.5)
