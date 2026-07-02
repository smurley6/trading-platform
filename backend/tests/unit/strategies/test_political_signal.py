from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.conftest import SignalFactory
from trading.domain.conviction import ConvictionParams, ConvictionThresholds
from trading.domain.positions import Portfolio, Position
from trading.domain.signals import Signal, SignalSource
from trading.strategies.political_signal import PoliticalSignalStrategy

THRESHOLDS = ConvictionThresholds(trade=70.0, watch=40.0)
_REF = datetime(2026, 6, 12, 14, 30, tzinfo=UTC)


def _strategy(params: ConvictionParams | None = None) -> PoliticalSignalStrategy:
    return PoliticalSignalStrategy(
        thresholds=THRESHOLDS,
        window_days=21,
        position_size=Decimal("10"),
        conviction_params=params or ConvictionParams(),
    )


def _signal(
    *,
    sid: str,
    created_at: datetime,
    sentiment: float,
    politician_weight: float = 1.5,
    source_type: SignalSource = SignalSource.STATEMENT,
    ticker: str = "AAPL",
) -> Signal:
    return Signal(
        id=sid,
        created_at=created_at,
        source_type=source_type,
        ticker=ticker,
        politician_id="pol-1",
        sentiment=sentiment,
        politician_weight=politician_weight,
        source_reliability=1.0,
        raw_payload={},
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


def test_none_ticker_signal_excluded_from_evaluation(make_signal: SignalFactory) -> None:
    """A None-ticker (unresolved-entity, Task 04b) signal must not crash the ticker-grouping
    `sorted({s.ticker ...})` set-build and must not produce a proposal for a "None" symbol."""
    signals = [
        make_signal(ticker=None, sentiment=0.95, politician_weight=3.0),
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
    assert [p.target.symbol for p in proposals] == ["AAPL"]


def test_decay_params_thread_through_strategy() -> None:
    # Two strong corroborated signals; with aggressive decay the older one is heavily
    # downweighted so conviction falls below the trade threshold -> no entry proposed.
    fresh = _signal(sid="fresh", created_at=_REF, sentiment=0.9)
    old = _signal(
        sid="old",
        created_at=_REF - timedelta(days=21),
        sentiment=0.9,
        source_type=SignalSource.CONTRACT,
    )
    portfolio = Portfolio(cash=Decimal("100000"))

    no_decay = _strategy().evaluate([fresh, old], portfolio)
    assert len(no_decay) == 1  # baseline behaviour: TRADE

    decayed = _strategy(ConvictionParams(decay_half_life_days=3.0)).evaluate(
        [fresh, old], portfolio
    )
    assert decayed == []  # decay suppressed the entry


def test_strategy_uses_shared_batch_reference_time() -> None:
    # The strategy anchors decay to the newest signal across the whole batch (here an MSFT
    # signal), so an AAPL signal dated one half-life earlier is decayed relative to it.
    # Without that shared anchor AAPL's own newest signal would set its reference and it
    # would not decay; the suppression below proves the batch-wide anchor is used.
    strategy = _strategy(ConvictionParams(decay_half_life_days=7.0))
    # AAPL: two corroborated signals strong enough to TRADE at full weight, but both dated
    # one half-life before the batch's newest (MSFT) signal -> each weight 0.5.
    aapl_a = _signal(
        sid="aa",
        created_at=_REF - timedelta(days=7),
        sentiment=0.95,
        source_type=SignalSource.STATEMENT,
        ticker="AAPL",
    )
    aapl_b = _signal(
        sid="ab",
        created_at=_REF - timedelta(days=7),
        sentiment=0.95,
        source_type=SignalSource.CONTRACT,
        ticker="AAPL",
    )
    msft_newest = _signal(sid="m", created_at=_REF, sentiment=0.1, ticker="MSFT")
    proposals = strategy.evaluate([aapl_a, aapl_b, msft_newest], Portfolio(cash=Decimal("100000")))
    # At full weight AAPL would TRADE (2*0.95*1.5*1.25*25 = 89). Halved by the shared
    # reference it lands at ~44.6 -> WATCH, so no entry is proposed.
    assert proposals == []
