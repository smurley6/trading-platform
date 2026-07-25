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


# --- Task 10 (docs/RESEARCH-PLAN.md): side-aware Form 4 reliability + benched House ------------
#
# These exercise `compute_conviction`'s `source_reliability` term directly with the actual
# production constants (imported from the adapters, not re-typed as literals) — the adapter stamp
# is exercised separately in `tests/unit/infrastructure/test_sec_form4.py` and
# `test_house_disclosures.py`. Phase 6 label run a38d3848-f1eb-4a55-8a6d-15e8f1464f7a (68,242 ok
# labels, 2026-07-07) is the source of these values; see docs/RESEARCH-PLAN.md's "Re-review
# triggers" section for when they may change.


def test_form4_purchase_signal_contributes_at_point_six_reliability(
    make_signal: SignalFactory,
) -> None:
    from trading.infrastructure.data_sources.sec_form4 import RELIABILITY_INSIDER_PURCHASE

    result = compute_conviction(
        "AAPL",
        [
            make_signal(
                sentiment=0.5,
                politician_weight=1.0,
                source_reliability=RELIABILITY_INSIDER_PURCHASE,
                source_type=SignalSource.DISCLOSED_TRADE,
            )
        ],
        window_days=21,
        thresholds=THRESHOLDS,
    )
    assert result.decayed_weighted_sum == pytest.approx(0.5 * 1.0 * RELIABILITY_INSIDER_PURCHASE)
    assert result.decayed_weighted_sum == pytest.approx(0.30)


def test_form4_sale_signal_contributes_exactly_zero(make_signal: SignalFactory) -> None:
    from trading.infrastructure.data_sources.sec_form4 import RELIABILITY_INSIDER_SALE

    assert RELIABILITY_INSIDER_SALE == 0.0
    result = compute_conviction(
        "AAPL",
        [
            make_signal(
                sentiment=-0.5,
                politician_weight=1.0,
                source_reliability=RELIABILITY_INSIDER_SALE,
                source_type=SignalSource.DISCLOSED_TRADE,
            )
        ],
        window_days=21,
        thresholds=THRESHOLDS,
    )
    assert result.decayed_weighted_sum == 0.0
    assert result.score == 0.0
    assert result.action is ConvictionAction.IGNORE


def test_house_disclosure_signal_contributes_exactly_zero(make_signal: SignalFactory) -> None:
    from trading.infrastructure.data_sources.house_disclosures import (
        RELIABILITY_HOUSE_DISCLOSURE,
    )

    assert RELIABILITY_HOUSE_DISCLOSURE == 0.0
    result = compute_conviction(
        "AAPL",
        [
            make_signal(
                sentiment=0.5,
                politician_weight=1.2,
                source_reliability=RELIABILITY_HOUSE_DISCLOSURE,
                source_type=SignalSource.DISCLOSED_TRADE,
            )
        ],
        window_days=21,
        thresholds=THRESHOLDS,
    )
    assert result.decayed_weighted_sum == 0.0
    assert result.score == 0.0
    assert result.action is ConvictionAction.IGNORE


def test_mixed_buy_sale_house_scores_the_same_as_buy_alone(make_signal: SignalFactory) -> None:
    """A mixed buy+sale+house scenario reaches the score buys alone would produce: sale and house
    contribute zero conviction weight (Task 10), so they add corroborating source-type diversity
    but no weighted-sum contribution — the acceptance criterion is exact score equality, not just
    "the buy still counts"."""
    from trading.infrastructure.data_sources.house_disclosures import (
        RELIABILITY_HOUSE_DISCLOSURE,
    )
    from trading.infrastructure.data_sources.sec_form4 import (
        RELIABILITY_INSIDER_PURCHASE,
        RELIABILITY_INSIDER_SALE,
    )

    buy = make_signal(
        ticker="AAPL",
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=RELIABILITY_INSIDER_PURCHASE,
        source_type=SignalSource.DISCLOSED_TRADE,
    )
    sale = make_signal(
        ticker="AAPL",
        sentiment=-0.5,
        politician_weight=1.0,
        source_reliability=RELIABILITY_INSIDER_SALE,
        source_type=SignalSource.DISCLOSED_TRADE,
    )
    house = make_signal(
        ticker="AAPL",
        sentiment=0.5,
        politician_weight=1.2,
        source_reliability=RELIABILITY_HOUSE_DISCLOSURE,
        source_type=SignalSource.DISCLOSED_TRADE,
    )

    buy_alone = compute_conviction("AAPL", [buy], window_days=21, thresholds=THRESHOLDS)
    mixed = compute_conviction("AAPL", [buy, sale, house], window_days=21, thresholds=THRESHOLDS)

    assert mixed.score == pytest.approx(buy_alone.score)
    assert mixed.decayed_weighted_sum == pytest.approx(buy_alone.decayed_weighted_sum)
    # All three still fed the engine (still "ingested" into the scoring call) even though only
    # the buy contributed weight — corroboration counts DISTINCT SOURCE TYPES, and all three
    # signals here share one source type (DISCLOSED_TRADE), so the corroboration factor is
    # unaffected too; the sale/house signals are inert on both terms, not merely on the sum.
    assert len(mixed.contributing_signal_ids) == 3
    assert mixed.corroboration_factor == pytest.approx(buy_alone.corroboration_factor)


def test_benched_source_type_does_not_corroborate(make_signal: SignalFactory) -> None:
    """A benched (reliability 0.0) signal of a DIFFERENT source type is fully inert: it must
    not lift the corroboration multiplier applied to the active source's score — same score,
    same factor, same distinct source types as the active signal alone (task-10 close-out)."""
    active = make_signal(
        sentiment=0.5,
        politician_weight=2.0,
        source_reliability=0.6,
        source_type=SignalSource.DISCLOSED_TRADE,
    )
    benched = make_signal(
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.0,
        source_type=SignalSource.STATEMENT,
    )

    alone = compute_conviction("AAPL", [active], window_days=21, thresholds=THRESHOLDS)
    mixed = compute_conviction("AAPL", [active, benched], window_days=21, thresholds=THRESHOLDS)

    assert mixed.score == pytest.approx(alone.score)
    assert mixed.corroboration_factor == pytest.approx(alone.corroboration_factor)
    assert mixed.distinct_source_types == alone.distinct_source_types
    assert mixed.distinct_source_types == (SignalSource.DISCLOSED_TRADE,)
    # The benched signal still feeds the audit trail (id + zero contribution) — only its
    # corroboration influence is removed.
    assert len(mixed.contributing_signal_ids) == 2
    assert mixed.contributions[benched.id] == 0.0


def test_all_benched_signals_yield_no_source_types(make_signal: SignalFactory) -> None:
    result = compute_conviction(
        "AAPL",
        [make_signal(sentiment=0.9, politician_weight=2.0, source_reliability=0.0)],
        window_days=21,
        thresholds=THRESHOLDS,
    )
    assert result.distinct_source_types == ()
    assert result.corroboration_factor == 1.0
    assert result.score == 0.0


def test_active_multi_source_still_corroborates_alongside_a_benched_source(
    make_signal: SignalFactory,
) -> None:
    """No over-exclusion: two ACTIVE signals of different source types keep their corroboration
    lift, and adding a benched (0.0) third source type neither adds to nor removes from it — the
    factor and score match the two-active pair alone and still exceed a single active source."""
    active_a = make_signal(sentiment=0.5, politician_weight=1.0, source_type=SignalSource.STATEMENT)
    active_b = make_signal(sentiment=0.5, politician_weight=1.0, source_type=SignalSource.CONTRACT)
    benched = make_signal(
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.0,
        source_type=SignalSource.DISCLOSED_TRADE,
    )

    single = compute_conviction("AAPL", [active_a], window_days=21, thresholds=THRESHOLDS)
    two_active = compute_conviction(
        "AAPL", [active_a, active_b], window_days=21, thresholds=THRESHOLDS
    )
    with_benched = compute_conviction(
        "AAPL", [active_a, active_b, benched], window_days=21, thresholds=THRESHOLDS
    )

    # Two active source types lift above one (corroboration is real when reliability is positive).
    assert two_active.corroboration_factor > single.corroboration_factor
    assert two_active.score > single.score
    # The benched third source is inert: it neither adds nor removes corroboration.
    assert with_benched.corroboration_factor == pytest.approx(two_active.corroboration_factor)
    assert with_benched.score == pytest.approx(two_active.score)
    assert with_benched.distinct_source_types == two_active.distinct_source_types
    assert set(with_benched.distinct_source_types) == {
        SignalSource.STATEMENT,
        SignalSource.CONTRACT,
    }


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
