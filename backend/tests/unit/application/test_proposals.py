"""Trade proposals: conviction scan (TRADE -> proposals, WATCH -> predicted) + execute path."""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import cast

import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.evaluate_signals import EvaluateSignals, EvaluationResult
from trading.application.ports.read_model import PortfolioSnapshotRecord, SignalRecord
from trading.application.proposals import ExecuteProposals, ScanProposals
from trading.domain.conviction import ConvictionParams, ConvictionThresholds
from trading.domain.positions import HeldPosition
from trading.domain.signals import Signal
from trading.strategies.political_signal.strategy import PoliticalSignalStrategy


class _FakeReadModel:
    """Minimal ReadModelPort: only signals_in_window + held_positions are exercised."""

    def __init__(self, signals: Sequence[Signal], held: Sequence[HeldPosition] = ()) -> None:
        self._signals = list(signals)
        self._held = list(held)

    async def held_positions(self) -> Sequence[HeldPosition]:
        return self._held

    async def order_statuses(self, *, sleeve_id: str | None = None) -> dict[str, str]:
        return {}

    async def signals_in_window(self, *, now: datetime, window_days: int) -> Sequence[Signal]:
        return self._signals

    async def recent_signals(self, *, limit: int) -> Sequence[SignalRecord]:
        raise NotImplementedError

    async def all_signals(self) -> Sequence[Signal]:
        return self._signals

    async def latest_portfolio_snapshots(self) -> Sequence[PortfolioSnapshotRecord]:
        raise NotImplementedError


def _scanner(read_model: _FakeReadModel, clock: FixedClock, *, top_n: int = 10) -> ScanProposals:
    return ScanProposals(
        strategy=PoliticalSignalStrategy(
            thresholds=ConvictionThresholds(), window_days=21, conviction_params=ConvictionParams()
        ),
        read_model=read_model,
        clock=clock,
        thresholds=ConvictionThresholds(),
        conviction_params=ConvictionParams(),
        window_days=21,
        top_n=top_n,
    )


async def test_scan_tiers_trade_to_proposals_and_watch_to_predicted(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # score = sentiment * politician_weight * source_reliability * scale(25); thresholds 70 / 40.
    signals = [
        make_signal(ticker="LMT", sentiment=1.0, politician_weight=3.0),  # 75 -> TRADE
        make_signal(ticker="NOC", sentiment=1.0, politician_weight=2.0),  # 50 -> WATCH
        make_signal(ticker="ZZZ", sentiment=1.0, politician_weight=0.5),  # 12.5 -> IGNORE
    ]
    scan = await _scanner(_FakeReadModel(signals), clock).execute()

    assert [p.symbol for p in scan.proposals] == ["LMT"]
    proposal = scan.proposals[0]
    assert proposal.side == "buy"
    assert proposal.action == "trade"
    assert proposal.quantity == Decimal("10")  # strategy position_size
    assert proposal.conviction_score >= 70

    assert [p.symbol for p in scan.predicted] == ["NOC"]
    assert scan.predicted[0].action == "watch"
    # IGNORE-tier ticker appears in neither list.
    seen = {p.symbol for p in scan.proposals} | {p.symbol for p in scan.predicted}
    assert "ZZZ" not in seen


async def test_scan_caps_proposals_at_top_n(clock: FixedClock, make_signal: SignalFactory) -> None:
    signals = [
        make_signal(ticker=t, sentiment=1.0, politician_weight=3.0) for t in ("AAA", "BBB", "CCC")
    ]
    scan = await _scanner(_FakeReadModel(signals), clock, top_n=2).execute()
    assert len(scan.proposals) == 2


async def test_none_ticker_signal_excluded_from_scan(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    """A None-ticker (unresolved-entity, Task 04b) signal must not crash the scan's ticker
    grouping (`sorted({s.ticker for s in window})` in both the proposals and predicted loops)
    and must not appear as a proposal/predicted symbol."""
    signals = [
        make_signal(ticker=None, sentiment=1.0, politician_weight=3.0),  # would-be TRADE tier
        make_signal(ticker="LMT", sentiment=1.0, politician_weight=3.0),  # 75 -> TRADE
        make_signal(ticker=None, sentiment=1.0, politician_weight=2.0),  # would-be WATCH tier
        make_signal(ticker="NOC", sentiment=1.0, politician_weight=2.0),  # 50 -> WATCH
    ]
    scan = await _scanner(_FakeReadModel(signals), clock).execute()

    assert [p.symbol for p in scan.proposals] == ["LMT"]
    assert [p.symbol for p in scan.predicted] == ["NOC"]
    seen = {p.symbol for p in scan.proposals} | {p.symbol for p in scan.predicted}
    assert None not in seen


async def test_predicted_candidate_excludes_benched_source_types(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    """The predicted-candidate path reports only source types with positive reliability. A
    benched (0.0-reliability) signal of a DIFFERENT source type sharing the ticker must neither
    appear in ``distinct_source_types`` nor lift the WATCH-tier score (task-10 close-out).

    Both the score used for tiering and the reported source types flow from
    ``compute_conviction``, so the domain's benched-source exclusion governs this path."""
    from trading.domain.signals import SignalSource

    # An active WATCH-tier signal (1.0 * 2.0 * 1.0 * scale(25) = 50 -> WATCH) plus a benched
    # signal of a different source type on the same ticker.
    signals = [
        make_signal(
            ticker="NOC",
            sentiment=1.0,
            politician_weight=2.0,
            source_type=SignalSource.DISCLOSED_TRADE,
        ),
        make_signal(
            ticker="NOC",
            sentiment=1.0,
            politician_weight=2.0,
            source_reliability=0.0,
            source_type=SignalSource.STATEMENT,
        ),
    ]
    active_only = [signals[0]]

    scan = await _scanner(_FakeReadModel(signals), clock).execute()
    baseline = await _scanner(_FakeReadModel(active_only), clock).execute()

    assert [p.symbol for p in scan.predicted] == ["NOC"]
    predicted = scan.predicted[0]
    # Benched STATEMENT must not appear; only the active DISCLOSED_TRADE type is reported.
    assert predicted.distinct_source_types == ("disclosed_trade",)
    # Score is unchanged vs. the active signal alone — no corroboration lift from the benched one.
    assert predicted.conviction_score == pytest.approx(baseline.predicted[0].conviction_score)


async def test_held_target_is_not_re_proposed(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # Already holding the strategy's target size -> no delta -> not a proposal (and not predicted).
    signals = [make_signal(ticker="LMT", sentiment=1.0, politician_weight=3.0)]
    held = [
        HeldPosition(
            symbol="LMT",
            sleeve_id="sleeve-political",
            quantity=Decimal("10"),
            avg_price=Decimal("100"),
        )
    ]
    scan = await _scanner(_FakeReadModel(signals, held), clock).execute()
    assert scan.proposals == ()
    assert "LMT" not in {p.symbol for p in scan.predicted}


def _form4_buy(
    sid: str, *, clock: FixedClock, ticker: str = "LUV", owner: str = "OWNER A", accession: str
) -> Signal:
    """A Form 4 purchase strong enough to clear TRADE alone: 0.5 * 10 * 0.6 * 25 = 75."""
    from trading.domain.provenance import Provenance
    from trading.domain.signals import SignalSource

    return Signal(
        id=sid,
        created_at=clock.now(),
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=None,
        sentiment=0.5,
        politician_weight=10.0,
        source_reliability=0.6,
        raw_payload={"owner": owner, "accession": accession},
        provenance=Provenance.REAL,
    )


async def test_scan_caps_solo_form4_one_off_to_predicted(clock: FixedClock) -> None:
    # A lone insider buy scores TRADE-tier on the formula (75) but has no second same-ticker filer
    # in the trailing 7d, so the one-off gate demotes it: no proposal, a WATCH-tier predicted at 69.
    solo = _form4_buy("f1", clock=clock, accession="acc-1")
    scan = await _scanner(_FakeReadModel([solo]), clock).execute()

    assert "LUV" not in {p.symbol for p in scan.proposals}
    predicted = {p.symbol: p for p in scan.predicted}
    assert "LUV" in predicted
    assert predicted["LUV"].action == "watch"
    assert predicted["LUV"].conviction_score == pytest.approx(69.0)


async def test_scan_two_owner_form4_buys_lift_the_cap_to_a_proposal(clock: FixedClock) -> None:
    # A second, independent filer on the same ticker in the trailing 7d lifts the one-off cap, so
    # the ticker proposes a trade again.
    signals = [
        _form4_buy("f1", clock=clock, owner="ALPHA", accession="acc-1"),
        _form4_buy("f2", clock=clock, owner="BETA", accession="acc-2"),
    ]
    scan = await _scanner(_FakeReadModel(signals), clock).execute()

    assert "LUV" in {p.symbol for p in scan.proposals}
    assert "LUV" not in {p.symbol for p in scan.predicted}


class _RecordingEvaluator:
    """Captures the execute() call shape so we can assert the filter + place flag."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        signals: Sequence[Signal],
        *,
        correlation_id: str | None = None,
        only_symbols: object = None,
        place: bool = True,
    ) -> EvaluationResult:
        self.calls.append({"only_symbols": only_symbols, "place": place})
        return EvaluationResult(correlation_id="exec-1", decisions=(), orders=())


def _executor(
    evaluator: _RecordingEvaluator, read_model: _FakeReadModel, clock: FixedClock
) -> ExecuteProposals:
    return ExecuteProposals(
        evaluators={"sleeve-political": cast(EvaluateSignals, evaluator)},
        default_sleeve_id="sleeve-political",
        read_model=read_model,
        clock=clock,
        window_days=21,
    )


async def test_execute_individual_passes_only_symbols(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    evaluator = _RecordingEvaluator()
    result = await _executor(evaluator, _FakeReadModel([make_signal()]), clock).execute(
        symbols=["AAPL"]
    )
    assert result.correlation_id == "exec-1"
    assert evaluator.calls[0]["only_symbols"] == {"AAPL"}
    assert evaluator.calls[0]["place"] is True


async def test_execute_blanket_passes_no_filter(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    evaluator = _RecordingEvaluator()
    await _executor(evaluator, _FakeReadModel([make_signal()]), clock).execute(symbols=None)
    assert evaluator.calls[0]["only_symbols"] is None


async def test_execute_unknown_sleeve_raises(clock: FixedClock) -> None:
    executor = ExecuteProposals(
        evaluators={},
        default_sleeve_id="missing",
        read_model=_FakeReadModel([]),
        clock=clock,
        window_days=21,
    )
    with pytest.raises(ValueError, match="unknown sleeve"):
        await executor.execute(symbols=["AAPL"])
