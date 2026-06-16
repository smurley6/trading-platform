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
