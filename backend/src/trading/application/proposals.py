"""Trade proposals: a read-only conviction scan + a manual-approval execution path.

In MANUAL cron mode the daily cycle ingests but does not place orders; actionable trades are
surfaced here as PROPOSALS (TRADE-tier conviction) for human approval, alongside PREDICTED /
incoming candidates (WATCH-tier — "about to buy") so the operator has foresight even in AUTO
mode. :class:`ScanProposals` is pure/read-only (no events, no orders); :class:`ExecuteProposals`
reuses the real evaluate -> risk -> place path for the chosen symbols (individual or blanket buy).
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal

from trading.application.evaluate_signals import EvaluateSignals
from trading.application.form4_corroboration import (
    ONE_OFF_CONVICTION_CAP,
    collapse_form4_signals,
    form4_oneoff_capped_tickers,
)
from trading.application.ports.broker import BrokerOrder
from trading.application.ports.clock import ClockPort
from trading.application.ports.read_model import ReadModelPort
from trading.domain.conviction import (
    ConvictionAction,
    ConvictionParams,
    ConvictionThresholds,
    compute_conviction,
)
from trading.domain.positions import Portfolio, Position
from trading.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class ProposalCandidate:
    """An actionable (TRADE-tier) trade awaiting human approval in manual mode."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: Decimal  # abs(target - current)
    target_quantity: Decimal
    current_quantity: Decimal
    conviction_score: float
    action: str  # "trade"
    contributing_signal_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class PredictedCandidate:
    """A WATCH-tier "about to buy" candidate — foresight into likely upcoming trades."""

    symbol: str
    conviction_score: float
    action: str  # "watch"
    distinct_source_types: tuple[str, ...]
    contributing_signal_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProposalScan:
    proposals: tuple[ProposalCandidate, ...]
    predicted: tuple[PredictedCandidate, ...]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    correlation_id: str
    orders: tuple[BrokerOrder, ...]


class ScanProposals:
    """Read-only: scores the corroboration window into actionable proposals (TRADE-tier) and
    predicted/incoming candidates (WATCH-tier). Emits no events and places no orders."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        read_model: ReadModelPort,
        clock: ClockPort,
        thresholds: ConvictionThresholds,
        conviction_params: ConvictionParams,
        window_days: int,
        top_n: int = 10,
    ) -> None:
        self._strategy = strategy
        self._read_model = read_model
        self._clock = clock
        self._thresholds = thresholds
        self._conviction_params = conviction_params
        self._window_days = window_days
        self._top_n = top_n

    async def execute(self) -> ProposalScan:
        raw_window = await self._read_model.signals_in_window(
            now=self._clock.now(), window_days=self._window_days
        )
        # Form 4 episode dedupe (task-11): repeated filings from one filer stream are one
        # corroboration vote, so both the strategy and the predicted-candidate scan below score
        # deduped signals. The one-off gate below is derived from the RAW window (its trailing-7d
        # filer count), independently of the corroboration-window dedupe.
        window = collapse_form4_signals(raw_window)
        capped_tickers = form4_oneoff_capped_tickers(raw_window, now=self._clock.now())
        held = await self._read_model.held_positions()
        portfolio = Portfolio(
            cash=Decimal("0"),
            positions={p.symbol: Position(p.symbol, p.quantity, p.avg_price) for p in held},
        )

        # Proposals: the strategy's actionable TRADE-tier targets vs current holdings.
        proposals: list[ProposalCandidate] = []
        for sp in self._strategy.evaluate(window, portfolio):
            # One-off WATCH gate: a lone Form 4 buy never auto-proposes a trade. It re-surfaces as
            # a WATCH-tier predicted candidate below (capped to ONE_OFF_CONVICTION_CAP there).
            if sp.target.symbol in capped_tickers and sp.target.quantity > 0:
                continue
            current = portfolio.quantity(sp.target.symbol)
            delta = sp.target.quantity - current
            if delta == 0:
                continue
            proposals.append(
                ProposalCandidate(
                    symbol=sp.target.symbol,
                    side="buy" if delta > 0 else "sell",
                    quantity=abs(delta),
                    target_quantity=sp.target.quantity,
                    current_quantity=current,
                    conviction_score=sp.conviction_score or 0.0,
                    action=ConvictionAction.TRADE.value,
                    contributing_signal_ids=sp.contributing_signal_ids,
                    rationale=sp.rationale,
                )
            )
        proposals.sort(key=lambda c: c.conviction_score, reverse=True)
        proposals = proposals[: self._top_n]

        # Predicted/incoming: WATCH-tier tickers not already proposed (foresight, both modes).
        proposed = {c.symbol for c in proposals}
        predicted: list[PredictedCandidate] = []
        # Unresolved-entity signals (ticker=None, Task 04b) are visible in the feed but excluded
        # from proposals/predicted candidates — there is no symbol to trade.
        for ticker in sorted({s.ticker for s in window if s.ticker is not None}):
            if ticker in proposed:
                continue
            result = compute_conviction(
                ticker,
                window,
                window_days=self._window_days,
                thresholds=self._thresholds,
                params=self._conviction_params,
            )
            score = result.score
            action = result.action
            if ticker in capped_tickers:
                # One-off WATCH gate: a lone Form 4 buy is capped below TRADE, so a formula-strong
                # one-off lands here as WATCH instead of proposing a trade.
                score = min(score, ONE_OFF_CONVICTION_CAP)
                action = (
                    ConvictionAction.WATCH
                    if score >= self._thresholds.watch
                    else ConvictionAction.IGNORE
                )
            if action is ConvictionAction.WATCH:
                predicted.append(
                    PredictedCandidate(
                        symbol=ticker,
                        conviction_score=score,
                        action=action.value,
                        distinct_source_types=tuple(s.value for s in result.distinct_source_types),
                        contributing_signal_ids=result.contributing_signal_ids,
                    )
                )
        predicted.sort(key=lambda c: c.conviction_score, reverse=True)
        return ProposalScan(proposals=tuple(proposals), predicted=tuple(predicted))


class ExecuteProposals:
    """Places orders for chosen symbols by reusing the real evaluate -> risk -> place path.

    ``symbols=None`` blanket-executes every current proposal; an explicit collection executes
    just those (individual buys). Delegates to the per-sleeve :class:`EvaluateSignals` with
    ``place=True`` so the full risk gate + audit trail apply exactly as in auto mode.
    """

    def __init__(
        self,
        *,
        evaluators: Mapping[str, EvaluateSignals],
        default_sleeve_id: str,
        read_model: ReadModelPort,
        clock: ClockPort,
        window_days: int,
    ) -> None:
        self._evaluators = evaluators
        self._default_sleeve_id = default_sleeve_id
        self._read_model = read_model
        self._clock = clock
        self._window_days = window_days

    async def execute(
        self, *, symbols: Collection[str] | None = None, sleeve_id: str | None = None
    ) -> ExecutionResult:
        target = sleeve_id or self._default_sleeve_id
        evaluator = self._evaluators.get(target)
        if evaluator is None:
            raise ValueError(f"unknown sleeve '{target}'")
        window = await self._read_model.signals_in_window(
            now=self._clock.now(), window_days=self._window_days
        )
        only = set(symbols) if symbols is not None else None
        result = await evaluator.execute(window, only_symbols=only, place=True)
        return ExecutionResult(correlation_id=result.correlation_id, orders=result.orders)
