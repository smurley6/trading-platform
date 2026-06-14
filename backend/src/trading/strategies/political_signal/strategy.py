"""Political Signal strategy: conviction-driven, long-only target positions.

Pure: it receives the corroboration-window signals and the current portfolio and returns
proposals. It owns no I/O — ingestion, NLP, and persistence happen upstream.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from trading.domain.conviction import (
    ConvictionAction,
    ConvictionThresholds,
    compute_conviction,
)
from trading.domain.decisions import TargetPosition
from trading.domain.positions import Portfolio
from trading.domain.signals import Signal
from trading.strategies.base import StrategyConfig, StrategyProposal
from trading.strategies.registry import register

STRATEGY_ID = "political_signal"


@dataclass
class PoliticalSignalStrategy:
    strategy_id: str = field(default=STRATEGY_ID, init=False)
    thresholds: ConvictionThresholds = field(default_factory=ConvictionThresholds)
    window_days: int = 21
    position_size: Decimal = Decimal("10")

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        tickers = sorted({s.ticker for s in signals})
        proposals: list[StrategyProposal] = []
        for ticker in tickers:
            result = compute_conviction(
                ticker, signals, window_days=self.window_days, thresholds=self.thresholds
            )
            held = portfolio.quantity(ticker)
            if result.action is ConvictionAction.TRADE:
                if held != self.position_size:
                    proposals.append(
                        StrategyProposal(
                            target=TargetPosition(ticker, self.position_size),
                            conviction_score=result.score,
                            contributing_signal_ids=result.contributing_signal_ids,
                            rationale=f"conviction {result.score:.1f} >= trade threshold",
                        )
                    )
            elif result.action is ConvictionAction.IGNORE and held > 0:
                # Conviction collapsed (or turned bearish) — exit the position.
                proposals.append(
                    StrategyProposal(
                        target=TargetPosition(ticker, Decimal("0")),
                        conviction_score=result.score,
                        contributing_signal_ids=result.contributing_signal_ids,
                        rationale=f"conviction {result.score:.1f} below watch threshold; exit",
                    )
                )
            # WATCH: hold whatever we have, propose nothing.
        return proposals


@register(STRATEGY_ID)
def _build(config: StrategyConfig) -> PoliticalSignalStrategy:
    return PoliticalSignalStrategy(thresholds=config.thresholds, window_days=config.window_days)
