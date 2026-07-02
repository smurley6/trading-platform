"""Political Signal strategy: conviction-driven, long-only target positions.

Pure: it receives the corroboration-window signals and the current portfolio and returns
proposals. It owns no I/O — ingestion, NLP, and persistence happen upstream.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from trading.domain.conviction import (
    ConvictionAction,
    ConvictionParams,
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
    conviction_params: ConvictionParams = field(default_factory=ConvictionParams)

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        # Unresolved-entity signals (ticker=None, Task 04b) are excluded from strategy
        # evaluation — there is no symbol to score, size, or place an order against.
        tickers = sorted({s.ticker for s in signals if s.ticker is not None})
        # Anchor time decay to a single deterministic reference shared across tickers:
        # the most recent signal in this evaluation batch. This keeps the per-ticker
        # scores comparable and the whole evaluation reproducible in backtest (FR-BT-1).
        now = max((s.created_at for s in signals), default=None)
        proposals: list[StrategyProposal] = []
        for ticker in tickers:
            result = compute_conviction(
                ticker,
                signals,
                window_days=self.window_days,
                thresholds=self.thresholds,
                params=self.conviction_params,
                now=now,
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
    return PoliticalSignalStrategy(
        thresholds=config.thresholds,
        window_days=config.window_days,
        conviction_params=config.conviction_params(),
    )
