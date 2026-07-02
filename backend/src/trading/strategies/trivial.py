"""Trivial M1 strategy: proves the rails end-to-end (spec section 17, M1).

Goes long a fixed number of shares in any ticker whose net signal sentiment is positive,
and exits when net sentiment turns negative. Not an investment thesis.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from trading.domain.decisions import TargetPosition
from trading.domain.positions import Portfolio
from trading.domain.signals import Signal
from trading.strategies.base import StrategyConfig, StrategyProposal
from trading.strategies.registry import register

STRATEGY_ID = "trivial"


@dataclass
class TrivialStrategy:
    strategy_id: str = field(default=STRATEGY_ID, init=False)
    position_size: Decimal = Decimal("1")

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        net_sentiment: dict[str, float] = {}
        for signal in signals:
            if signal.ticker is None:
                # Unresolved-entity signal (Task 04b) — no symbol to accumulate sentiment on.
                continue
            net_sentiment[signal.ticker] = net_sentiment.get(signal.ticker, 0.0) + (
                signal.sentiment * signal.politician_weight * signal.source_reliability
            )

        proposals: list[StrategyProposal] = []
        for ticker, net in sorted(net_sentiment.items()):
            if net > 0:
                proposals.append(StrategyProposal(TargetPosition(ticker, self.position_size)))
            elif net < 0 and portfolio.quantity(ticker) > 0:
                proposals.append(StrategyProposal(TargetPosition(ticker, Decimal("0"))))
        return proposals


@register(STRATEGY_ID)
def _build(config: StrategyConfig) -> TrivialStrategy:
    # The trivial strategy is a fixed-size proof of the rails; it takes no config.
    return TrivialStrategy()
