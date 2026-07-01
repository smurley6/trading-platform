"""Testing strategy: a clean drop-in slot for a second algorithm.

This is an INERT placeholder. It backs the `sleeve-testing` sleeve so the second algo is fully
wired into the shared capital pool — it routes to the same one Alpaca book, its orders are
tagged `sleeve-testing`, and it shows up in the capital-pool attribution — but it emits NO
proposals, so it never places a trade. Drop a real thesis in by replacing `evaluate` below
(match the `Strategy` protocol); everything downstream (routing, risk, reconciliation, UI
attribution) already works with no other changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from trading.domain.positions import Portfolio
from trading.domain.signals import Signal
from trading.strategies.base import StrategyConfig, StrategyProposal
from trading.strategies.registry import register

STRATEGY_ID = "testing"


@dataclass
class PlaceholderStrategy:
    # Named to avoid pytest's "Test*" auto-collection; the drop-in slot's id is STRATEGY_ID.
    strategy_id: str = field(default=STRATEGY_ID, init=False)

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        # Inert by design: no proposals => no orders. Replace this body with a real strategy.
        return []


@register(STRATEGY_ID)
def _build(config: StrategyConfig) -> PlaceholderStrategy:
    # The drop-in slot takes no config until a real strategy needs one.
    return PlaceholderStrategy()
