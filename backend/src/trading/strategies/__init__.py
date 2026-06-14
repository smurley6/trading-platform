"""Strategy plugins. Strategies are pure: no broker, DB, or network access (FR-STR-1)."""

# Importing the strategy modules runs their @register side effect (FR-STR-2).
from trading.strategies import political_signal as _political_signal  # noqa: F401
from trading.strategies import trivial as _trivial  # noqa: F401
from trading.strategies.base import Strategy, StrategyProposal
from trading.strategies.registry import available, create, register

__all__ = ["Strategy", "StrategyProposal", "available", "create", "register"]
