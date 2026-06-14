"""Political Signal strategy plugin (spec section 2.2, FR-POL).

Reacts to U.S. politicians' statements, government contract awards, and (optionally)
disclosed trades, weighting by role and committee-to-sector relevance. One plugin among
many; the platform is general-purpose.
"""

from trading.strategies.political_signal.strategy import (
    STRATEGY_ID,
    PoliticalSignalStrategy,
)

__all__ = ["STRATEGY_ID", "PoliticalSignalStrategy"]
