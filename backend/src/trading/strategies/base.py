"""Strategy interface (FR-STR-1): consumes signals, emits target positions.

Pure and backtestable — the same code path runs live and in replay (FR-BT-1).

A strategy returns `StrategyProposal`s rather than bare targets so the reasoning behind a
trade (conviction, the signals that drove it) flows into the audit log, satisfying the
"what conviction, what signals" auditability requirement (spec section 12).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from trading.domain.conviction import ConvictionThresholds
from trading.domain.decisions import TargetPosition
from trading.domain.positions import Portfolio
from trading.domain.signals import Signal


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Typed configuration handed to strategy factories by the registry.

    Keeps the strategies layer free of `trading.config`: the composition root translates the
    global `Settings` into this small, explicit object so factories depend only on inward
    (domain) types.
    """

    thresholds: ConvictionThresholds = field(default_factory=ConvictionThresholds)
    window_days: int = 21


@dataclass(frozen=True, slots=True)
class StrategyProposal:
    target: TargetPosition
    conviction_score: float | None = None
    contributing_signal_ids: tuple[str, ...] = ()
    rationale: str = ""


@runtime_checkable
class Strategy(Protocol):
    strategy_id: str

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        """Return absolute target positions. MUST be deterministic and side-effect free."""
        ...
