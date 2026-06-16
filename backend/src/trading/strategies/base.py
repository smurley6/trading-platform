"""Strategy interface (FR-STR-1): consumes signals, emits target positions.

Pure and backtestable — the same code path runs live and in replay (FR-BT-1).

A strategy returns `StrategyProposal`s rather than bare targets so the reasoning behind a
trade (conviction, the signals that drove it) flows into the audit log, satisfying the
"what conviction, what signals" auditability requirement (spec section 12).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from trading.domain.conviction import ConvictionParams, ConvictionThresholds
from trading.domain.decisions import TargetPosition
from trading.domain.positions import Portfolio
from trading.domain.signals import Signal


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Typed configuration handed to strategy factories by the registry.

    Keeps the strategies layer free of `trading.config`: the composition root translates the
    global `Settings` into this small, explicit object so factories depend only on inward
    (domain) types.

    The conviction tunables (decay, corroboration shape) default to values that reproduce
    the original M1 scoring behaviour, so existing wiring and tests are unaffected until the
    owner deliberately tunes them. `conviction_params()` assembles them into a single
    `ConvictionParams` for the conviction engine.
    """

    thresholds: ConvictionThresholds = field(default_factory=ConvictionThresholds)
    window_days: int = 21
    decay_half_life_days: float = 0.0
    corroboration_bonus_per_source: float = 0.25
    min_sources_for_full_corroboration: int = 1
    max_corroboration_factor: float | None = None
    conviction_scale: float = 25.0

    def conviction_params(self) -> ConvictionParams:
        """Bundle the conviction tunables into a `ConvictionParams` value object."""
        return ConvictionParams(
            scale=self.conviction_scale,
            corroboration_bonus_per_source=self.corroboration_bonus_per_source,
            decay_half_life_days=self.decay_half_life_days,
            min_sources_for_full_corroboration=self.min_sources_for_full_corroboration,
            max_corroboration_factor=self.max_corroboration_factor,
        )


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
