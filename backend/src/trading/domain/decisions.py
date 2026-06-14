"""Decisions: a strategy's intended target position, before risk approval."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class RiskVerdict(StrEnum):
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class TargetPosition:
    symbol: str
    quantity: Decimal  # absolute target holding, not a delta

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError("target quantity must be non-negative (long-only)")


@dataclass(frozen=True, slots=True)
class Decision:
    id: str
    strategy_id: str
    sleeve_id: str
    created_at: datetime
    symbol: str
    target_quantity: Decimal
    correlation_id: str
    conviction_score: float | None = None
    contributing_signal_ids: tuple[str, ...] = ()
