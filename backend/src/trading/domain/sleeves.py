"""Sleeves: per-algorithm sub-portfolios with independent capital, mode, and status."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class SleeveStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class Sleeve:
    id: str
    strategy_id: str
    allocated_capital: Decimal
    mode: TradingMode
    status: SleeveStatus

    def __post_init__(self) -> None:
        if self.allocated_capital <= 0:
            raise ValueError("allocated_capital must be positive")
