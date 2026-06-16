"""Signals: typed, scored observations emitted by ingestion pipelines."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from trading.domain.provenance import Provenance


class SignalSource(StrEnum):
    STATEMENT = "statement"
    CONTRACT = "contract"
    DISCLOSED_TRADE = "disclosed_trade"


@dataclass(frozen=True, slots=True)
class Signal:
    id: str
    created_at: datetime
    source_type: SignalSource
    ticker: str
    politician_id: str | None
    sentiment: float  # -1 (bearish) .. +1 (bullish)
    politician_weight: float
    source_reliability: float  # 0 .. 1
    raw_payload: dict[str, Any] = field(default_factory=dict)
    # Origin trust level. Defaults to SYNTHETIC (fail-closed): a Signal that does not explicitly
    # declare itself REAL can never drive a real order. Only live data adapters stamp REAL.
    provenance: Provenance = Provenance.SYNTHETIC

    def __post_init__(self) -> None:
        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError("sentiment must be in [-1, 1]")
        if not 0.0 <= self.source_reliability <= 1.0:
            raise ValueError("source_reliability must be in [0, 1]")
        if self.politician_weight < 0:
            raise ValueError("politician_weight must be non-negative")
