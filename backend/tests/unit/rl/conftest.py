"""Shared builders for the RL env tests (task-rl-06).

Plain helper functions (imported as ``from tests.unit.rl.conftest import ...``, the repo's pattern
for ``tests/conftest.py``), not fixtures. The env import is deferred into :func:`make_env` so that
loading this conftest never pulls gymnasium/numpy into the pure-stdlib RL test modules.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from trading.application.rl_dataset import rl_dataset_from_json
from trading.domain.prices import PriceSeries
from trading.rl.bars import BarTable

if TYPE_CHECKING:
    from trading.rl.env import SingleTickerTradingEnv

FROZEN_SPY = Path(__file__).parents[3] / "data" / "rl" / "spy_daily.json"
_NY = ZoneInfo("America/New_York")


def synthetic_random_walk(n: int, seed: int, *, vol: float = 0.01) -> list[float]:
    """``n`` closes of a seeded geometric random walk starting at 100 (``random.Random(seed)``;
    never the global RNG)."""
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0.0003, vol)))
    return closes


def bar_table(closes: Sequence[float], symbol: str = "SPY") -> BarTable:
    """Weekday sessions stamped 00:00 New York (Alpaca's convention), through the ONE
    knowledge-time rule (``from_price_series`` -> ``session_known_at``). ``Decimal(repr(c))``
    round-trips each float exactly, so ``closes()`` returns these closes bit-for-bit."""
    day = date(2020, 1, 2)
    points: list[tuple[datetime, Decimal]] = []
    for close in closes:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        stamp = datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC)
        points.append((stamp, Decimal(repr(close))))
        day += timedelta(days=1)
    return BarTable.from_price_series(PriceSeries(series={symbol: tuple(points)}), symbol)


def with_closes(table: BarTable, closes: Sequence[float]) -> BarTable:
    """``table`` with every bar's close replaced (bit-exactly, via ``repr``) and its
    ``session_ts`` / ``known_at`` kept — so a mutated table has the SAME knowledge times."""
    if len(closes) != len(table):
        raise ValueError(f"need {len(table)} closes, got {len(closes)}")
    bars = tuple(
        replace(bar, close=Decimal(repr(close)))
        for bar, close in zip(table.bars, closes, strict=True)
    )
    return BarTable(symbol=table.symbol, bars=bars)


def frozen_spy_table() -> BarTable:
    """The committed SPY dataset through its canonical reader (``rl_dataset_from_json`` ->
    ``BarTable.from_dataset``, which re-checks every stored ``known_at``)."""
    return BarTable.from_dataset(rl_dataset_from_json(FROZEN_SPY.read_bytes()))


def make_env(
    closes: Sequence[float] | BarTable, *, audit: bool = False, **config: Any
) -> "SingleTickerTradingEnv":
    """A ``SingleTickerTradingEnv`` over ``closes`` (or a ready ``BarTable``) with ``EnvConfig``
    built from ``config``."""
    from trading.rl.env import EnvConfig, SingleTickerTradingEnv

    table = closes if isinstance(closes, BarTable) else bar_table(closes)
    return SingleTickerTradingEnv(table, EnvConfig(**config), audit=audit)
