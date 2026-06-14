"""Backtest environment factory port.

A backtest needs a throwaway, isolated stack — manual clock, in-memory event store,
historical-price market data, paper broker — that must never touch live state. The
factory keeps the domain pure: the `Backtest` use case asks for an environment through
this port, and the composition root supplies the concrete adapters.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading.application.ports.broker import BrokerPort
from trading.application.ports.clock import MutableClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.market_data import MarketDataPort
from trading.domain.prices import PriceSeries


@dataclass(frozen=True, slots=True)
class BacktestEnv:
    clock: MutableClockPort
    event_store: EventStorePort
    broker: BrokerPort
    market_data: MarketDataPort


class BacktestEnvFactory(Protocol):
    def create(self, *, prices: PriceSeries, starting_cash: Decimal) -> BacktestEnv:
        """Build a fresh, isolated backtest environment."""
        ...
