"""Concrete backtest environment factory: assembles the throwaway in-memory stack."""

from decimal import Decimal

from trading.application.ports.backtest_env import BacktestEnv
from trading.domain.prices import PriceSeries
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.clock import ManualClock
from trading.infrastructure.market_data.historical import HistoricalMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore


class InMemoryBacktestEnvFactory:
    def create(self, *, prices: PriceSeries, starting_cash: Decimal) -> BacktestEnv:
        clock = ManualClock()
        event_store = InMemoryEventStore(clock)
        market_data = HistoricalMarketData(prices, clock)
        broker = PaperBroker(market_data=market_data, clock=clock, starting_cash=starting_cash)
        return BacktestEnv(
            clock=clock,
            event_store=event_store,
            broker=broker,
            market_data=market_data,
        )
