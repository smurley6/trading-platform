"""End-to-end political pipeline on real wiring (fake data source): ingest -> conviction
window -> strategy -> risk -> paper order -> fill -> event log. Spec sections 4.3, 14."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.ingest_signals import IngestSignals
from trading.application.place_order import PlaceOrder
from trading.application.political_cycle import PoliticalCycle
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.domain.conviction import ConvictionThresholds
from trading.domain.events import EventType
from trading.domain.orders import OrderStatus
from trading.domain.signals import Signal, SignalSource
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.strategies.political_signal import PoliticalSignalStrategy


class FakeSource:
    def __init__(self, name: str, signals: list[Signal]) -> None:
        self._name = name
        self._signals = signals

    @property
    def name(self) -> str:
        return self._name

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        return self._signals


@dataclass
class CycleEnv:
    make_cycle: Callable[[list[Signal]], PoliticalCycle]
    broker: PaperBroker
    event_store: InMemoryEventStore


@pytest.fixture
def cycle_env(clock: FixedClock) -> CycleEnv:
    event_store = InMemoryEventStore(clock)
    market_data = StaticMarketData({"LMT": Decimal("450")}, clock)
    broker = PaperBroker(market_data=market_data, clock=clock, starting_cash=Decimal("100000"))
    sleeve = Sleeve(
        id="sleeve-political",
        strategy_id="political_signal",
        allocated_capital=Decimal("100000"),
        mode=TradingMode.PAPER,
        status=SleeveStatus.RUNNING,
    )
    risk_gate = RiskGate(
        RiskLimits(
            max_trade_pct_of_sleeve=Decimal("10"),
            max_position_concentration_pct=Decimal("25"),
            max_drawdown_pct=Decimal("20"),
        ),
        KillSwitch(),
    )
    evaluator = EvaluateSignals(
        strategy=PoliticalSignalStrategy(
            thresholds=ConvictionThresholds(trade=70.0, watch=40.0),
            window_days=21,
            position_size=Decimal("2"),
        ),
        sleeve=sleeve,
        risk_gate=risk_gate,
        place_order=PlaceOrder(broker=broker, event_store=event_store, clock=clock),
        broker=broker,
        market_data=market_data,
        event_store=event_store,
        clock=clock,
    )

    def make_cycle(source_signals: list[Signal]) -> PoliticalCycle:
        return PoliticalCycle(
            ingest=IngestSignals(
                sources=[FakeSource("news", source_signals)],
                event_store=event_store,
                clock=clock,
                lookback_start=clock.now() - timedelta(days=21),
            ),
            evaluate=evaluator,
            event_store=event_store,
            clock=clock,
            window_days=21,
        )

    return CycleEnv(make_cycle=make_cycle, broker=broker, event_store=event_store)


async def test_corroborated_high_conviction_trades(
    cycle_env: CycleEnv,
    make_signal: SignalFactory,
) -> None:
    make_cycle, broker, event_store = cycle_env.make_cycle, cycle_env.broker, cycle_env.event_store
    # Two source types aligned and strongly positive on LMT -> conviction over threshold.
    signals = [
        make_signal(
            ticker="LMT",
            sentiment=0.9,
            politician_weight=1.5,
            source_type=SignalSource.STATEMENT,
        ),
        make_signal(
            ticker="LMT",
            sentiment=0.8,
            politician_weight=1.5,
            source_type=SignalSource.CONTRACT,
        ),
    ]
    result = await make_cycle(signals).execute()

    assert len(result.evaluation.orders) == 1
    assert result.evaluation.orders[0].status is OrderStatus.FILLED
    positions = {p.symbol: p for p in await broker.list_positions()}
    assert positions["LMT"].quantity == Decimal("2")

    # The decision records the conviction and the signals behind it (spec section 12).
    decision_events = [
        s for s in await event_store.read_all() if s.event.event_type == EventType.DECISION_MADE
    ]
    assert len(decision_events) == 1
    payload = decision_events[0].event.payload
    assert payload["conviction_score"] >= 70
    assert len(payload["contributing_signal_ids"]) == 2


async def test_below_threshold_signal_is_recorded_but_not_traded(
    cycle_env: CycleEnv,
    make_signal: SignalFactory,
) -> None:
    make_cycle, broker, event_store = cycle_env.make_cycle, cycle_env.broker, cycle_env.event_store
    result = await make_cycle(
        [make_signal(ticker="LMT", sentiment=0.1, politician_weight=1.0)]
    ).execute()

    assert result.evaluation.orders == ()
    assert await broker.list_positions() == []
    # ...but the signal is still captured (spec section 9: capture every signal).
    signal_events = [
        s for s in await event_store.read_all() if s.event.event_type == EventType.SIGNAL_RECEIVED
    ]
    assert len(signal_events) == 1
