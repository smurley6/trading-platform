"""Shared fixtures. All time flows through FixedClock; no real I/O in unit/e2e tests."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from trading.application.evaluate_signals import EvaluateSignals
from trading.application.place_order import PlaceOrder
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.application.sleeve_registry import SleeveRegistry, SleeveState
from trading.domain.signals import Signal, SignalSource
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.strategies.trivial import TrivialStrategy

SignalFactory = Callable[..., Signal]


class FixedClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 6, 12, 14, 30, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock()


@pytest.fixture
def market_data(clock: FixedClock) -> StaticMarketData:
    return StaticMarketData({"AAPL": Decimal("100"), "MSFT": Decimal("200")}, clock)


@pytest.fixture
def read_model() -> InMemoryReadModel:
    return InMemoryReadModel()


@pytest.fixture
def event_store(clock: FixedClock, read_model: InMemoryReadModel) -> ProjectingEventStore:
    # Projecting store keeps the read model current on append (CQRS), exactly as production
    # wiring does, so tests exercise the same read-after-write path.
    return ProjectingEventStore(InMemoryEventStore(clock), read_model)


@pytest.fixture
def broker(market_data: StaticMarketData, clock: FixedClock) -> PaperBroker:
    return PaperBroker(market_data=market_data, clock=clock, starting_cash=Decimal("10000"))


@pytest.fixture
def kill_switch() -> KillSwitch:
    return KillSwitch()


@pytest.fixture
def risk_limits() -> RiskLimits:
    return RiskLimits(
        max_trade_pct_of_sleeve=Decimal("10"),
        max_position_concentration_pct=Decimal("25"),
        max_drawdown_pct=Decimal("20"),
    )


@pytest.fixture
def risk_gate(risk_limits: RiskLimits, kill_switch: KillSwitch) -> RiskGate:
    return RiskGate(risk_limits, kill_switch)


@pytest.fixture
def sleeve() -> Sleeve:
    return Sleeve(
        id="sleeve-test",
        strategy_id="trivial",
        allocated_capital=Decimal("10000"),
        mode=TradingMode.PAPER,
        status=SleeveStatus.RUNNING,
    )


@pytest.fixture
def sleeve_registry(sleeve: Sleeve) -> SleeveRegistry:
    return SleeveRegistry([SleeveState(sleeve=sleeve, mode=sleeve.mode, status=sleeve.status)])


@pytest.fixture
def place_order(
    broker: PaperBroker, event_store: ProjectingEventStore, clock: FixedClock
) -> PlaceOrder:
    return PlaceOrder(broker=broker, event_store=event_store, clock=clock)


@pytest.fixture
def reconcile(
    broker: PaperBroker, event_store: ProjectingEventStore, clock: FixedClock
) -> Reconcile:
    return Reconcile(broker=broker, event_store=event_store, clock=clock)


@pytest.fixture
def evaluate_signals(
    sleeve: Sleeve,
    risk_gate: RiskGate,
    place_order: PlaceOrder,
    broker: PaperBroker,
    market_data: StaticMarketData,
    event_store: ProjectingEventStore,
    clock: FixedClock,
) -> EvaluateSignals:
    return EvaluateSignals(
        strategy=TrivialStrategy(position_size=Decimal("5")),
        sleeve=sleeve,
        risk_gate=risk_gate,
        place_order=place_order,
        broker=broker,
        market_data=market_data,
        event_store=event_store,
        clock=clock,
    )


@pytest.fixture
def make_signal(clock: FixedClock) -> SignalFactory:
    def _make(
        ticker: str = "AAPL",
        sentiment: float = 0.8,
        source_type: SignalSource = SignalSource.STATEMENT,
        politician_weight: float = 1.0,
        source_reliability: float = 1.0,
    ) -> Signal:
        return Signal(
            id=str(uuid4()),
            created_at=clock.now(),
            source_type=source_type,
            ticker=ticker,
            politician_id="pol-1",
            sentiment=sentiment,
            politician_weight=politician_weight,
            source_reliability=source_reliability,
            raw_payload={"source": "test-fixture"},
        )

    return _make
