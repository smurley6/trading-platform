"""The max-drawdown breaker, exercised THROUGH ``EvaluateSignals`` (review F3).

The existing ``test_risk.py`` drawdown test hands ``RiskGate.evaluate`` a ``RiskContext`` with the
peak set directly, which HIDES the real defect: in the use case ``_peak_equity`` started at 0 and
was set to ``max(0, current)`` right before the check, so peak == current and drawdown == 0 every
run — the breaker was structurally dead. These tests drive the real ``execute`` path, seeding the
peak the way production now does: from the durable log via ``rehydrate_safety_state``, never by
hand-setting ``_peak_equity``.
"""

from decimal import Decimal

from tests.conftest import FixedClock, SignalFactory
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.place_order import PlaceOrder
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.application.safety_rehydration import rehydrate_safety_state
from trading.domain.decisions import RiskVerdict
from trading.domain.events import DomainEvent, EventType
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.strategies.trivial import TrivialStrategy

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("10"),
    max_position_concentration_pct=Decimal("25"),
    max_drawdown_pct=Decimal("20"),
)
_SLEEVE = Sleeve(
    id="sleeve-test",
    strategy_id="trivial",
    allocated_capital=Decimal("10000"),
    mode=TradingMode.PAPER,
    status=SleeveStatus.RUNNING,
)


def _snapshot(clock: FixedClock, equity: str) -> DomainEvent:
    return DomainEvent(
        stream_id="sleeve-test",
        event_type=EventType.PORTFOLIO_SNAPSHOT,
        payload={"sleeve_id": "sleeve-test", "cash": equity, "equity": equity, "positions": {}},
        occurred_at=clock.now(),
        correlation_id="seed",
    )


def _evaluator(
    *,
    store: InMemoryEventStore,
    broker: PaperBroker,
    market_data: StaticMarketData,
    clock: FixedClock,
    initial_peak_equity: Decimal,
) -> EvaluateSignals:
    return EvaluateSignals(
        strategy=TrivialStrategy(position_size=Decimal("5")),
        sleeve=_SLEEVE,
        risk_gate=RiskGate(_LIMITS, KillSwitch()),
        place_order=PlaceOrder(broker=broker, event_store=store, clock=clock),
        broker=broker,
        market_data=market_data,
        event_store=store,
        clock=clock,
        initial_peak_equity=initial_peak_equity,
    )


async def _risk_verdicts(store: InMemoryEventStore) -> list[tuple[str, str]]:
    return [
        (str(s.event.payload["verdict"]), str(s.event.payload["reason"]))
        for s in await store.read_all()
        if s.event.event_type == EventType.RISK_EVALUATED
    ]


async def test_rehydrated_peak_trips_the_drawdown_breaker(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # Equity has fallen to 7000 from a logged peak of 10000 => 30% drawdown >= the 20% cap.
    store = InMemoryEventStore(clock)
    await store.append(_snapshot(clock, "10000"))
    await store.append(_snapshot(clock, "8000"))
    market_data = StaticMarketData({"AAPL": Decimal("100")}, clock)
    broker = PaperBroker(market_data=market_data, clock=clock, starting_cash=Decimal("7000"))

    # Peak comes from the log, exactly as the composition root seeds it — NOT hand-set.
    peak = (await rehydrate_safety_state(store)).peak_equity
    assert peak == Decimal("10000")

    evaluator = _evaluator(
        store=store, broker=broker, market_data=market_data, clock=clock, initial_peak_equity=peak
    )
    result = await evaluator.execute([make_signal(ticker="AAPL", sentiment=0.8)])

    # The order is blocked and the rejection is the drawdown breaker firing.
    assert result.orders == ()
    verdicts = await _risk_verdicts(store)
    assert verdicts, "the decision reached the risk gate"
    assert all(v == RiskVerdict.REJECTED.value for v, _ in verdicts)
    assert any("drawdown" in reason for _, reason in verdicts)


async def test_without_rehydration_the_breaker_stays_dead(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # The pre-fix behaviour: peak seeded to 0 (a fresh process). Same 30%-down account, but with
    # peak == current the drawdown is computed as 0 and the order sails through — the bug F3 fixes.
    store = InMemoryEventStore(clock)
    await store.append(_snapshot(clock, "10000"))
    market_data = StaticMarketData({"AAPL": Decimal("100")}, clock)
    broker = PaperBroker(market_data=market_data, clock=clock, starting_cash=Decimal("7000"))

    evaluator = _evaluator(
        store=store,
        broker=broker,
        market_data=market_data,
        clock=clock,
        initial_peak_equity=Decimal("0"),
    )
    result = await evaluator.execute([make_signal(ticker="AAPL", sentiment=0.8)])

    assert len(result.orders) == 1  # no drawdown rejection when the peak is not rehydrated
    verdicts = await _risk_verdicts(store)
    assert verdicts and all("drawdown" not in reason for _, reason in verdicts)
