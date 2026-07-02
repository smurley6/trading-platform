"""End-to-end on the real wiring (in-memory adapters): signal -> decision -> risk ->
paper order -> fill -> event log -> reconciliation. Spec section 14."""

from decimal import Decimal

from tests.conftest import FixedClock, SignalFactory
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.reconcile import Reconcile
from trading.application.risk import KillSwitch
from trading.application.signal_recording import record_signals
from trading.domain.events import EventType
from trading.domain.orders import OrderStatus
from trading.domain.provenance import Provenance
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.persistence.memory import InMemoryEventStore


async def test_positive_signal_results_in_filled_paper_order(
    evaluate_signals: EvaluateSignals,
    broker: PaperBroker,
    event_store: InMemoryEventStore,
    reconcile: Reconcile,
    make_signal: SignalFactory,
    clock: FixedClock,
) -> None:
    # Signals are recorded on observation, then evaluated under the same correlation id —
    # exactly what the manual-push endpoint does (spec section 9).
    signals = [make_signal(ticker="AAPL", sentiment=0.9)]
    correlation_id = "corr-e2e"
    await record_signals(
        event_store=event_store, clock=clock, signals=signals, correlation_id=correlation_id
    )
    result = await evaluate_signals.execute(signals, correlation_id=correlation_id)

    assert len(result.orders) == 1
    assert result.orders[0].status is OrderStatus.FILLED
    positions = {p.symbol: p for p in await broker.list_positions()}
    assert positions["AAPL"].quantity == Decimal("5")

    # The full audit chain shares one correlation id (spec section 12).
    chain = {
        s.event.event_type for s in await event_store.read_by_correlation(result.correlation_id)
    }
    assert {
        EventType.SIGNAL_RECEIVED,
        EventType.DECISION_MADE,
        EventType.RISK_EVALUATED,
        EventType.ORDER_INTENT,
        EventType.ORDER_SUBMITTED,
        EventType.ORDER_FILLED,
        EventType.PORTFOLIO_SNAPSHOT,
    } <= chain

    # Raw provider payload is retained on the signal event (spec section 9).
    signal_events = [
        s
        for s in await event_store.read_by_correlation(result.correlation_id)
        if s.event.event_type == EventType.SIGNAL_RECEIVED
    ]
    assert signal_events[0].event.payload["raw_payload"] == {"source": "test-fixture"}

    report = await reconcile.execute()
    assert report.clean


async def test_negative_signal_exits_position(
    evaluate_signals: EvaluateSignals,
    broker: PaperBroker,
    make_signal: SignalFactory,
) -> None:
    await evaluate_signals.execute([make_signal(ticker="AAPL", sentiment=0.9)])
    await evaluate_signals.execute([make_signal(ticker="AAPL", sentiment=-0.9)])

    assert await broker.list_positions() == []


async def test_backfilled_signal_never_trades_but_does_not_abort_the_batch(
    evaluate_signals: EvaluateSignals,
    broker: PaperBroker,
    event_store: InMemoryEventStore,
    make_signal: SignalFactory,
) -> None:
    # Provenance wall (RESEARCH-PLAN §2): BACKFILLED must never reach order placement, but a
    # BACKFILLED signal in the batch must NOT abort the whole evaluation — a REAL signal on a
    # different ticker in the SAME evaluation still trades normally (filter, not abort).
    backfilled = make_signal(ticker="MSFT", sentiment=0.9, provenance=Provenance.BACKFILLED)
    real = make_signal(ticker="AAPL", sentiment=0.9, provenance=Provenance.REAL)

    result = await evaluate_signals.execute([backfilled, real])

    assert len(result.orders) == 1
    assert result.orders[0].symbol == "AAPL"
    positions = {p.symbol: p for p in await broker.list_positions()}
    assert "MSFT" not in positions
    assert positions["AAPL"].quantity == Decimal("5")

    # No decision/order-intent event was ever recorded for the backfilled signal's ticker.
    events = await event_store.read_by_correlation(result.correlation_id)
    symbols_with_decisions = {
        e.event.payload["symbol"] for e in events if e.event.event_type == EventType.DECISION_MADE
    }
    assert symbols_with_decisions == {"AAPL"}
    order_intent_symbols = {
        e.event.payload["symbol"] for e in events if e.event.event_type == EventType.ORDER_INTENT
    }
    assert order_intent_symbols == {"AAPL"}


async def test_kill_switch_blocks_new_orders(
    evaluate_signals: EvaluateSignals,
    broker: PaperBroker,
    event_store: InMemoryEventStore,
    kill_switch: KillSwitch,
    make_signal: SignalFactory,
) -> None:
    kill_switch.engage()

    result = await evaluate_signals.execute([make_signal(ticker="MSFT", sentiment=0.9)])

    assert result.orders == ()
    assert await broker.list_positions() == []
    verdicts = [
        s.event.payload["verdict"]
        for s in await event_store.read_by_correlation(result.correlation_id)
        if s.event.event_type == EventType.RISK_EVALUATED
    ]
    assert verdicts == ["rejected"]
