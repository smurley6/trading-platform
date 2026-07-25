"""One broker rejection must not abort the evaluate loop and skip risk-reducing SELLs (F17).

Scenario: a fully-invested pool's BUY reaches the broker and is rejected (Alpaca 403). Before this
fix the raised ``OrderSubmissionError`` propagated out of ``EvaluateSignals`` and aborted every
remaining decision — including the risk-reducing SELLs and the closing portfolio snapshot. Each
placement is now isolated: the failed BUY is recorded (ORDER_SUBMIT_FAILED) and the loop carries on
to place the SELL and write the snapshot.
"""

from collections.abc import Sequence
from decimal import Decimal

from tests.conftest import FixedClock
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.place_order import PlaceOrder
from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderNotFoundError,
)
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.domain.decisions import TargetPosition
from trading.domain.events import EventType
from trading.domain.orders import OrderSide, OrderStatus, OrderType
from trading.domain.positions import Portfolio
from trading.domain.signals import Signal
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.strategies.base import StrategyProposal

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("90"),
    max_position_concentration_pct=Decimal("90"),
    max_drawdown_pct=Decimal("90"),
)
_SLEEVE = Sleeve(
    id="sleeve-political",
    strategy_id="two",
    allocated_capital=Decimal("100000"),
    mode=TradingMode.PAPER,
    status=SleeveStatus.RUNNING,
)


class _BrokerRejectError(Exception):
    """Stand-in for an Alpaca 403 insufficient-buying-power reject."""


class _TwoProposalStrategy:
    """Emits a BUY (AAPL) first, then a risk-reducing SELL (MSFT), ignoring signals."""

    strategy_id = "two"

    def evaluate(
        self, signals: Sequence[Signal], portfolio: Portfolio
    ) -> Sequence[StrategyProposal]:
        return [
            StrategyProposal(target=TargetPosition(symbol="AAPL", quantity=Decimal("5"))),
            StrategyProposal(target=TargetPosition(symbol="MSFT", quantity=Decimal("0"))),
        ]


class _RejectBuysBroker:
    """Holds MSFT 10; rejects every BUY submission, accepts SELLs."""

    def __init__(self) -> None:
        self.buy_attempts = 0
        self.sell_orders: list[str] = []

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        if side is OrderSide.BUY:
            self.buy_attempts += 1
            raise _BrokerRejectError("403 insufficient buying power")
        self.sell_orders.append(client_order_id)
        return BrokerOrder(
            client_order_id=client_order_id,
            broker_order_id="b-sell",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=OrderStatus.SUBMITTED,
        )

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        raise OrderNotFoundError(client_order_id)

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [BrokerPosition("MSFT", Decimal("10"), Decimal("100"))]

    async def get_account(self) -> BrokerAccount:
        # buying_power == 0: a real exhausted reserve; the BUY that slips past reaches the broker
        # (via a non-guarded path here) and is rejected — the loop must survive it.
        return BrokerAccount(
            cash=Decimal("0"), equity=Decimal("100000"), buying_power=Decimal("100000")
        )


async def test_broker_reject_does_not_abort_the_loop_and_the_sell_still_places(
    clock: FixedClock,
) -> None:
    broker = _RejectBuysBroker()
    store = InMemoryEventStore(clock)
    evaluator = EvaluateSignals(
        strategy=_TwoProposalStrategy(),
        sleeve=_SLEEVE,
        risk_gate=RiskGate(_LIMITS, KillSwitch()),
        place_order=PlaceOrder(broker=broker, event_store=store, clock=clock),
        broker=broker,
        market_data=StaticMarketData({"AAPL": Decimal("100"), "MSFT": Decimal("100")}, clock),
        event_store=store,
        clock=clock,
    )

    result = await evaluator.execute([])

    # The BUY was attempted and rejected, but the SELL that follows it STILL placed.
    assert broker.buy_attempts == 1
    assert len(broker.sell_orders) == 1
    assert [o.side for o in result.orders] == [OrderSide.SELL]

    types = [s.event.event_type for s in await store.read_all()]
    # The failed BUY is on the audit trail...
    assert EventType.ORDER_SUBMIT_FAILED in types
    # ...and the loop ran to completion (closing snapshot written).
    assert EventType.PORTFOLIO_SNAPSHOT in types
