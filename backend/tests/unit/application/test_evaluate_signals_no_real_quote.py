"""Under require_real_data, a symbol with no real price is SKIPPED, not sized synthetically (F4).

At 13:00 UTC pre-open the IEX book is often empty, so the live feed returns no quote / last=0
for some symbols. The market-data seam is wired with no static fallback in that mode, so it
raises ``NoRealQuoteError`` for those symbols. ``EvaluateSignals`` must skip them BEFORE any
decision/market_context event is recorded (so the audit trail never carries a synthetic price)
while symbols that DO have a real quote proceed normally.
"""

from decimal import Decimal

from tests.conftest import FixedClock, SignalFactory
from trading.application.evaluate_signals import EvaluateSignals
from trading.application.place_order import PlaceOrder
from trading.application.ports.market_data import NoRealQuoteError, Quote
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.domain.events import EventType
from trading.domain.provenance import Provenance
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.market_data.static import StaticMarketData
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.strategies.trivial import TrivialStrategy

_LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("50"),
    max_position_concentration_pct=Decimal("90"),
    max_drawdown_pct=Decimal("90"),
)
_SLEEVE = Sleeve(
    id="sleeve-test",
    strategy_id="trivial",
    allocated_capital=Decimal("100000"),
    mode=TradingMode.PAPER,
    status=SleeveStatus.RUNNING,
)


class _SelectiveMarketData:
    """Real-mode seam stub: a real quote for priced symbols, NoRealQuoteError for the rest."""

    def __init__(self, priced: dict[str, Decimal], clock: FixedClock) -> None:
        self._priced = priced
        self._clock = clock

    async def get_quote(self, symbol: str) -> Quote:
        if symbol not in self._priced:
            raise NoRealQuoteError(symbol)
        price = self._priced[symbol]
        return Quote(symbol=symbol, bid=price, ask=price, last=price, as_of=self._clock.now())


async def test_skips_symbol_with_no_real_quote_and_trades_the_priced_one(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    store = InMemoryEventStore(clock)
    # Broker fills use a static book that prices BOTH tickers; the decision seam is the stub that
    # only prices GOOD, so DEAD reaches the seam and is skipped (the two seams are independent).
    broker = PaperBroker(
        market_data=StaticMarketData({"GOOD": Decimal("300"), "DEAD": Decimal("900")}, clock),
        clock=clock,
        starting_cash=Decimal("100000"),
    )
    seam = _SelectiveMarketData({"GOOD": Decimal("300")}, clock)
    evaluator = EvaluateSignals(
        strategy=TrivialStrategy(position_size=Decimal("5")),
        sleeve=_SLEEVE,
        risk_gate=RiskGate(_LIMITS, KillSwitch()),
        place_order=PlaceOrder(broker=broker, event_store=store, clock=clock),
        broker=broker,
        market_data=seam,
        event_store=store,
        clock=clock,
        require_real_data=True,
    )

    result = await evaluator.execute(
        [
            make_signal(ticker="GOOD", sentiment=0.8, provenance=Provenance.REAL),
            make_signal(ticker="DEAD", sentiment=0.8, provenance=Provenance.REAL),
        ]
    )

    # Exactly one order, for the symbol with a real quote; the unpriced symbol placed none.
    assert [o.symbol for o in result.orders] == ["GOOD"]

    decisions = [
        s.event for s in await store.read_all() if s.event.event_type == EventType.DECISION_MADE
    ]
    symbols = {d.payload["symbol"] for d in decisions}
    # No decision (hence no market_context, no synthetic price) was ever recorded for DEAD.
    assert symbols == {"GOOD"}
    good_decision = next(d for d in decisions if d.payload["symbol"] == "GOOD")
    assert good_decision.payload["market_context"]["last"] == "300"

    # DEAD never reached the risk gate either — it was skipped before any decision was made.
    risk_events = [
        s.event for s in await store.read_all() if s.event.event_type == EventType.RISK_EVALUATED
    ]
    assert all(
        e.payload["decision_id"] == good_decision.payload["decision_id"] for e in risk_events
    )
