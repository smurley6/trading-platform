"""Backtest use case (FR-BT).

Replays recorded signals through the **same** strategy, risk gate, and execution path used
live — `EvaluateSignals` against an isolated, synthetic environment (FR-BT-1). There is no
parallel backtest logic: the only difference from production is the wiring (a manual clock,
historical prices, a throwaway broker/store), supplied via the `BacktestEnvFactory` port.
Output is an equity curve, return, drawdown, and trade list (FR-BT-2).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import groupby
from uuid import uuid4

from trading.application.evaluate_signals import EvaluateSignals
from trading.application.place_order import PlaceOrder
from trading.application.ports.backtest_env import BacktestEnv, BacktestEnvFactory
from trading.application.projections import signals_from_events
from trading.application.risk import KillSwitch, RiskGate, RiskLimits
from trading.application.signal_recording import record_signals
from trading.domain.events import EventType
from trading.domain.metrics import EquityPoint, max_drawdown_pct, total_return_pct
from trading.domain.prices import PriceSeries
from trading.domain.signals import Signal
from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode
from trading.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    side: str
    price: Decimal
    quantity: Decimal
    filled_at: datetime


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    strategy: Strategy
    signals: Sequence[Signal]
    risk_limits: RiskLimits
    prices: PriceSeries
    starting_cash: Decimal
    window_days: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    starting_equity: Decimal
    ending_equity: Decimal
    total_return_pct: float
    max_drawdown_pct: float
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[Trade, ...]
    decision_count: int


class Backtest:
    def __init__(self, *, env_factory: BacktestEnvFactory) -> None:
        self._env_factory = env_factory

    async def run(self, request: BacktestRequest) -> BacktestResult:
        env = self._env_factory.create(prices=request.prices, starting_cash=request.starting_cash)
        sleeve = Sleeve(
            id="backtest",
            strategy_id=request.strategy.strategy_id,
            allocated_capital=request.starting_cash,
            mode=TradingMode.PAPER,
            status=SleeveStatus.RUNNING,
        )
        evaluate = EvaluateSignals(
            strategy=request.strategy,
            sleeve=sleeve,
            risk_gate=RiskGate(request.risk_limits, KillSwitch()),
            place_order=PlaceOrder(broker=env.broker, event_store=env.event_store, clock=env.clock),
            broker=env.broker,
            market_data=env.market_data,
            event_store=env.event_store,
            clock=env.clock,
        )

        # Replay chronologically: at each timestamp, advance the clock, record that batch,
        # then evaluate the whole corroboration window — exactly the live ingest->cycle flow.
        ordered = sorted(request.signals, key=lambda s: s.created_at)
        decision_count = 0
        for when, group in groupby(ordered, key=lambda s: s.created_at):
            batch = list(group)
            env.clock.set(when)
            await record_signals(
                event_store=env.event_store,
                clock=env.clock,
                signals=batch,
                correlation_id=str(uuid4()),
            )
            window = signals_from_events(
                await env.event_store.read_all(), now=when, window_days=request.window_days
            )
            result = await evaluate.execute(window)
            decision_count += len(result.decisions)

        return await self._summarize(env, request.starting_cash, decision_count)

    async def _summarize(
        self, env: BacktestEnv, starting_cash: Decimal, decision_count: int
    ) -> BacktestResult:
        events = await env.event_store.read_all()
        curve = tuple(
            EquityPoint(s.event.occurred_at, Decimal(str(s.event.payload["equity"])))
            for s in events
            if s.event.event_type == EventType.PORTFOLIO_SNAPSHOT
        )
        trades = tuple(
            Trade(
                symbol=str(s.event.payload["symbol"]),
                side=str(s.event.payload["side"]),
                price=Decimal(str(s.event.payload["price"])),
                quantity=Decimal(str(s.event.payload["quantity"])),
                filled_at=datetime.fromisoformat(str(s.event.payload["filled_at"])),
            )
            for s in events
            if s.event.event_type == EventType.ORDER_FILLED
        )
        ending = curve[-1].equity if curve else starting_cash
        values = [p.equity for p in curve] or [starting_cash]
        return BacktestResult(
            starting_equity=starting_cash,
            ending_equity=ending,
            total_return_pct=total_return_pct(starting_cash, ending),
            max_drawdown_pct=max_drawdown_pct(values),
            equity_curve=curve,
            trades=trades,
            decision_count=decision_count,
        )
