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
from trading.domain.metrics import (
    DEFAULT_ANNUALIZATION,
    EquityPoint,
    TradeStats,
    benchmark_total_return_pct,
    buy_and_hold_curve,
    cagr_pct,
    max_drawdown_pct,
    period_returns,
    return_volatility,
    round_trips,
    sharpe_ratio,
    sortino_ratio,
    total_return_pct,
    trade_stats,
)
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
    """Backtest outcome: the existing equity/return/drawdown/trade summary (FR-BT-2) plus
    risk-adjusted ratios, trade statistics, and a passive buy-and-hold benchmark.

    Every metric is computed by a pure function in ``domain.metrics`` from the equity curve
    and realised fills the harness already records (FR-BT-1); see ``Backtest._summarize``.
    The annualization assumption (one equity point per trading day -> 252) and the FIFO P&L
    pairing rule are documented on those functions.
    """

    starting_equity: Decimal
    ending_equity: Decimal
    total_return_pct: float
    max_drawdown_pct: float
    equity_curve: tuple[EquityPoint, ...]
    trades: tuple[Trade, ...]
    decision_count: int
    # Risk-adjusted return (annualized, risk-free = 0; 0.0 when undefined). FR-BT-2.
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    # Annualized return / CAGR over the curve's wall-clock span, as a percentage. FR-BT-2.
    annualized_return_pct: float = 0.0
    # Annualized stdev of the equity curve's period returns. FR-BT-2.
    return_volatility: float = 0.0
    # Realised round-trip trade statistics (win rate, profit factor, ...). FR-BT-2.
    trade_stats: TradeStats | None = None
    # Passive buy-and-hold benchmark valued on the strategy's timeline (FR-BT-2): the curve
    # and its total return vs. the same starting cash, for strategy-vs-passive comparison.
    benchmark_curve: tuple[EquityPoint, ...] = ()
    benchmark_total_return_pct: float = 0.0


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

        return await self._summarize(env, request, decision_count)

    async def _summarize(
        self, env: BacktestEnv, request: BacktestRequest, decision_count: int
    ) -> BacktestResult:
        starting_cash = request.starting_cash
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

        # All risk/return/trade metrics are pure functions of the curve and fills (FR-BT-2).
        returns = period_returns(values)
        stats = trade_stats(round_trips(trades))

        # Passive benchmark over the same timeline: hold the full priced universe equally.
        timeline = [p.when for p in curve]
        bench_curve = buy_and_hold_curve(
            request.prices.symbols(), request.prices, timeline, starting_cash
        )

        return BacktestResult(
            starting_equity=starting_cash,
            ending_equity=ending,
            total_return_pct=total_return_pct(starting_cash, ending),
            max_drawdown_pct=max_drawdown_pct(values),
            equity_curve=curve,
            trades=trades,
            decision_count=decision_count,
            sharpe_ratio=sharpe_ratio(returns, annualization=DEFAULT_ANNUALIZATION),
            sortino_ratio=sortino_ratio(returns, annualization=DEFAULT_ANNUALIZATION),
            annualized_return_pct=cagr_pct(curve),
            return_volatility=return_volatility(returns, annualization=DEFAULT_ANNUALIZATION),
            trade_stats=stats,
            benchmark_curve=bench_curve,
            benchmark_total_return_pct=benchmark_total_return_pct(bench_curve, starting_cash),
        )
