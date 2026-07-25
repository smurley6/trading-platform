"""EvaluateSignals use case: signals -> strategy -> risk gate -> execution.

Signals are recorded upstream (see `signal_recording`); this use case reads them in,
runs the strategy, and emits decision/risk/order/snapshot events. Every step shares one
correlation id, so any trade answers "what signals, what conviction, what risk verdict,
what market context" (spec section 12).
"""

import asyncio
import logging
from collections.abc import Collection, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from trading.application.data_integrity import ensure_real_signals, tradeable_signals
from trading.application.place_order import PlaceOrder, PlaceOrderRequest
from trading.application.ports.broker import BrokerOrder, BrokerPort
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.lock import CrossProcessLockPort
from trading.application.ports.market_data import MarketDataPort, NoRealQuoteError, Quote
from trading.application.risk import RiskContext, RiskGate
from trading.domain.decisions import Decision, RiskVerdict
from trading.domain.events import DomainEvent, EventType
from trading.domain.orders import TERMINAL_STATUSES, OrderSide
from trading.domain.positions import Portfolio, Position
from trading.domain.signals import Signal
from trading.domain.sleeves import Sleeve
from trading.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    correlation_id: str
    decisions: tuple[Decision, ...]
    orders: tuple[BrokerOrder, ...]


class EvaluateSignals:
    def __init__(
        self,
        *,
        strategy: Strategy,
        sleeve: Sleeve,
        risk_gate: RiskGate,
        place_order: PlaceOrder,
        broker: BrokerPort,
        market_data: MarketDataPort,
        event_store: EventStorePort,
        clock: ClockPort,
        require_real_data: bool = False,
        initial_peak_equity: Decimal = Decimal("0"),
        cross_process_lock: CrossProcessLockPort | None = None,
    ) -> None:
        self._strategy = strategy
        self._sleeve = sleeve
        self._risk_gate = risk_gate
        self._place_order = place_order
        self._broker = broker
        self._market_data = market_data
        self._event_store = event_store
        self._clock = clock
        # When True (a real broker is in play), refuse to evaluate on any non-REAL signal.
        self._require_real_data = require_real_data
        # Serialize the evaluate->place path (review F6). One asyncio.Lock per sleeve (this instance
        # is the sleeve's single evaluator) closes IN-PROCESS overlap — two execute() coroutines
        # snapshotting the account before either places. The cross-process lock (a Postgres advisory
        # lock in prod, no-op in-memory) closes the CRON-vs-WEB case a per-process lock cannot see.
        self._lock = asyncio.Lock()
        self._cross_process_lock = cross_process_lock
        # Peak-equity high-water mark for the drawdown circuit breaker. Seeded at startup from
        # the durable log (max PORTFOLIO_SNAPSHOT equity; review F3) so a one-shot cron / redeploy
        # compares current equity against the REAL peak instead of a fresh zero — otherwise peak
        # == current every run and the max_drawdown_pct breaker can never trip. Updated in-process
        # below on each cycle.
        self._peak_equity = initial_peak_equity

    async def execute(
        self,
        signals: Sequence[Signal],
        *,
        correlation_id: str | None = None,
        only_symbols: Collection[str] | None = None,
        place: bool = True,
    ) -> EvaluationResult:
        correlation_id = correlation_id or str(uuid4())
        # Serialize the whole snapshot->size->place critical section (review F6). The per-sleeve
        # asyncio.Lock stops in-process overlap; the cross-process advisory lock stops the
        # cron-vs-web race. Both are context managers, so a mid-loop failure releases them on ALL
        # paths. Paired with the DETERMINISTIC client_order_id below, so even a lock that is somehow
        # bypassed falls back to broker-side idempotency instead of double exposure (review F5).
        guard = (
            self._cross_process_lock.guard(f"evaluate-{self._sleeve.id}")
            if self._cross_process_lock is not None
            else nullcontext()
        )
        async with self._lock, guard:
            return await self._evaluate(
                signals,
                correlation_id=correlation_id,
                only_symbols=only_symbols,
                place=place,
            )

    async def _evaluate(
        self,
        signals: Sequence[Signal],
        *,
        correlation_id: str,
        only_symbols: Collection[str] | None,
        place: bool,
    ) -> EvaluationResult:
        # Anti-placeholder guard (fail-closed): with a real broker, a single SYNTHETIC signal in
        # the window aborts the whole evaluation — no real order is ever driven by fake data.
        if self._require_real_data:
            ensure_real_signals(signals)

        # Provenance wall (RESEARCH-PLAN §2): BACKFILLED signals are real but must never reach
        # the strategy/order-placement path. Unlike the SYNTHETIC guard above, this FILTERS
        # rather than aborts, so REAL (or SYNTHETIC, outside require_real_data) signals in the
        # same batch still evaluate normally instead of the whole cycle halting.
        signals = tradeable_signals(signals)

        account = await self._broker.get_account()
        portfolio = Portfolio(
            cash=account.cash,
            positions={
                p.symbol: Position(p.symbol, p.quantity, p.avg_entry_price)
                for p in await self._broker.list_positions()
            },
        )
        # Open-order awareness (review F5): a DAY order placed pre-open queues UNFILLED until 09:30
        # ET, so it is invisible to list_positions() during the window the cron runs in. Count each
        # open order's not-yet-filled quantity as held (BUY +, SELL -) so a queued order is not
        # re-proposed on a second evaluation (a crash-retry, POST /ingest, the scheduler). Paper
        # fills synchronously, so its orders are always terminal and `pending` is empty — the paper
        # path is unchanged and deterministic.
        pending = _pending_by_symbol(await self._broker.list_orders())
        # Deterministic per-run trade date: the idempotency key below is stable across retries and
        # overlapping runs on the same calendar day (review F5).
        cycle_date = self._clock.now().date().isoformat()
        self._peak_equity = max(self._peak_equity, account.equity)
        # Shared-pool buying-power reserve for this cycle. `portfolio`/`account` come from the ONE
        # shared broker book, so both algos size against the same live pool; this running reserve
        # stops multiple BUYs in a single evaluation from double-spending it. (Cross-evaluator
        # racing in the same cycle still relies on the account itself as the backstop.)
        remaining_buying_power = account.buying_power

        decisions: list[Decision] = []
        orders: list[BrokerOrder] = []
        for proposal in self._strategy.evaluate(signals, portfolio):
            target = proposal.target
            # Execute-proposal path: restrict to the explicitly chosen symbols (None = all).
            if only_symbols is not None and target.symbol not in only_symbols:
                continue
            # `current` folds in pending open-order quantity (review F5) so a queued-but-unfilled
            # order counts as held: the delta below — and the position value the risk gate sizes
            # against — reflect the exposure already in flight, not just settled positions.
            current = portfolio.quantity(target.symbol) + pending.get(target.symbol, Decimal("0"))
            delta = target.quantity - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            quantity = abs(delta)

            try:
                quote = await self._market_data.get_quote(target.symbol)
            except NoRealQuoteError:
                # Fail closed (review F4): with a real account, no live price means NO trade for
                # this symbol — never a guessed one. Skip before any decision/market_context event
                # is recorded, so the audit trail never carries a synthetic price. Other symbols
                # with valid quotes still evaluate.
                logger.warning(
                    "skipping %s: no real market price available; no order placed",
                    target.symbol,
                    extra={"correlation_id": correlation_id},
                )
                continue
            decision = Decision(
                id=str(uuid4()),
                strategy_id=self._strategy.strategy_id,
                sleeve_id=self._sleeve.id,
                created_at=self._clock.now(),
                symbol=target.symbol,
                target_quantity=target.quantity,
                correlation_id=correlation_id,
                conviction_score=proposal.conviction_score,
                contributing_signal_ids=proposal.contributing_signal_ids,
            )
            decisions.append(decision)
            decision_stream = f"decision-{decision.id}"
            await self._append(
                decision_stream,
                EventType.DECISION_MADE,
                {
                    "decision_id": decision.id,
                    "strategy_id": decision.strategy_id,
                    "sleeve_id": decision.sleeve_id,
                    "symbol": decision.symbol,
                    "target_quantity": str(decision.target_quantity),
                    "current_quantity": str(current),
                    "side": side.value,
                    "quantity": str(quantity),
                    "conviction_score": proposal.conviction_score,
                    "contributing_signal_ids": list(proposal.contributing_signal_ids),
                    "rationale": proposal.rationale,
                    "market_context": _quote_payload(quote),
                },
                correlation_id,
            )

            assessment = self._risk_gate.evaluate(
                side,
                quantity,
                RiskContext(
                    sleeve_equity=account.equity,
                    peak_equity=self._peak_equity,
                    current_position_value=current * quote.last,
                    price=quote.last,
                    remaining_buying_power=remaining_buying_power,
                ),
            )
            await self._append(
                decision_stream,
                EventType.RISK_EVALUATED,
                {
                    "decision_id": decision.id,
                    "verdict": assessment.verdict.value,
                    "reason": assessment.reason,
                    "requested_quantity": str(quantity),
                    "approved_quantity": str(assessment.approved_quantity),
                },
                correlation_id,
            )
            if assessment.verdict is RiskVerdict.REJECTED:
                logger.info(
                    "decision rejected by risk gate: %s",
                    assessment.reason,
                    extra={"correlation_id": correlation_id},
                )
                continue

            # Reserve the approved BUY notional from the shared pool so a later BUY in this same
            # cycle sees the reduced headroom (the shared-pool guard in RiskGate).
            if side is OrderSide.BUY:
                remaining_buying_power -= assessment.approved_quantity * quote.last
                if remaining_buying_power < 0:
                    remaining_buying_power = Decimal("0")

            if not place:
                # Manual mode: the decision + risk verdict are recorded for audit, but no order
                # is placed — the trade is surfaced as a proposal for human approval instead.
                continue
            # DETERMINISTIC idempotency key (review F5): derived from stable inputs, NOT a fresh
            # uuid4 per attempt. A retry or overlapping run for the same (sleeve, symbol, side,
            # cycle-date) mints the SAME client_order_id, so the broker's own idempotency
            # (paper: dict dedupe; Alpaca: 422 -> fetch existing) collapses it to one order.
            client_order_id = f"ord-{self._sleeve.id}-{target.symbol}-{side.value}-{cycle_date}"
            broker_order = await self._place_order.execute(
                PlaceOrderRequest(
                    sleeve_id=self._sleeve.id,
                    decision_id=decision.id,
                    client_order_id=client_order_id,
                    symbol=target.symbol,
                    side=side,
                    quantity=assessment.approved_quantity,
                    correlation_id=correlation_id,
                )
            )
            orders.append(broker_order)

        await self._snapshot(correlation_id)
        return EvaluationResult(
            correlation_id=correlation_id, decisions=tuple(decisions), orders=tuple(orders)
        )

    async def _snapshot(self, correlation_id: str) -> None:
        account = await self._broker.get_account()
        positions = await self._broker.list_positions()
        await self._append(
            f"sleeve-{self._sleeve.id}",
            EventType.PORTFOLIO_SNAPSHOT,
            {
                "sleeve_id": self._sleeve.id,
                "cash": str(account.cash),
                "equity": str(account.equity),
                "positions": {
                    p.symbol: {
                        "quantity": str(p.quantity),
                        "avg_entry_price": str(p.avg_entry_price),
                    }
                    for p in positions
                },
            },
            correlation_id,
        )

    async def _append(
        self, stream_id: str, event_type: str, payload: dict[str, Any], correlation_id: str
    ) -> None:
        await self._event_store.append(
            DomainEvent(
                stream_id=stream_id,
                event_type=event_type,
                payload=payload,
                occurred_at=self._clock.now(),
                correlation_id=correlation_id,
            )
        )


def _pending_by_symbol(orders: Sequence[BrokerOrder]) -> dict[str, Decimal]:
    """Net not-yet-filled quantity per symbol across OPEN broker orders (review F5).

    Only non-terminal orders contribute, and only their unfilled remainder (an order's already-
    filled shares are already reflected in ``list_positions``). BUY remainders add to held, SELL
    remainders subtract, so a queued order counts as exposure in flight and is not re-proposed.
    """
    pending: dict[str, Decimal] = {}
    for order in orders:
        if order.status in TERMINAL_STATUSES:
            continue
        filled = Decimal("0")
        for fill in order.fills:
            filled += fill.quantity
        remainder = order.quantity - filled
        if remainder <= 0:
            continue
        signed = remainder if order.side is OrderSide.BUY else -remainder
        pending[order.symbol] = pending.get(order.symbol, Decimal("0")) + signed
    return pending


def _quote_payload(quote: Quote) -> dict[str, str]:
    return {
        "symbol": quote.symbol,
        "bid": str(quote.bid),
        "ask": str(quote.ask),
        "last": str(quote.last),
        "as_of": quote.as_of.isoformat(),
    }
