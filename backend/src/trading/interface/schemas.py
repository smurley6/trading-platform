"""Request/response DTOs (FR-API-2): explicit Pydantic schemas, decoupled from domain."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from trading.application.backtest import BacktestResult
from trading.application.cancel_order import CancelResult
from trading.application.live_trading import LiveReadinessReport
from trading.application.ports.broker import BrokerOrder
from trading.application.ports.market_data import Quote
from trading.application.ports.outcome_labels import OutcomeLabel
from trading.application.ports.read_model import SignalRecord
from trading.application.proposals import PredictedCandidate, ProposalCandidate
from trading.application.reconcile import ReconciliationReport
from trading.application.risk_precheck import RiskPrecheckResult
from trading.application.sleeve_registry import SleeveState
from trading.application.sync_orders import SyncReport
from trading.domain.events import StoredEvent
from trading.domain.positions import MarkedPosition
from trading.domain.signals import Signal, SignalSource
from trading.domain.sleeves import Sleeve, TradingMode
from trading.interface.system_status import DataSourceStatus, ScheduledLoop


class SignalIn(BaseModel):
    id: str | None = None
    source_type: SignalSource
    ticker: str
    politician_id: str | None = None
    sentiment: float = Field(ge=-1, le=1)
    politician_weight: float = Field(default=1.0, ge=0)
    source_reliability: float = Field(default=1.0, ge=0, le=1)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self, now: datetime) -> Signal:
        return Signal(
            id=self.id or str(uuid4()),
            created_at=now,
            source_type=self.source_type,
            ticker=self.ticker,
            politician_id=self.politician_id,
            sentiment=self.sentiment,
            politician_weight=self.politician_weight,
            source_reliability=self.source_reliability,
            raw_payload=self.raw_payload,
        )


class EventOut(BaseModel):
    event_id: str
    stream_id: str
    seq: int
    event_type: str
    schema_version: int
    occurred_at: datetime
    recorded_at: datetime
    correlation_id: str
    payload: dict[str, Any]

    @classmethod
    def from_stored(cls, stored: StoredEvent) -> "EventOut":
        event = stored.event
        return cls(
            event_id=str(event.event_id),
            stream_id=event.stream_id,
            seq=stored.seq,
            event_type=event.event_type,
            schema_version=event.schema_version,
            occurred_at=event.occurred_at,
            recorded_at=stored.recorded_at,
            correlation_id=event.correlation_id,
            payload=event.payload,
        )


class OrderOut(BaseModel):
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    status: str

    @classmethod
    def from_broker(cls, order: BrokerOrder) -> "OrderOut":
        return cls(
            client_order_id=order.client_order_id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            status=order.status.value,
        )


class CancelOut(BaseModel):
    correlation_id: str
    sleeve_id: str
    client_order_id: str
    broker_order_id: str
    status: str

    @classmethod
    def from_result(cls, result: CancelResult) -> "CancelOut":
        return cls(
            correlation_id=result.correlation_id,
            sleeve_id=result.sleeve_id,
            client_order_id=result.order.client_order_id,
            broker_order_id=result.order.broker_order_id,
            status=result.order.status.value,
        )


class EvaluateOut(BaseModel):
    correlation_id: str
    orders: list[OrderOut]


class SleeveOut(BaseModel):
    id: str
    strategy_id: str
    allocated_capital: Decimal
    mode: str
    status: str

    @classmethod
    def from_domain(cls, sleeve: Sleeve) -> "SleeveOut":
        return cls(
            id=sleeve.id,
            strategy_id=sleeve.strategy_id,
            allocated_capital=sleeve.allocated_capital,
            mode=sleeve.mode.value,
            status=sleeve.status.value,
        )

    @classmethod
    def from_state(cls, state: SleeveState) -> "SleeveOut":
        # The live mode/status live in the registry, not the immutable Sleeve entity.
        return cls(
            id=state.sleeve.id,
            strategy_id=state.sleeve.strategy_id,
            allocated_capital=state.sleeve.allocated_capital,
            mode=state.mode.value,
            status=state.status.value,
        )


class SignalFeedItem(BaseModel):
    signal_id: str
    occurred_at: datetime
    source_type: str
    # None = unresolved entity (Task 04b backfilled signal whose asset name never matched a
    # ticker). Visible in the feed for research purposes; excluded from conviction/proposals.
    ticker: str | None
    politician_id: str | None
    sentiment: float
    politician_weight: float
    source_reliability: float
    # The source record that was analyzed (headline/filing/award details) — surfaced so the UI
    # can show "the actual text that was analyzed" when a signal row is clicked.
    raw_payload: dict[str, Any]

    @classmethod
    def from_record(cls, record: SignalRecord) -> "SignalFeedItem":
        signal = record.signal
        return cls(
            signal_id=signal.id,
            occurred_at=record.occurred_at,
            source_type=signal.source_type.value,
            ticker=signal.ticker,
            politician_id=signal.politician_id,
            sentiment=signal.sentiment,
            politician_weight=signal.politician_weight,
            source_reliability=signal.source_reliability,
            raw_payload=signal.raw_payload,
        )


class ConvictionItem(BaseModel):
    ticker: str
    score: float
    action: str
    contributing_signal_ids: list[str]


class ProposalItem(BaseModel):
    symbol: str
    side: str
    quantity: Decimal
    target_quantity: Decimal
    current_quantity: Decimal
    conviction_score: float
    action: str
    contributing_signal_ids: list[str]
    rationale: str

    @classmethod
    def from_candidate(cls, c: ProposalCandidate) -> "ProposalItem":
        return cls(
            symbol=c.symbol,
            side=c.side,
            quantity=c.quantity,
            target_quantity=c.target_quantity,
            current_quantity=c.current_quantity,
            conviction_score=c.conviction_score,
            action=c.action,
            contributing_signal_ids=list(c.contributing_signal_ids),
            rationale=c.rationale,
        )


class PredictedItem(BaseModel):
    symbol: str
    conviction_score: float
    action: str
    distinct_source_types: list[str]
    contributing_signal_ids: list[str]

    @classmethod
    def from_candidate(cls, c: PredictedCandidate) -> "PredictedItem":
        return cls(
            symbol=c.symbol,
            conviction_score=c.conviction_score,
            action=c.action,
            distinct_source_types=list(c.distinct_source_types),
            contributing_signal_ids=list(c.contributing_signal_ids),
        )


class ProposalsOut(BaseModel):
    mode: str
    proposals: list[ProposalItem]
    predicted: list[PredictedItem]


class ExecuteProposalsIn(BaseModel):
    symbols: list[str] | None = None


class ExecuteProposalsOut(BaseModel):
    correlation_id: str
    orders: list[OrderOut]


class IngestOut(BaseModel):
    ingestion_correlation_id: str
    signals_ingested: int
    failed_sources: list[str]
    evaluation_correlation_id: str
    orders: list[OrderOut]


class BacktestPointOut(BaseModel):
    time: int  # unix seconds, for the charting library
    equity: float


class BacktestTradeOut(BaseModel):
    symbol: str
    side: str
    price: float
    quantity: float
    filled_at: datetime


class BacktestOut(BaseModel):
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    # Risk/return metrics (always present; 0.0 when undefined, e.g. a no-trade run).
    sharpe: float
    sortino: float
    annualized_return_pct: float
    volatility: float
    benchmark_total_return_pct: float
    # Trade stats — null when the run realised no round-trips (frontend renders "—").
    win_rate: float | None
    profit_factor: float | None
    avg_win: float | None
    avg_loss: float | None
    num_trades: int
    decision_count: int
    equity_curve: list[BacktestPointOut]
    benchmark_curve: list[BacktestPointOut]
    trades: list[BacktestTradeOut]

    @classmethod
    def from_result(cls, result: BacktestResult) -> "BacktestOut":
        stats = result.trade_stats
        return cls(
            starting_equity=float(result.starting_equity),
            ending_equity=float(result.ending_equity),
            total_return_pct=result.total_return_pct,
            max_drawdown_pct=result.max_drawdown_pct,
            sharpe=result.sharpe_ratio,
            sortino=result.sortino_ratio,
            annualized_return_pct=result.annualized_return_pct,
            volatility=result.return_volatility,
            benchmark_total_return_pct=result.benchmark_total_return_pct,
            win_rate=stats.win_rate if stats else None,
            profit_factor=stats.profit_factor if stats else None,
            avg_win=float(stats.average_win) if stats else None,
            avg_loss=float(stats.average_loss) if stats else None,
            num_trades=len(result.trades),
            decision_count=result.decision_count,
            equity_curve=[
                BacktestPointOut(time=int(p.when.timestamp()), equity=float(p.equity))
                for p in result.equity_curve
            ],
            benchmark_curve=[
                BacktestPointOut(time=int(p.when.timestamp()), equity=float(p.equity))
                for p in result.benchmark_curve
            ],
            trades=[
                BacktestTradeOut(
                    symbol=t.symbol,
                    side=t.side,
                    price=float(t.price),
                    quantity=float(t.quantity),
                    filled_at=t.filled_at,
                )
                for t in result.trades
            ],
        )


class LiveCheckOut(BaseModel):
    name: str
    passed: bool
    detail: str


class LiveReadinessOut(BaseModel):
    sleeve_id: str
    ready: bool
    checks: list[LiveCheckOut]

    @classmethod
    def from_report(cls, report: LiveReadinessReport) -> "LiveReadinessOut":
        return cls(
            sleeve_id=report.sleeve_id,
            ready=report.ready,
            checks=[
                LiveCheckOut(name=c.name, passed=c.passed, detail=c.detail) for c in report.checks
            ],
        )


class RiskCheckOut(BaseModel):
    # One named risk gate's outcome, iterated by §5.2's evidence pane to render
    # "RISK PRE-CHECK ✓ size ✓ concentration ✓ BP" (label + passed) with limit/current/
    # projected/reason available on hover or expand.
    name: str  # stable machine key: kill_switch|price|sleeve_equity|drawdown|trade_size|...
    label: str  # human label rendered in the evidence pane: "size", "concentration", "BP", ...
    passed: bool
    limit: str | None  # configured limit as text; None for non-limit gates (kill switch/price)
    current: str | None  # current value going in (money/pct text); None when not applicable
    projected: str | None  # value after this order; None when not applicable
    reason: str  # human sentence


class RiskPrecheckOut(BaseModel):
    # Read-only risk pre-check for a single prospective order (§5.2). `enabled` is False (with an
    # empty checks list) when the pre-check is not configured, so the SPA renders unconditionally
    # exactly like CapitalOut's disabled shape.
    enabled: bool
    symbol: str
    side: str
    requested_quantity: Decimal
    approved_quantity: Decimal
    verdict: str  # RiskVerdict.value: approved|modified|rejected
    reason: str
    checks: list[RiskCheckOut]

    @classmethod
    def from_result(cls, result: RiskPrecheckResult) -> "RiskPrecheckOut":
        return cls(
            enabled=True,
            symbol=result.symbol,
            side=result.side,
            requested_quantity=result.requested_quantity,
            approved_quantity=result.approved_quantity,
            verdict=result.verdict,
            reason=result.reason,
            checks=[
                RiskCheckOut(
                    name=c.name,
                    label=c.label,
                    passed=c.passed,
                    limit=c.limit,
                    current=c.current,
                    projected=c.projected,
                    reason=c.reason,
                )
                for c in result.checks
            ],
        )

    @classmethod
    def disabled(cls) -> "RiskPrecheckOut":
        return cls(
            enabled=False,
            symbol="",
            side="",
            requested_quantity=Decimal("0"),
            approved_quantity=Decimal("0"),
            verdict="",
            reason="",
            checks=[],
        )


class SleeveModeIn(BaseModel):
    mode: TradingMode
    confirm: bool = False


class SleeveModeOut(BaseModel):
    sleeve: SleeveOut
    readiness: LiveReadinessOut | None = None


class CycleModeIn(BaseModel):
    mode: Literal["auto", "manual"]


class CycleModeOut(BaseModel):
    mode: str


class QuoteOut(BaseModel):
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    as_of: datetime

    @classmethod
    def from_quote(cls, quote: Quote) -> "QuoteOut":
        return cls(
            symbol=quote.symbol,
            bid=quote.bid,
            ask=quote.ask,
            last=quote.last,
            as_of=quote.as_of,
        )


class QuotesOut(BaseModel):
    enabled: bool
    as_of: datetime | None
    quotes: list[QuoteOut]


class TrackedSymbolsOut(BaseModel):
    symbols: list[str]


class LivePositionOut(BaseModel):
    symbol: str
    sleeve_id: str
    quantity: Decimal
    avg_price: Decimal
    last: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal

    @classmethod
    def from_marked(cls, marked: MarkedPosition) -> "LivePositionOut":
        return cls(
            symbol=marked.symbol,
            sleeve_id=marked.sleeve_id,
            quantity=marked.quantity,
            avg_price=marked.avg_price,
            last=marked.last,
            market_value=marked.market_value,
            unrealized_pnl=marked.unrealized_pnl,
        )


class LivePositionsOut(BaseModel):
    enabled: bool
    positions: list[LivePositionOut]


class CapitalAllocationOut(BaseModel):
    # One algo's slice of the shared pool: what it currently holds in the one book (marked to the
    # live feed when available). Attribution only — the capital itself is the single pool below.
    sleeve_id: str
    strategy_id: str
    market_value: Decimal
    unrealized_pnl: Decimal
    positions: int


class CapitalOut(BaseModel):
    # The single shared capital pool = the live Alpaca account both algos draw from. `enabled` is
    # False (and figures are zero) when no real account is wired, so the SPA renders a graceful
    # empty state exactly like the other live feeds.
    enabled: bool
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    allocations: list[CapitalAllocationOut]


class KillSwitchIn(BaseModel):
    engaged: bool


class KillSwitchOut(BaseModel):
    engaged: bool


class DriftOut(BaseModel):
    kind: str
    key: str
    local: str
    broker: str


class ReconcileOut(BaseModel):
    correlation_id: str
    sleeve_id: str | None
    clean: bool
    drifts: list[DriftOut]
    checked_positions: int
    checked_orders: int

    @classmethod
    def from_report(cls, report: ReconciliationReport) -> "ReconcileOut":
        return cls(
            correlation_id=report.correlation_id,
            sleeve_id=report.sleeve_id,
            clean=report.clean,
            drifts=[
                DriftOut(kind=d.kind, key=d.key, local=d.local, broker=d.broker)
                for d in report.drifts
            ],
            checked_positions=report.checked_positions,
            checked_orders=report.checked_orders,
        )


class OrderSyncOut(BaseModel):
    correlation_id: str
    sleeve_id: str
    fills_emitted: int
    status_changes_emitted: int
    checked_orders: int

    @classmethod
    def from_report(cls, report: SyncReport) -> "OrderSyncOut":
        return cls(
            correlation_id=report.correlation_id,
            sleeve_id=report.sleeve_id,
            fills_emitted=report.fills_emitted,
            status_changes_emitted=report.status_changes_emitted,
            checked_orders=report.checked_orders,
        )


# --- System screen (UI Phase D, docs/UI-DESIGN.md §5.5) -----------------------------------------


class RiskCapOut(BaseModel):
    # One risk cap rendered as a configured-vs-ceiling bar (§5.5 Risk pane): the configured value
    # must sit at or under the hardened live ceiling before a sleeve may go live. Decimals
    # serialize as strings (like CapitalOut); the frontend parses with Number(...).
    name: str  # stable key: trade_size | concentration | drawdown
    label: str  # human label: "Per-trade size", "Position concentration", "Max drawdown"
    configured: Decimal
    ceiling: Decimal
    unit: str  # "%" — all three caps are percentages


class SystemRiskOut(BaseModel):
    # The Risk pane's backing data (§5.5): the configured caps + their live ceilings and the global
    # kill-switch state. Kill-switch HISTORY and the current drawdown gauge are derived on the
    # frontend from the event log / pool equity curve (no backend change), so only the ceilings and
    # the live engaged flag come from here. `enabled` is False only if risk limits were never wired.
    enabled: bool
    kill_switch_engaged: bool
    caps: list[RiskCapOut]


class JobStatusOut(BaseModel):
    # One scheduler loop's status (§5.5 Jobs pane): interval + last/next run. `enabled` is False for
    # a loop whose interval is 0 or whose dependency is unwired — still listed so the operator sees
    # the full set of loops. last_run/next_run are null until the loop has run / been scheduled.
    name: str
    description: str
    interval_seconds: int
    enabled: bool
    last_run: datetime | None
    next_run: datetime | None

    @classmethod
    def from_loop(
        cls, loop: ScheduledLoop, *, last_run: datetime | None, next_run: datetime | None
    ) -> "JobStatusOut":
        return cls(
            name=loop.name,
            description=loop.description,
            interval_seconds=loop.interval_seconds,
            enabled=loop.enabled,
            last_run=last_run,
            next_run=next_run,
        )


class DataSourceOut(BaseModel):
    # One ingest adapter, as a read-only mirror of config (§5.5 Data sources pane): enabled state +
    # reliability prior. No toggle — this reflects env truth, it does not mutate it.
    source_id: str
    label: str
    enabled: bool
    reliability_prior: float

    @classmethod
    def from_status(cls, status: DataSourceStatus) -> "DataSourceOut":
        return cls(
            source_id=status.source_id,
            label=status.label,
            enabled=status.enabled,
            reliability_prior=status.reliability_prior,
        )


class SystemStatusOut(BaseModel):
    # The System screen's single read (§5.5): the read-only operational state of the platform —
    # risk caps/ceilings + kill-switch, the scheduler loops, and the data-source config mirror.
    # Reconciliation (last run / drift) and kill-switch history use the existing /reconcile and
    # /events surfaces, so they are not duplicated here.
    risk: SystemRiskOut
    jobs: list[JobStatusOut]
    data_sources: list[DataSourceOut]


# --- Research surfaces on Signals (UI Phase D §5.5 / §8) -----------------------------------------


class OutcomeLabelOut(BaseModel):
    # One (signal, horizon) outcome-study row from the rebuildable outcome_labels projection. The
    # Signals research surface joins these by signal_id to render 1/5/20-day forward abnormal
    # returns. `benchmark` is "sector:<ETF>" or "market"; the frontend splits on the prefix for the
    # sector-vs-market column (§5.5). `abnormal_return` is null for an unresolvable label.
    signal_id: str
    ticker: str
    horizon_days: int
    abnormal_return: float | None
    benchmark: str
    status: str  # ok | unresolvable
    reason: str | None

    @classmethod
    def from_label(cls, label: OutcomeLabel) -> "OutcomeLabelOut":
        return cls(
            signal_id=label.signal_id,
            ticker=label.ticker,
            horizon_days=label.horizon_days,
            abnormal_return=label.abnormal_return,
            benchmark=label.benchmark,
            status=label.status,
            reason=label.reason,
        )


class SignalLabelsOut(BaseModel):
    # The outcome labels for the currently-loaded signal window (GET /signals/labels). `enabled` is
    # False (empty labels) when no labeler run exists yet, so the SPA renders unconditionally and
    # shows "—" columns — mirroring the other graceful disabled-feed shapes.
    enabled: bool
    label_run_id: str | None
    labels: list[OutcomeLabelOut]
