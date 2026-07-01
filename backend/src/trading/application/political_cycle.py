"""PoliticalCycle: one end-to-end turn of the political-signal sleeve.

Wires the spec's signal-to-execution flow (section 4.3): ingest fresh signals, read the
whole corroboration window back from the event log, and run it through the strategy, risk
gate, and execution. This is the unit the scheduler invokes; it is also exactly what a
backtest will replay (FR-BT-1), since it reuses the live evaluation path.
"""

from dataclasses import dataclass

from trading.application.evaluate_signals import EvaluateSignals, EvaluationResult
from trading.application.ingest_signals import IngestionResult, IngestSignals
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.projections import signals_from_events


@dataclass(frozen=True, slots=True)
class CycleResult:
    ingestion: IngestionResult
    evaluation: EvaluationResult


class PoliticalCycle:
    def __init__(
        self,
        *,
        ingest: IngestSignals,
        evaluate: EvaluateSignals,
        event_store: EventStorePort,
        clock: ClockPort,
        window_days: int,
        trade_mode: str = "auto",
    ) -> None:
        self._ingest = ingest
        self._evaluate = evaluate
        self._event_store = event_store
        self._clock = clock
        self._window_days = window_days
        self._trade_mode = trade_mode

    @property
    def trade_mode(self) -> str:
        """Current trade mode ('auto' | 'manual'). Mutable at runtime so an operator can
        arm/disarm autonomous trading live (e.g. via POST /cycle/mode) without a restart;
        the next execute() reads the updated value."""
        return self._trade_mode

    @trade_mode.setter
    def trade_mode(self, value: str) -> None:
        self._trade_mode = value

    async def execute(self) -> CycleResult:
        ingestion = await self._ingest.execute()
        events = await self._event_store.read_all()
        window = signals_from_events(events, now=self._clock.now(), window_days=self._window_days)
        # AUTO places orders automatically; MANUAL records the decisions for audit but places
        # nothing — actionable trades are surfaced as proposals (/proposals) for human approval.
        evaluation = await self._evaluate.execute(window, place=self._trade_mode == "auto")
        return CycleResult(ingestion=ingestion, evaluation=evaluation)
