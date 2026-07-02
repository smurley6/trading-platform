"""Anti-placeholder guard: SYNTHETIC data is structurally kept out of real decisions."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.data_integrity import (
    SyntheticDataError,
    ensure_real_signals,
    non_real_signals,
    tradeable_signals,
)
from trading.application.ingest_signals import IngestSignals
from trading.application.projections import _signal_from_payload, signals_from_events
from trading.application.signal_recording import record_signals
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal
from trading.infrastructure.persistence.memory import InMemoryEventStore


class _FakeSource:
    def __init__(self, batch: list[Signal]) -> None:
        self._batch = batch

    @property
    def name(self) -> str:
        return "fake"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        return self._batch


def test_default_provenance_is_synthetic(make_signal: SignalFactory) -> None:
    # Fail-closed: an un-stamped signal is SYNTHETIC and can never drive a real order.
    assert make_signal().provenance is Provenance.SYNTHETIC


def test_ensure_real_signals_passes_when_all_real(make_signal: SignalFactory) -> None:
    ensure_real_signals([make_signal(provenance=Provenance.REAL) for _ in range(3)])


def test_ensure_real_signals_raises_on_any_synthetic(make_signal: SignalFactory) -> None:
    mixed = [make_signal(provenance=Provenance.REAL), make_signal(provenance=Provenance.SYNTHETIC)]
    with pytest.raises(SyntheticDataError):
        ensure_real_signals(mixed)


def test_non_real_signals_returns_only_fakes(make_signal: SignalFactory) -> None:
    real = make_signal(provenance=Provenance.REAL)
    fake = make_signal(provenance=Provenance.SYNTHETIC)
    assert non_real_signals([real, fake]) == [fake]


def test_ensure_real_signals_does_not_raise_on_backfilled(make_signal: SignalFactory) -> None:
    # The execution wall for BACKFILLED is filter-not-abort (tradeable_signals), NOT this
    # abort-the-batch guard — otherwise any backfill run would halt live evaluation until the
    # backfilled signals aged out of the corroboration window (RESEARCH-PLAN §2).
    ensure_real_signals([make_signal(provenance=Provenance.BACKFILLED)])


def test_tradeable_signals_drops_backfilled_but_keeps_real_and_synthetic(
    make_signal: SignalFactory,
) -> None:
    real = make_signal(provenance=Provenance.REAL)
    backfilled = make_signal(provenance=Provenance.BACKFILLED)
    synthetic = make_signal(provenance=Provenance.SYNTHETIC)
    assert tradeable_signals([real, backfilled, synthetic]) == [real, synthetic]


async def test_backfilled_signals_appear_in_signals_from_events_projection(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # BACKFILLED must never trade, but IS permitted into projections and (future) labeler/
    # learner paths — unlike SYNTHETIC, which is only kept for tests/backtests.
    event_store = InMemoryEventStore(clock)
    backfilled = make_signal(provenance=Provenance.BACKFILLED)
    await record_signals(
        event_store=event_store, clock=clock, signals=[backfilled], correlation_id="c"
    )

    [reconstructed] = signals_from_events(await event_store.read_all())

    assert reconstructed.provenance is Provenance.BACKFILLED
    assert reconstructed.id == backfilled.id


async def test_provenance_survives_event_roundtrip(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    # The trade-time guard reads signals back from the event log, so REAL provenance MUST
    # serialize into the SIGNAL_RECEIVED payload and reconstruct intact.
    event_store = InMemoryEventStore(clock)
    await record_signals(
        event_store=event_store,
        clock=clock,
        signals=[make_signal(provenance=Provenance.REAL)],
        correlation_id="c",
    )
    [reconstructed] = signals_from_events(await event_store.read_all())
    assert reconstructed.provenance is Provenance.REAL


async def test_pit_confidence_survives_event_roundtrip(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    await record_signals(
        event_store=event_store,
        clock=clock,
        signals=[make_signal(provenance=Provenance.BACKFILLED, pit_confidence="approximate")],
        correlation_id="c",
    )
    [reconstructed] = signals_from_events(await event_store.read_all())
    assert reconstructed.pit_confidence == "approximate"


def test_pit_confidence_defaults_to_exact_for_legacy_payloads() -> None:
    # Legacy events recorded before pit_confidence existed have no such key in their payload;
    # they must default to "exact" so pre-backfill REAL/SYNTHETIC signals are undisturbed.
    legacy_payload: dict[str, object] = {
        "signal_id": "s-legacy",
        "created_at": datetime(2026, 6, 1, tzinfo=UTC).isoformat(),
        "source_type": "statement",
        "ticker": "AAPL",
        "politician_id": None,
        "sentiment": 0.5,
        "politician_weight": 1.0,
        "source_reliability": 1.0,
        "raw_payload": {},
        # no "provenance" or "pit_confidence" key at all
    }
    signal = _signal_from_payload(legacy_payload)
    assert signal.pit_confidence == "exact"
    assert signal.provenance is Provenance.SYNTHETIC


async def test_ingestion_gate_drops_synthetic_in_real_mode(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    real = make_signal(ticker="AAPL", provenance=Provenance.REAL)
    fake = make_signal(ticker="MSFT", provenance=Provenance.SYNTHETIC)
    ingest = IngestSignals(
        sources=[_FakeSource([real, fake])],
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
        require_real_data=True,
    )

    result = await ingest.execute()

    recorded = signals_from_events(await event_store.read_all())
    assert [s.id for s in result.signals] == [real.id]
    assert [s.id for s in recorded] == [real.id]


async def test_ingestion_keeps_everything_when_not_real_mode(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    event_store = InMemoryEventStore(clock)
    ingest = IngestSignals(
        sources=[
            _FakeSource(
                [
                    make_signal(ticker="AAPL", provenance=Provenance.REAL),
                    make_signal(ticker="MSFT", provenance=Provenance.SYNTHETIC),
                ]
            )
        ],
        event_store=event_store,
        clock=clock,
        lookback_start=clock.now() - timedelta(days=1),
    )

    result = await ingest.execute()

    assert len(result.signals) == 2  # both kept: the gate is off
