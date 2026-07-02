"""BackfillSignals use case: research-invariant stamping + event-log idempotency.

Drives a FAKE historical source (no network) through the use case and asserts the point-in-time
timestamp, BACKFILLED provenance, per-source pit_confidence, and dedupe behaviour at the
event-log layer.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import FixedClock
from trading.application.backfill_signals import (
    BACKFILL_PIT_CONFIDENCE,
    BackfillError,
    BackfillSignals,
    pit_confidence_for,
)
from trading.application.projections import signals_from_events
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_backfill_seen import InMemoryBackfillSeen

# Disclosure timestamps are historical (well before the FixedClock's "now"), as a real backfill
# window is; the event's occurred_at must be THIS, not the ingest wall-clock.
_DISCLOSED_A = datetime(2024, 3, 1, 20, 30, tzinfo=UTC)
_DISCLOSED_B = datetime(2024, 3, 5, 18, 0, tzinfo=UTC)


def _hist_signal(
    *, signal_id: str, source_uid: str | None, disclosed_at: datetime, ticker: str = "AAPL"
) -> Signal:
    """A signal as a historical fetcher would emit it: created_at = disclosure timestamp, the
    source-native id inside raw_payload, and (deliberately) a NON-backfilled provenance so the
    test proves the USE CASE stamps it, not the adapter."""
    payload: dict[str, object] = {"source": "fake-historical"}
    if source_uid is not None:
        payload["source_uid"] = source_uid
    return Signal(
        id=signal_id,
        created_at=disclosed_at,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=None,
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.75,
        raw_payload=payload,
        provenance=Provenance.REAL,  # NOT backfilled: the use case must overwrite this
        pit_confidence="approximate",  # NOT the source's tier: the use case must overwrite this
    )


class _FakeHistoricalSource:
    """Returns a fixed batch for any window; `name` selects the pit_confidence tier."""

    def __init__(self, name: str, batch: Sequence[Signal]) -> None:
        self._name = name
        self._batch = list(batch)

    @property
    def name(self) -> str:
        return self._name

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        return list(self._batch)


def _make(
    source: _FakeHistoricalSource, clock: FixedClock
) -> tuple[BackfillSignals, InMemoryEventStore, InMemoryBackfillSeen]:
    store = InMemoryEventStore(clock)
    seen = InMemoryBackfillSeen()
    return (
        BackfillSignals(source=source, event_store=store, seen=seen, clock=clock),
        store,
        seen,
    )


async def test_stamps_backfilled_provenance_and_source_pit_confidence(clock: FixedClock) -> None:
    source = _FakeHistoricalSource(
        "sec_form4",  # tier "exact"
        [_hist_signal(signal_id="s1", source_uid="0001-24-000001", disclosed_at=_DISCLOSED_A)],
    )
    backfill, store, _seen = _make(source, clock)

    await backfill.execute(
        start=_DISCLOSED_A,
        end=_DISCLOSED_B + timedelta(days=1),
        correlation_id="run-1",
        dry_run=False,
    )

    [signal] = signals_from_events(await store.read_all())
    assert signal.provenance is Provenance.BACKFILLED
    assert signal.pit_confidence == "exact"  # sec_form4 tier, NOT the source's "approximate"


async def test_approximate_tier_source_stamps_approximate(clock: FixedClock) -> None:
    source = _FakeHistoricalSource(
        "senate_efd",  # tier "approximate"
        [_hist_signal(signal_id="s1", source_uid="uid-1", disclosed_at=_DISCLOSED_A)],
    )
    backfill, store, _seen = _make(source, clock)
    await backfill.execute(
        start=_DISCLOSED_A, end=_DISCLOSED_B, correlation_id="run-1", dry_run=False
    )
    [signal] = signals_from_events(await store.read_all())
    assert signal.pit_confidence == "approximate"


async def test_occurred_at_is_disclosure_timestamp_not_clock(clock: FixedClock) -> None:
    # The event's occurred_at must be the signal's disclosure date, NOT clock.now() (the ingest
    # wall-clock) and NOT any later event/trade date — the load-bearing PIT invariant.
    source = _FakeHistoricalSource(
        "sec_form4",
        [
            _hist_signal(signal_id="s1", source_uid="u1", disclosed_at=_DISCLOSED_A),
            _hist_signal(signal_id="s2", source_uid="u2", disclosed_at=_DISCLOSED_B),
        ],
    )
    backfill, store, _seen = _make(source, clock)
    await backfill.execute(
        start=_DISCLOSED_A,
        end=_DISCLOSED_B + timedelta(days=1),
        correlation_id="run-1",
        dry_run=False,
    )

    events = [e for e in await store.read_all() if e.event.event_type == "signal.received"]
    occurred = {e.event.payload["signal_id"]: e.event.occurred_at for e in events}
    assert occurred["s1"] == _DISCLOSED_A
    assert occurred["s2"] == _DISCLOSED_B
    # And definitely not the fixed clock's "now".
    assert clock.now() not in occurred.values()


async def test_rerunning_a_window_appends_no_duplicate_events(clock: FixedClock) -> None:
    source = _FakeHistoricalSource(
        "sec_form4",
        [
            _hist_signal(signal_id="s1", source_uid="u1", disclosed_at=_DISCLOSED_A),
            _hist_signal(signal_id="s2", source_uid="u2", disclosed_at=_DISCLOSED_B),
        ],
    )
    backfill, store, _seen = _make(source, clock)

    first = await backfill.execute(
        start=_DISCLOSED_A,
        end=_DISCLOSED_B + timedelta(days=1),
        correlation_id="run-1",
        dry_run=False,
    )
    count_after_first = len(
        [e for e in await store.read_all() if e.event.event_type == "signal.received"]
    )
    second = await backfill.execute(
        start=_DISCLOSED_A,
        end=_DISCLOSED_B + timedelta(days=1),
        correlation_id="run-2",
        dry_run=False,
    )
    count_after_second = len(
        [e for e in await store.read_all() if e.event.event_type == "signal.received"]
    )

    assert first.new == 2 and first.duplicate == 0
    assert second.new == 0 and second.duplicate == 2  # both seen from the first run
    assert count_after_first == count_after_second == 2  # no double-append at the event-log layer


async def test_intra_batch_duplicate_uid_recorded_once(clock: FixedClock) -> None:
    # A fetcher emitting the same source_uid twice in one batch must still record it once.
    source = _FakeHistoricalSource(
        "sec_form4",
        [
            _hist_signal(signal_id="s1", source_uid="dup", disclosed_at=_DISCLOSED_A),
            _hist_signal(signal_id="s2", source_uid="dup", disclosed_at=_DISCLOSED_B),
        ],
    )
    backfill, store, _seen = _make(source, clock)
    report = await backfill.execute(
        start=_DISCLOSED_A, end=_DISCLOSED_B, correlation_id="run-1", dry_run=False
    )
    assert report.fetched == 2
    assert report.new == 1
    assert report.duplicate == 1
    recorded = [e for e in await store.read_all() if e.event.event_type == "signal.received"]
    assert len(recorded) == 1


async def test_signals_without_source_uid_are_refused_not_recorded(clock: FixedClock) -> None:
    source = _FakeHistoricalSource(
        "sec_form4",
        [
            _hist_signal(signal_id="s1", source_uid=None, disclosed_at=_DISCLOSED_A),
            _hist_signal(signal_id="s2", source_uid="u2", disclosed_at=_DISCLOSED_B),
        ],
    )
    backfill, store, _seen = _make(source, clock)
    report = await backfill.execute(
        start=_DISCLOSED_A, end=_DISCLOSED_B, correlation_id="run-1", dry_run=False
    )
    assert report.refused == 1
    assert report.new == 1
    recorded = [e for e in await store.read_all() if e.event.event_type == "signal.received"]
    assert len(recorded) == 1


async def test_dry_run_writes_nothing_but_reports_the_split(clock: FixedClock) -> None:
    source = _FakeHistoricalSource(
        "sec_form4",
        [_hist_signal(signal_id="s1", source_uid="u1", disclosed_at=_DISCLOSED_A)],
    )
    backfill, store, seen = _make(source, clock)
    report = await backfill.execute(
        start=_DISCLOSED_A, end=_DISCLOSED_B, correlation_id="run-1", dry_run=True
    )
    assert report.new == 1 and report.committed is False
    assert report.correlation_id is None
    assert len(await store.read_all()) == 0  # nothing appended
    assert await seen.seen("sec_form4", ["u1"]) == set()  # nothing marked seen


async def test_news_source_cannot_be_backfilled(clock: FixedClock) -> None:
    # news is excluded from the tier table: constructing the use case must fail fast.
    assert "news" not in BACKFILL_PIT_CONFIDENCE
    with pytest.raises(BackfillError):
        pit_confidence_for("news")
    source = _FakeHistoricalSource("news", [])
    store = InMemoryEventStore(clock)
    with pytest.raises(BackfillError):
        BackfillSignals(source=source, event_store=store, seen=InMemoryBackfillSeen(), clock=clock)
