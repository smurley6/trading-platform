"""Synthetic-data purge job, driven against an in-memory event store + read model.

NEVER touches a real DB: ``build_dependencies`` is monkeypatched to return a lightweight deps
backed by the same in-memory wiring production uses (ProjectingEventStore + InMemoryReadModel),
seeded with REAL, BACKFILLED, and SYNTHETIC signals. We assert the dry run changes nothing,
``--confirm`` purges ONLY SYNTHETIC (task-08: BACKFILLED must survive untouched), and the
summaries report REAL/BACKFILLED/SYNTHETIC counts separately alongside retained-by-provenance
counts.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import pytest

from tests.conftest import FixedClock, SignalFactory
from trading.application.data_integrity import purge_eligible
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.read_model import ReadModelPort
from trading.application.projections import signals_from_events
from trading.application.signal_recording import record_signals, signal_payload
from trading.config import Settings
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel
from trading.infrastructure.persistence.projecting import ProjectingEventStore
from trading.jobs import purge_synthetic

# The composition root reads required risk caps from the environment; the purge job calls
# Settings() for real before our patched build_dependencies. Mirror the cycle-job test: set the
# caps and CLEAR every credential / DATABASE_URL so a developer .env or shell can NEVER leak a
# real DB or live credentials into these tests (we also pass _env_file=None below).
_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "DATABASE_URL",
    "NEWS_FEED_URLS",
    "USASPENDING_ENABLED",
    "DISCLOSED_TRADE_LEG_ENABLED",
    "CONGRESS_TRADES_ENDPOINT",
    "FMP_API_KEY",
    "ENABLE_LIVE_QUOTES",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "ENABLE_TRADE_UPDATES_STREAM",
    "REQUIRE_REAL_DATA",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_LIVE_KEY_ID",
    "ALPACA_LIVE_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    # Never read a developer .env file when the job constructs Settings().
    monkeypatch.setattr(purge_synthetic, "Settings", lambda: Settings(_env_file=None))


@dataclass
class _FakeDeps:
    """Just the AppDeps surface the purge job reads (event_store, read_model, aclosers)."""

    event_store: EventStorePort
    read_model: ReadModelPort
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


async def _seed(clock: FixedClock) -> tuple[ProjectingEventStore, InMemoryReadModel]:
    read_model = InMemoryReadModel()
    store = ProjectingEventStore(InMemoryEventStore(clock), read_model)
    return store, read_model


def _patch_build(
    monkeypatch: pytest.MonkeyPatch, store: EventStorePort, read_model: ReadModelPort
) -> None:
    deps = _FakeDeps(event_store=store, read_model=read_model)
    # Settings() is patched (autouse) to read no .env and DATABASE_URL is cleared; here the heavy
    # composition root itself is swapped so no real adapters / DB are ever constructed.
    monkeypatch.setattr(purge_synthetic, "build_dependencies", lambda _settings: deps)


async def _make_signals(
    store: ProjectingEventStore, clock: FixedClock, make_signal: SignalFactory
) -> tuple[Signal, Signal, Signal, Signal]:
    real_a = make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL)
    real_b = make_signal(source_type=SignalSource.CONTRACT, provenance=Provenance.REAL)
    fake_a = make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.SYNTHETIC)
    fake_b = make_signal(source_type=SignalSource.DISCLOSED_TRADE, provenance=Provenance.SYNTHETIC)
    await record_signals(
        event_store=store,
        clock=clock,
        signals=[real_a, real_b, fake_a, fake_b],
        correlation_id="seed",
    )
    return real_a, real_b, fake_a, fake_b


async def test_dry_run_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    clock: FixedClock,
    make_signal: SignalFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, read_model = await _seed(clock)
    await _make_signals(store, clock, make_signal)
    _patch_build(monkeypatch, store, read_model)

    rc = await purge_synthetic.run_purge(confirm=False)

    assert rc == 0
    # Read side is untouched: both synthetic signals are still present.
    after = list(await read_model.all_signals())
    assert len(purge_eligible(after)) == 2
    assert len(after) == 4
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "SYNTHETIC signals identified for purge: 2" in out


async def test_confirm_makes_read_model_synthetic_free(
    monkeypatch: pytest.MonkeyPatch,
    clock: FixedClock,
    make_signal: SignalFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, read_model = await _seed(clock)
    real_a, real_b, _fake_a, _fake_b = await _make_signals(store, clock, make_signal)
    _patch_build(monkeypatch, store, read_model)

    rc = await purge_synthetic.run_purge(confirm=True)

    assert rc == 0
    after = list(await read_model.all_signals())
    # The live/read-model view is now synthetic-free: only the two real signals survive (no
    # BACKFILLED was seeded here — that mixed-provenance case is covered separately below).
    assert purge_eligible(after) == []
    assert {s.id for s in after} == {real_a.id, real_b.id}
    out = capsys.readouterr().out
    assert "purged 2 synthetic signal(s)" in out
    assert "read side is now SYNTHETIC-free (REAL + BACKFILLED retained)." in out


async def test_confirm_preserves_append_only_event_log(
    monkeypatch: pytest.MonkeyPatch,
    clock: FixedClock,
    make_signal: SignalFactory,
) -> None:
    store, read_model = await _seed(clock)
    await _make_signals(store, clock, make_signal)
    _patch_build(monkeypatch, store, read_model)

    before_count = len(await store.read_all())
    await purge_synthetic.run_purge(confirm=True)

    # The event log is the append-only system of record: nothing is deleted from it. All four
    # signal.received events (real AND synthetic) remain on the log; only the read side changed.
    assert len(await store.read_all()) == before_count


async def test_confirm_purges_synthetic_and_legacy_but_retains_real_and_backfilled(
    monkeypatch: pytest.MonkeyPatch,
    clock: FixedClock,
    make_signal: SignalFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """task-08 acceptance: mixed-provenance regression fixture.

    REAL + BACKFILLED + SYNTHETIC + one legacy event with NO provenance key at all. The purge
    must remove ONLY the SYNTHETIC signal and the legacy (fail-closed-to-SYNTHETIC) event,
    retaining REAL and BACKFILLED untouched. All four counts are asserted exactly.
    """
    store, read_model = await _seed(clock)
    real = make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL)
    backfilled = make_signal(source_type=SignalSource.CONTRACT, provenance=Provenance.BACKFILLED)
    synthetic = make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.SYNTHETIC)
    await record_signals(
        event_store=store,
        clock=clock,
        signals=[real, backfilled, synthetic],
        correlation_id="seed",
    )
    # A legacy event predating the `provenance` field: no such key in its payload at all. The
    # rest of the system treats this as SYNTHETIC (fail-closed default), so it must also be
    # purged — exercising `_is_synthetic_signal_event`'s payload-default path directly, since
    # `record_signals`/`signal_payload` always stamp a provenance key for a live `Signal`.
    legacy_payload: dict[str, object] = {
        "signal_id": "s-legacy",
        "created_at": clock.now().isoformat(),
        "source_type": SignalSource.DISCLOSED_TRADE.value,
        "ticker": "AAPL",
        "politician_id": None,
        "sentiment": 0.5,
        "politician_weight": 1.0,
        "source_reliability": 1.0,
        "raw_payload": {},
        # no "provenance" key
    }
    await store.append(
        DomainEvent(
            stream_id="signal-s-legacy",
            event_type=EventType.SIGNAL_RECEIVED,
            payload=legacy_payload,
            occurred_at=clock.now(),
            correlation_id="seed-legacy",
        )
    )
    _patch_build(monkeypatch, store, read_model)

    before = list(await read_model.all_signals())
    assert len(before) == 4  # sanity: all four rows landed in the read model pre-purge

    rc = await purge_synthetic.run_purge(confirm=True)

    assert rc == 0
    after = list(await read_model.all_signals())
    assert {s.id for s in after} == {real.id, backfilled.id}
    # All four counts, asserted exactly: 1 REAL, 1 BACKFILLED retained; 2 SYNTHETIC (explicit +
    # legacy fail-closed) purged.
    real_after = [s for s in after if s.provenance is Provenance.REAL]
    backfilled_after = [s for s in after if s.provenance is Provenance.BACKFILLED]
    assert len(real_after) == 1
    assert len(backfilled_after) == 1
    assert len(purge_eligible(before)) == 2  # explicit SYNTHETIC + legacy, pre-purge
    assert purge_eligible(after) == []  # nothing purge-eligible survives
    out = capsys.readouterr().out
    assert "purged 2 synthetic signal(s)" in out
    assert "read side now has 1 real, 1 backfilled (retained, untouched), 0 synthetic" in out


class _SyncEventStore:
    """Lock-free, loop-agnostic event store for the `main()` exit-code tests.

    The full async/projecting wiring is already exercised by the `run_purge` tests above (in a
    single loop). `main()` owns its OWN event loop via `asyncio.run`, so its tests stay
    synchronous; sharing a real `asyncio.Lock`-backed store across the seed loop and main's loop
    would bind the lock to two loops. These fakes hold no lock, so they are safe across loops.
    """

    def __init__(self, events: list[StoredEvent]) -> None:
        self._events = events

    async def read_all(self, limit: int | None = None) -> list[StoredEvent]:
        return list(self._events)


class _SyncReadModel:
    """Lock-free read model: `all_signals` reflects the last `rebuild` (or the seed)."""

    def __init__(self, events: list[StoredEvent]) -> None:
        self._signals = signals_from_events(events)

    async def rebuild(self, events: Sequence[StoredEvent]) -> None:
        self._signals = signals_from_events(events)

    async def all_signals(self) -> list[Signal]:
        return list(self._signals)


@dataclass
class _SyncDeps:
    """The deps surface the purge job reads, backed by the lock-free fakes above."""

    event_store: _SyncEventStore
    read_model: _SyncReadModel
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def _sync_deps(clock: FixedClock, make_signal: SignalFactory) -> _SyncDeps:
    real = [
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL),
        make_signal(source_type=SignalSource.CONTRACT, provenance=Provenance.REAL),
    ]
    fake = [
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.SYNTHETIC),
        make_signal(source_type=SignalSource.DISCLOSED_TRADE, provenance=Provenance.SYNTHETIC),
    ]
    events = [
        StoredEvent(
            event=DomainEvent(
                stream_id=f"signal-{s.id}",
                event_type=EventType.SIGNAL_RECEIVED,
                payload=signal_payload(s),
                occurred_at=clock.now(),
                correlation_id="seed",
            ),
            seq=1,
            recorded_at=clock.now(),
        )
        for s in [*real, *fake]
    ]
    return _SyncDeps(event_store=_SyncEventStore(events), read_model=_SyncReadModel(events))


def test_main_dry_run_returns_zero(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, make_signal: SignalFactory
) -> None:
    deps = _sync_deps(clock, make_signal)
    monkeypatch.setattr(purge_synthetic, "build_dependencies", lambda _settings: deps)

    # No --confirm: dry run, exit 0, read side unchanged (both synthetic signals remain).
    assert purge_synthetic.main([]) == 0
    assert len(purge_eligible(asyncio.run(deps.read_model.all_signals()))) == 2


def test_main_confirm_returns_zero_and_purges(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, make_signal: SignalFactory
) -> None:
    deps = _sync_deps(clock, make_signal)
    monkeypatch.setattr(purge_synthetic, "build_dependencies", lambda _settings: deps)

    assert purge_synthetic.main(["--confirm"]) == 0
    assert purge_eligible(asyncio.run(deps.read_model.all_signals())) == []


def test_main_returns_one_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_settings: object) -> object:
        raise RuntimeError("composition failed")

    monkeypatch.setattr(purge_synthetic, "build_dependencies", _boom)
    assert purge_synthetic.main([]) == 1
