"""Backfill job CLI, driven against in-memory wiring (never a real DB or network).

`build_dependencies` is monkeypatched to return a lightweight deps backed by the same in-memory
event store + dedupe store production uses, plus a fake historical source. Asserts the dry-run
default writes nothing, `--confirm` records + emits a user.action event, the confirm-gate and
lookback guard refuse unsafe runs, and an unknown source fails cleanly.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from tests.conftest import FixedClock
from trading.application.ports.backfill_seen import BackfillSeenPort
from trading.application.ports.data_source import HistoricalDataSourcePort
from trading.application.ports.event_store import EventStorePort
from trading.config import Settings
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_backfill_seen import InMemoryBackfillSeen
from trading.jobs import backfill

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "DATABASE_URL",
    "BACKFILL_ENABLED",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "REQUIRE_REAL_DATA",
    "ENABLE_LIVE_QUOTES",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)

_DISCLOSED = datetime(2024, 3, 1, 20, 30, tzinfo=UTC)
_START = "2024-01-01"
_END = "2024-06-01"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)


def _hist_signal(*, source_uid: str) -> Signal:
    return Signal(
        id=f"sig-{source_uid}",
        created_at=_DISCLOSED,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker="AAPL",
        politician_id=None,
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.75,
        raw_payload={"source_uid": source_uid, "source": "fake"},
        provenance=Provenance.REAL,
    )


class _FakeHistoricalSource:
    def __init__(self, name: str, batch: Sequence[Signal]) -> None:
        self._name = name
        self._batch = list(batch)

    @property
    def name(self) -> str:
        return self._name

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        return list(self._batch)


@dataclass
class _FakeDeps:
    """The AppDeps surface the backfill job reads."""

    event_store: EventStorePort
    clock: FixedClock
    backfill_seen: BackfillSeenPort
    historical_sources: dict[str, HistoricalDataSourcePort]
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clock: FixedClock,
    store: EventStorePort,
    seen: BackfillSeenPort,
    sources: dict[str, HistoricalDataSourcePort],
    settings: Settings | None = None,
) -> None:
    deps = _FakeDeps(event_store=store, clock=clock, backfill_seen=seen, historical_sources=sources)
    monkeypatch.setattr(backfill, "build_dependencies", lambda _settings: deps)
    resolved = settings or Settings(_env_file=None)
    monkeypatch.setattr(backfill, "Settings", lambda: resolved)


def _wiring(
    clock: FixedClock, source: _FakeHistoricalSource
) -> tuple[InMemoryEventStore, InMemoryBackfillSeen, dict[str, HistoricalDataSourcePort]]:
    return InMemoryEventStore(clock), InMemoryBackfillSeen(), {source.name: source}


async def _signal_events(store: EventStorePort) -> int:
    return len([e for e in await store.read_all() if e.event.event_type == "signal.received"])


async def _user_actions(store: EventStorePort) -> list[dict[str, object]]:
    return [e.event.payload for e in await store.read_all() if e.event.event_type == "user.action"]


async def test_dry_run_default_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _FakeHistoricalSource("sec_form4", [_hist_signal(source_uid="u1")])
    store, seen, sources = _wiring(clock, source)
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources)

    rc = await backfill.run_backfill(
        source_id="sec_form4",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, tzinfo=UTC),
        confirm=False,
    )

    assert rc == 0
    assert await _signal_events(store) == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "new:       1" in out


async def test_confirm_requires_backfill_enabled(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _FakeHistoricalSource("sec_form4", [_hist_signal(source_uid="u1")])
    store, seen, sources = _wiring(clock, source)
    # BACKFILL_ENABLED is cleared -> false: a --confirm run must be refused before writing.
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources)

    rc = await backfill.run_backfill(
        source_id="sec_form4",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, tzinfo=UTC),
        confirm=True,
    )

    assert rc == 1
    assert await _signal_events(store) == 0
    assert "BACKFILL_ENABLED=true" in capsys.readouterr().out


async def test_confirm_records_events_and_user_action(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _FakeHistoricalSource(
        "sec_form4", [_hist_signal(source_uid="u1"), _hist_signal(source_uid="u2")]
    )
    store, seen, sources = _wiring(clock, source)
    # Risk caps come from the env (autouse fixture); only the backfill flag is overridden.
    settings = Settings(_env_file=None, backfill_enabled=True)
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources, settings=settings)

    rc = await backfill.run_backfill(
        source_id="sec_form4",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, tzinfo=UTC),
        confirm=True,
    )

    assert rc == 0
    assert await _signal_events(store) == 2
    actions = await _user_actions(store)
    assert len(actions) == 1
    action = actions[0]
    assert action["action"] == "backfill"
    assert action["source"] == "sec_form4"
    assert action["counts"] == {"fetched": 2, "new": 2, "duplicate": 0, "refused": 0}
    assert "recorded 2 new backfilled signal(s)" in capsys.readouterr().out


async def test_confirm_lookback_guard_refuses_overlong_window(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _FakeHistoricalSource("sec_form4", [_hist_signal(source_uid="u1")])
    store, seen, sources = _wiring(clock, source)
    # 30-day cap; the requested window is a full year -> refused before writing.
    settings = Settings(
        _env_file=None,
        backfill_enabled=True,
        backfill_sec_form4_max_lookback_days=30,
    )
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources, settings=settings)

    rc = await backfill.run_backfill(
        source_id="sec_form4",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC),
        confirm=True,
    )

    assert rc == 1
    assert await _signal_events(store) == 0
    assert "exceeds the max-lookback guard" in capsys.readouterr().out


async def test_unknown_source_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    # Empty registry (the current state): any --source is unknown and lists supported ones.
    store, seen = InMemoryEventStore(clock), InMemoryBackfillSeen()
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources={})

    rc = await backfill.run_backfill(
        source_id="sec_form4",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, tzinfo=UTC),
        confirm=False,
    )

    assert rc == 1
    assert "unknown/unsupported --source 'sec_form4'" in capsys.readouterr().out


async def test_empty_window_is_rejected(
    monkeypatch: pytest.MonkeyPatch, clock: FixedClock, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _FakeHistoricalSource("sec_form4", [_hist_signal(source_uid="u1")])
    store, seen, sources = _wiring(clock, source)
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources)

    rc = await backfill.run_backfill(
        source_id="sec_form4",
        start=datetime(2024, 6, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),  # end < start
        confirm=False,
    )
    assert rc == 1
    assert "must be after --start" in capsys.readouterr().out


def test_main_dry_run_returns_zero(monkeypatch: pytest.MonkeyPatch, clock: FixedClock) -> None:
    source = _FakeHistoricalSource("sec_form4", [_hist_signal(source_uid="u1")])
    store, seen, sources = _wiring(clock, source)
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources)

    rc = backfill.main(["--source", "sec_form4", "--start", _START, "--end", _END])
    assert rc == 0


def test_main_missing_required_arg_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # argparse exits 2 on a usage error; the entrypoint normalizes it to 1.
    rc = backfill.main(["--start", _START, "--end", _END])
    assert rc == 1


def test_main_bad_date_returns_one(monkeypatch: pytest.MonkeyPatch, clock: FixedClock) -> None:
    source = _FakeHistoricalSource("sec_form4", [])
    store, seen, sources = _wiring(clock, source)
    _patch(monkeypatch, clock=clock, store=store, seen=seen, sources=sources)
    rc = backfill.main(["--source", "sec_form4", "--start", "not-a-date", "--end", _END])
    assert rc == 1
