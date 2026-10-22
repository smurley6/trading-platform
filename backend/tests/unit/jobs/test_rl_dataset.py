"""RL-dataset job CLI, driven against a fake builder (never a real DB or network).

`build_dependencies` is monkeypatched to return a lightweight deps carrying a fake builder, an
in-memory event store and a `real_historical_prices` flag. Asserts `--confirm` requires the REAL
historical adapter, the dry-run default writes nothing, a confirmed run writes the file + exactly
one audit event, and main() maps outcomes to exit codes.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FixedClock
from trading.application.rl_dataset import RlBar, RlDataset, RlDatasetError, session_known_at
from trading.config import Settings
from trading.domain.events import EventType
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.jobs import rl_dataset

_NOW = datetime(2026, 7, 2, 23, 0, tzinfo=UTC)
# Risk caps are required for Settings() to construct; a real (offline) Settings gives log_level.
_SETTINGS = Settings(
    _env_file=None,
    max_trade_pct_of_sleeve="10",
    max_position_concentration_pct="25",
    max_drawdown_pct="20",
)


def _bar(session_ts: datetime, close: float) -> RlBar:
    return RlBar(session_ts=session_ts, known_at=session_known_at(session_ts), close=close)


class _FakeBuilder:
    def __init__(self, *, adjustment: str = "raw", error: str | None = None) -> None:
        self.adjustment = adjustment
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def execute(self, *, symbol: str, lookback_days: int) -> RlDataset:
        self.calls.append((symbol, lookback_days))
        if self.error is not None:
            raise RlDatasetError(self.error)
        return RlDataset(
            symbol=symbol,
            bars=(
                _bar(datetime(2026, 6, 30, 4, tzinfo=UTC), 600.1),
                _bar(datetime(2026, 7, 1, 4, tzinfo=UTC), 601.25),
                _bar(datetime(2026, 7, 2, 4, tzinfo=UTC), 602.5),
            ),
            adjustment=self.adjustment,
            source_adapter="FakeAdapter",
            start=datetime(2018, 7, 2, 4, tzinfo=UTC),
            end=_NOW,
            generated_at=_NOW,
        )


@dataclass
class _FakeDeps:
    """The AppDeps surface the rl-dataset job reads."""

    rl_dataset: _FakeBuilder
    real_historical_prices: bool
    event_store: InMemoryEventStore
    clock: FixedClock
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def _patch(
    monkeypatch: pytest.MonkeyPatch, *, builder: _FakeBuilder, real_prices: bool
) -> tuple[_FakeDeps, list[dict[str, Any]]]:
    clock = FixedClock(_NOW)
    deps = _FakeDeps(
        rl_dataset=builder,
        real_historical_prices=real_prices,
        event_store=InMemoryEventStore(clock),
        clock=clock,
    )
    build_kwargs: list[dict[str, Any]] = []

    def fake_build(_settings: Settings, **kwargs: Any) -> _FakeDeps:
        build_kwargs.append(kwargs)
        return deps

    monkeypatch.setattr(rl_dataset, "build_dependencies", fake_build)
    monkeypatch.setattr(rl_dataset, "Settings", lambda: _SETTINGS)
    monkeypatch.setattr(rl_dataset, "configure_logging", lambda _level: None)
    return deps, build_kwargs


async def test_confirm_requires_real_historical_prices(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    builder = _FakeBuilder()
    deps, _ = _patch(monkeypatch, builder=builder, real_prices=False)  # synthetic provider
    out = tmp_path / "spy.json"

    rc = await rl_dataset.run_dataset(confirm=True, symbol="SPY", years=8, out=out)

    assert rc == 1
    assert "error: --confirm requires the REAL historical-bars adapter" in capsys.readouterr().out
    assert builder.calls == []  # never fetched, never froze synthetic bars
    assert not out.exists()
    assert await deps.event_store.read_all() == []


async def test_dry_run_writes_nothing_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    builder = _FakeBuilder()
    deps, build_kwargs = _patch(monkeypatch, builder=builder, real_prices=False)
    out = tmp_path / "data" / "spy.json"

    rc = await rl_dataset.run_dataset(confirm=False, symbol="SPY", years=8, out=out)

    assert rc == 0
    assert builder.calls == [("SPY", 2922)]  # round(8 * 365.25)
    assert build_kwargs == [{"historical_adjustment": "raw"}]
    text = capsys.readouterr().out
    assert "DRY-RUN" in text
    assert "bars:            3" in text
    assert "first session:   2026-06-30" in text
    assert "last session:    2026-07-02" in text
    assert not out.exists() and not out.parent.exists()
    assert await deps.event_store.read_all() == []


async def test_confirm_writes_file_and_exactly_one_audit_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    builder = _FakeBuilder(adjustment="all")
    deps, build_kwargs = _patch(monkeypatch, builder=builder, real_prices=True)
    out = tmp_path / "data" / "rl" / "spy_daily.json"

    rc = await rl_dataset.run_dataset(
        confirm=True, symbol="SPY", years=8, out=out, adjustment="all"
    )

    assert rc == 0
    assert build_kwargs == [{"historical_adjustment": "all"}]
    data = out.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    header = json.loads(data)["header"]
    assert (header["adjustment"], header["source_adapter"], header["n_bars"]) == (
        "all",
        "FakeAdapter",
        3,
    )
    assert not (out.parent / "spy_daily.json.tmp").exists()

    events = await deps.event_store.read_all()
    assert len(events) == 1
    event = events[0].event
    assert (event.stream_id, event.event_type, event.occurred_at) == (
        "user-actions",
        EventType.USER_ACTION,
        _NOW,
    )
    assert event.payload == {
        "action": "rl.dataset",
        "symbol": "SPY",
        "years": 8,
        "adjustment": "all",
        "n_bars": 3,
        "sha256": digest,
    }
    assert digest in capsys.readouterr().out


async def test_adjustment_mismatch_with_wired_builder_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    builder = _FakeBuilder(adjustment="raw")
    _patch(monkeypatch, builder=builder, real_prices=True)
    out = tmp_path / "spy.json"

    rc = await rl_dataset.run_dataset(
        confirm=True, symbol="SPY", years=8, out=out, adjustment="all"
    )

    assert rc == 1
    assert "implicit adjustment" in capsys.readouterr().out
    assert builder.calls == [] and not out.exists()


async def test_pit_validation_failure_returns_one_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    builder = _FakeBuilder(error="not the 00:00 session-open boundary")
    deps, _ = _patch(monkeypatch, builder=builder, real_prices=True)
    out = tmp_path / "spy.json"

    rc = await rl_dataset.run_dataset(confirm=True, symbol="SPY", years=8, out=out)

    assert rc == 1
    assert "error: not the 00:00 session-open boundary" in capsys.readouterr().out
    assert not out.exists()
    assert await deps.event_store.read_all() == []


def test_main_dry_run_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    builder = _FakeBuilder(adjustment="all")
    _patch(monkeypatch, builder=builder, real_prices=False)
    argv = ["--symbol", "spy", "--years", "2", "--adjustment", "all", "--out", str(tmp_path / "x")]
    assert rl_dataset.main(argv) == 0
    assert builder.calls == [("spy", 730)]


def test_main_confirm_without_real_adapter_returns_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch(monkeypatch, builder=_FakeBuilder(), real_prices=False)
    assert rl_dataset.main(["--out", str(tmp_path / "x"), "--confirm"]) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--bogus"],
        [],  # --out is required
        ["--out", "x", "--adjustment", "split"],  # only raw|all are recorded modes
        ["--out", "x", "--years", "0"],
    ],
)
def test_main_usage_errors_return_one(argv: list[str]) -> None:
    # argparse exits 2 on a usage error; the entrypoint normalizes it to 1.
    assert rl_dataset.main(argv) == 1
