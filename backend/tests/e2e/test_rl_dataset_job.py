"""RL-dataset job end-to-end on the real composition root (in-memory store), stub bar source.

`build_dependencies` runs for real (DATABASE_URL and Alpaca credentials cleared => in-memory store,
synthetic historical provider). For the --confirm path the wired builder is swapped for one over a
stub `HistoricalPricesPort` emitting Alpaca-shaped opening-boundary stamps across both 2026 DST
switches, and `real_historical_prices` is set — exactly what the real adapter wiring provides.
No network.
"""

import dataclasses
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import FixedClock
from trading.application.rl_dataset import BuildRlDataset
from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.prices import PriceSeries
from trading.interface.app import AppDeps
from trading.jobs import rl_dataset as rl_job
from trading.main import build_dependencies

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "DATABASE_URL",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "REQUIRE_REAL_DATA",
    "ENABLE_LIVE_QUOTES",
    "ENABLE_HISTORICAL_DATA",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)
_NY = ZoneInfo("America/New_York")
_NOW = datetime(2026, 12, 1, 23, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rl_job, "Settings", lambda: Settings(_env_file=None))


class _AlpacaShapedStub:
    """HistoricalPricesPort stub: one bar per weekday in [start, end], stamped like Alpaca (00:00
    America/New_York, so 05:00Z in EST and 04:00Z in EDT), with a deterministic close path."""

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        points: list[tuple[datetime, Decimal]] = []
        day: date = start.astimezone(_NY).date()
        index = 0
        while True:
            stamp = datetime.combine(day, time(0, 0), tzinfo=_NY).astimezone(UTC)
            if stamp > end:
                break
            if stamp >= start and day.weekday() < 5:
                points.append((stamp, Decimal("400") + Decimal(index % 37) / Decimal("4")))
                index += 1
            day += timedelta(days=1)
        return PriceSeries(series={s: tuple(points) for s in symbols})


def _patch_real_like(monkeypatch: pytest.MonkeyPatch, *, adjustment: str) -> list[AppDeps]:
    built: list[AppDeps] = []

    def build(settings: Settings, **kwargs: Any) -> AppDeps:
        deps = build_dependencies(settings, **kwargs)
        deps = dataclasses.replace(
            deps,
            rl_dataset=BuildRlDataset(
                prices=_AlpacaShapedStub(), clock=FixedClock(_NOW), adjustment=adjustment
            ),
            real_historical_prices=True,
        )
        built.append(deps)
        return deps

    monkeypatch.setattr(rl_job, "build_dependencies", build)
    return built


async def test_output_is_byte_identical_across_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _patch_real_like(monkeypatch, adjustment="all")
    first, second = tmp_path / "a" / "spy.json", tmp_path / "b" / "spy.json"

    for out in (first, second):
        rc = await rl_job.run_dataset(
            confirm=True, symbol="SPY", years=1, out=out, adjustment="all"
        )
        assert rc == 0

    assert first.read_bytes() == second.read_bytes()
    header = json.loads(first.read_bytes())["header"]
    assert header["source_adapter"] == "_AlpacaShapedStub"
    assert header["adjustment"] == "all"
    assert header["n_bars"] > 250  # ~1 year of weekdays

    # Each --confirm run appended exactly ONE rl.dataset audit event to its own store.
    for deps in built:
        events = await deps.event_store.read_all()
        audits = [e.event for e in events if e.event.event_type == EventType.USER_ACTION]
        assert [a.payload["action"] for a in audits] == ["rl.dataset"]


async def test_every_timestamp_is_tz_aware_utc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_real_like(monkeypatch, adjustment="raw")
    out = tmp_path / "spy.json"
    assert await rl_job.run_dataset(confirm=True, symbol="SPY", years=1, out=out) == 0
    doc = json.loads(out.read_bytes())

    stamps = [doc["header"][k] for k in ("start", "end", "generated_at")]
    for bar in doc["bars"]:
        stamps += [bar["session_ts"], bar["known_at"]]
    for text in stamps:
        assert text.endswith("Z"), text
        parsed = datetime.fromisoformat(text)
        assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), text

    # The window spans both 2026 DST switches: EDT and EST bars each carry the PIT known_at.
    pairs = {(b["session_ts"][11:], b["known_at"][11:]) for b in doc["bars"]}
    assert pairs == {("04:00:00Z", "20:15:00Z"), ("05:00:00Z", "21:15:00Z")}
    for bar in doc["bars"]:
        assert bar["known_at"][:10] == bar["session_ts"][:10]  # same session date, after close


async def test_dry_run_against_synthetic_provider_fails_closed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Real wiring, synthetic provider: its daily grid is not session-stamped across a DST switch,
    # so the session-boundary check refuses it rather than inventing a knowledge time.
    out = tmp_path / "spy.json"
    rc = await rl_job.run_dataset(confirm=False, symbol="SPY", years=1, out=out)
    assert rc == 1
    assert "00:00 session-open boundary" in capsys.readouterr().out
    assert not out.exists()
