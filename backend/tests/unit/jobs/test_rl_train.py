"""RL-train job CLI (task-rl-08), driven with a STUB trainer — no stable-baselines3 / torch needed.

The always-on seam layer: argument validation (>= 5 seeds, the required ``--train-end``), the
``--confirm`` real-data guard, the dry-run's "writes nothing" contract, the train split handed to
the trainer, and the single audit event. ``trading.rl.train`` imports SB3 lazily, so its
``train_ppo`` / ``require_rl_extra`` are monkeypatched here; the real trainer is exercised by the
``rl``-marked ``tests/unit/rl/test_train_smoke.py``.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import FixedClock
from trading.application.rl_dataset import RlBar, RlDataset, rl_dataset_to_json, session_known_at
from trading.config import Settings
from trading.domain.events import EventType
from trading.infrastructure.market_data.alpaca_history import AlpacaHistoricalData
from trading.infrastructure.market_data.historical import SyntheticHistoricalPrices
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.jobs import rl_train
from trading.rl import train as train_module
from trading.rl.bars import BarTable
from trading.rl.env import EnvConfig
from trading.rl.train import RlExtraMissingError, TrainConfig, TrainResult

_NY = ZoneInfo("America/New_York")
_NOW = datetime(2026, 7, 2, 23, 0, tzinfo=UTC)
_SETTINGS = Settings(
    _env_file=None,
    max_trade_pct_of_sleeve="10",
    max_position_concentration_pct="25",
    max_drawdown_pct="20",
)
_N_BARS = 420
_FIRST_SESSION = date(2020, 1, 2)
_SEEDS = "0,1,2,3,4"


def _sessions(n: int) -> list[datetime]:
    day, out = _FIRST_SESSION, list[datetime]()
    while len(out) < n:
        if day.weekday() < 5:
            out.append(datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC))
        day += timedelta(days=1)
    return out


def _write_dataset(path: Path, *, source_adapter: str) -> list[datetime]:
    sessions = _sessions(_N_BARS)
    bars = tuple(
        RlBar(session_ts=s, known_at=session_known_at(s), close=100.0 + 0.1 * i)
        for i, s in enumerate(sessions)
    )
    dataset = RlDataset(
        symbol="SPY",
        bars=bars,
        adjustment="all",
        source_adapter=source_adapter,
        start=sessions[0] - timedelta(days=1),
        end=_NOW,
        generated_at=_NOW,
    )
    path.write_bytes(rl_dataset_to_json(dataset))
    return sessions


@dataclass
class _FakeDeps:
    event_store: InMemoryEventStore
    clock: FixedClock
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


@dataclass
class _Call:
    bars: BarTable
    env_config: EnvConfig
    config: TrainConfig
    out_dir: Path
    provenance: dict[str, Any]


@dataclass
class _Harness:
    calls: list[_Call]
    deps: _FakeDeps
    builds: list[object]


def _patch(monkeypatch: pytest.MonkeyPatch, *, extra_missing: bool = False) -> _Harness:
    clock = FixedClock(_NOW)
    harness = _Harness(calls=[], deps=_FakeDeps(InMemoryEventStore(clock), clock), builds=[])

    def stub_train(
        bars: BarTable,
        env_config: EnvConfig,
        config: TrainConfig,
        *,
        out_dir: Path,
        provenance: dict[str, Any],
        run_date: date,
        overwrite: bool = False,
    ) -> TrainResult:
        harness.calls.append(_Call(bars, env_config, config, out_dir, provenance))
        stem = train_module.artifact_stem(
            bars.symbol, run_date, config.seed, untrained=config.untrained
        )
        model, manifest = out_dir / f"{stem}.zip", out_dir / f"{stem}.json"
        if model.exists() and not overwrite:
            raise AssertionError("the job's pre-check should have refused this run")
        out_dir.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"zip")
        manifest.write_text(json.dumps({"training": {"ep_rew_mean_first": 1.5}}))
        return TrainResult(
            seed=config.seed,
            model_path=model,
            final_ep_rew_mean=2.5 + config.seed,
            sb3_version="2.9.0",
            torch_version="2.14.1",
            gymnasium_version="1.4.0",
            numpy_version="2.5.4",
            manifest_path=manifest,
            num_timesteps=config.total_timesteps,
            wall_seconds=0.01,
            untrained=config.untrained,
            policy_sha256=f"{config.seed:064x}",
        )

    def require() -> None:
        if extra_missing:
            raise RlExtraMissingError("the optional [rl] extra is not installed (stub)")

    def fake_build(settings: Settings) -> _FakeDeps:
        harness.builds.append(settings)
        return harness.deps

    monkeypatch.setattr(train_module, "train_ppo", stub_train)
    monkeypatch.setattr(train_module, "require_rl_extra", require)
    monkeypatch.setattr(rl_train, "build_dependencies", fake_build)
    monkeypatch.setattr(rl_train, "Settings", lambda: _SETTINGS)
    monkeypatch.setattr(rl_train, "configure_logging", lambda _level: None)
    return harness


def _argv(dataset: Path, out: Path, *extra: str, seeds: str = _SEEDS) -> list[str]:
    return [
        "--dataset",
        str(dataset),
        "--seeds",
        seeds,
        "--train-end",
        "2021-05-03",
        "--timesteps",
        "200000",
        "--out",
        str(out),
        *extra,
    ]


# --- the spec's acceptance tests ----------------------------------------------------------------


@pytest.mark.parametrize(
    "source_adapter",
    # The synthetic provider, and an unknown adapter: the guard is an allowlist (fails closed).
    [SyntheticHistoricalPrices.__name__, "SomeVendorHistoricalBars"],
)
def test_confirm_requires_real_data_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    source_adapter: str,
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    _write_dataset(dataset, source_adapter=source_adapter)
    out = tmp_path / "artifacts"

    rc = rl_train.main(_argv(dataset, out, "--confirm"))

    assert rc == 1
    text = capsys.readouterr().out
    assert (
        "error: --confirm requires a dataset frozen from the REAL historical-bars adapter" in text
    )
    assert f"source_adapter={source_adapter!r}" in text
    assert harness.calls == []  # never trained
    assert harness.builds == []
    assert not out.exists()


def test_seeds_argument_requires_at_least_five(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    _write_dataset(dataset, source_adapter=AlpacaHistoricalData.__name__)

    rc = rl_train.main(_argv(dataset, tmp_path / "out", seeds="0"))

    assert rc == 1
    assert (
        "error: at least 5 training seeds are required; single-seed results are noise"
        in capsys.readouterr().out
    )
    assert harness.calls == []


def test_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    # A dry run is allowed even on a synthetic-source dataset.
    _write_dataset(dataset, source_adapter=SyntheticHistoricalPrices.__name__)
    out = tmp_path / "artifacts"
    before = sorted(p.name for p in tmp_path.iterdir())

    rc = rl_train.main(_argv(dataset, out))

    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing new under tmp_path
    assert not out.exists()
    assert [c.config.seed for c in harness.calls] == [0, 1, 2, 3, 4]
    assert {c.config.total_timesteps for c in harness.calls} == {rl_train.SMOKE_TIMESTEPS}
    for call in harness.calls:  # each smoke trained into a throwaway dir, now deleted
        assert out not in (call.out_dir, *call.out_dir.parents)
        assert not call.out_dir.exists()
    assert harness.builds == []  # no composition root, so no event store to append to
    text = capsys.readouterr().out
    assert "DRY-RUN: nothing was written" in text
    assert "passes over the train bars: 200000 / " in text
    assert "IN-SAMPLE" in text


# --- the confirmed run --------------------------------------------------------------------------


async def test_confirm_trains_every_seed_on_the_train_split_and_appends_one_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    sessions = _write_dataset(dataset, source_adapter=AlpacaHistoricalData.__name__)
    out = tmp_path / "artifacts"
    args = rl_train.parse_args(_argv(dataset, out, "--confirm"))

    rc = await rl_train.run_training(args)

    assert rc == 0
    split_at = rl_train.split_instant(date(2021, 5, 3))
    expected_bars = [s for s in sessions if session_known_at(s) < split_at]
    assert len(harness.calls) == 5
    for call in harness.calls:
        assert call.out_dir == out
        assert call.config.total_timesteps == 200_000
        assert [b.session_ts for b in call.bars.bars] == expected_bars  # train split only
        assert max(b.known_at for b in call.bars.bars) < split_at
        assert call.env_config.random_start is True
        assert call.env_config.episode_length == 252
        assert call.env_config.cost.one_way_bps == 1.0
        assert call.provenance["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert call.provenance["train_end"] == "2021-05-03"
    events = await harness.deps.event_store.read_all()
    assert len(events) == 1
    event = events[0].event
    assert event.event_type == EventType.USER_ACTION
    payload = event.payload
    assert payload["action"] == "rl.train"
    assert payload["symbol"] == "SPY"
    assert payload["seeds"] == [0, 1, 2, 3, 4]
    assert payload["timesteps"] == 200_000
    assert payload["untrained"] is False
    assert payload["final_ep_rew_mean"] == {str(s): 2.5 + s for s in range(5)}
    assert payload["policy_sha256"] == {str(s): f"{s:064x}" for s in range(5)}
    assert set(payload["versions"]) == {"sb3", "torch", "gymnasium", "numpy"}
    assert payload["n_train_bars"] == len(expected_bars)
    assert "wrote 5 model(s)" in capsys.readouterr().out


async def test_timesteps_zero_is_the_untrained_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    _write_dataset(dataset, source_adapter=AlpacaHistoricalData.__name__)
    argv = _argv(dataset, tmp_path / "out", "--confirm")
    argv[argv.index("200000")] = "0"

    rc = await rl_train.run_training(rl_train.parse_args(argv))

    assert rc == 0
    assert {c.config.total_timesteps for c in harness.calls} == {0}
    assert all(c.config.untrained for c in harness.calls)
    (stored,) = await harness.deps.event_store.read_all()
    assert stored.event.payload["untrained"] is True


async def test_a_repeat_confirm_refuses_to_overwrite_unless_asked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    _write_dataset(dataset, source_adapter=AlpacaHistoricalData.__name__)
    out = tmp_path / "artifacts"
    assert await rl_train.run_training(rl_train.parse_args(_argv(dataset, out, "--confirm"))) == 0
    written = {p.name: p.read_bytes() for p in out.iterdir()}
    assert len(written) == 10  # 5 seeds x (.zip + .json)
    capsys.readouterr()

    rc = await rl_train.run_training(rl_train.parse_args(_argv(dataset, out, "--confirm")))

    assert rc == 1  # refused BEFORE training anything
    assert "error: refusing to overwrite existing artifact(s)" in capsys.readouterr().out
    assert len(harness.calls) == 5
    assert {p.name: p.read_bytes() for p in out.iterdir()} == written
    assert len(await harness.deps.event_store.read_all()) == 1

    argv = _argv(dataset, out, "--confirm", "--overwrite")
    assert await rl_train.run_training(rl_train.parse_args(argv)) == 0
    assert len(harness.calls) == 10
    assert len(await harness.deps.event_store.read_all()) == 2


def test_missing_rl_extra_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    harness = _patch(monkeypatch, extra_missing=True)
    dataset = tmp_path / "spy.json"
    _write_dataset(dataset, source_adapter=AlpacaHistoricalData.__name__)

    rc = rl_train.main(_argv(dataset, tmp_path / "out"))

    assert rc == 1
    assert "[rl] extra is not installed" in capsys.readouterr().out
    assert harness.calls == []


def test_a_train_split_too_short_for_one_episode_fails_before_training(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    harness = _patch(monkeypatch)
    dataset = tmp_path / "spy.json"
    _write_dataset(dataset, source_adapter=AlpacaHistoricalData.__name__)
    argv = _argv(dataset, tmp_path / "out")
    argv[argv.index("2021-05-03")] = "2020-06-01"  # ~100 bars < 50 + 252

    rc = rl_train.main(argv)

    assert rc == 1
    assert "cannot hold a 252-step episode" in capsys.readouterr().out
    assert harness.calls == []


# --- argument validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("0,1,2,3", rl_train.SEEDS_ERROR),
        ("0,1,2,3,3", "must be distinct"),
        ("0,1,2,3,-4", "must be non-negative"),
        ("0,1,two,3,4", "comma-separated integers"),
    ],
)
def test_parse_seeds_rejects(raw: str, message: str) -> None:
    with pytest.raises(rl_train._UsageError, match=message):
        rl_train.parse_seeds(raw)


def test_parse_seeds_accepts_five_or_more() -> None:
    assert rl_train.parse_seeds(" 4, 3,2,1,0 ,9") == (4, 3, 2, 1, 0, 9)


def test_train_end_is_required_and_must_be_a_date(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    base = ["--dataset", "d.json", "--seeds", _SEEDS, "--out", str(tmp_path)]
    assert rl_train.main(base) == 1  # argparse: --train-end is required
    assert rl_train.main([*base, "--train-end", "2021/05/03"]) == 1
    assert "--train-end must be YYYY-MM-DD" in capsys.readouterr().out


def test_negative_timesteps_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--dataset", "d.json", "--seeds", _SEEDS, "--train-end", "2021-05-03", "--out", "o"]
    assert rl_train.main([*argv, "--timesteps", "-1"]) == 1
    assert "--timesteps must be >= 0" in capsys.readouterr().out


def test_split_instant_is_new_york_midnight_under_both_dst_regimes() -> None:
    assert rl_train.split_instant(date(2026, 1, 2)) == datetime(2026, 1, 2, 5, tzinfo=UTC)
    assert rl_train.split_instant(date(2026, 7, 2)) == datetime(2026, 7, 2, 4, tzinfo=UTC)


def test_real_source_allowlist_names_the_real_adapter_only() -> None:
    assert {AlpacaHistoricalData.__name__} == rl_train.REAL_SOURCE_ADAPTERS
    assert SyntheticHistoricalPrices.__name__ not in rl_train.REAL_SOURCE_ADAPTERS
