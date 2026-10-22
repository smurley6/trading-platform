"""The RL-export job end to end on a real (briefly trained) PPO model (task-rl-09). ``rl``-marked;
skipped without the ``[rl]`` extra. The SB3-free guards are covered by
``tests/unit/jobs/test_rl_export.py``.

Pins the guardrails: the parity gate blocks the write even under ``--confirm``, a dry run writes
nothing, ``--confirm`` refuses to overwrite, and the manifest / dataset must belong to the model.
"""

import hashlib
import json
import math
import random
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import FixedClock
from trading.application.rl_dataset import (
    RlBar,
    RlDataset,
    rl_dataset_from_json,
    rl_dataset_to_json,
    session_known_at,
)
from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.rl_features import FEATURE_NAMES, WARMUP_BARS, FeatureScales, FeatureSpec
from trading.domain.rl_policy import ENV_ACTION_ORDER
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.rl.policy_store import PolicyArtifactError, load_policy
from trading.jobs import rl_export as job
from trading.rl import export as export_module
from trading.rl.bars import BarTable
from trading.rl.env import EnvConfig
from trading.rl.export import ParityError, ParityReport
from trading.rl.train import TrainConfig, train_ppo

pytestmark = pytest.mark.rl
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

_NY = ZoneInfo("America/New_York")
_NOW = datetime(2026, 10, 22, 23, 0, tzinfo=UTC)
_SETTINGS = Settings(
    _env_file=None,
    max_trade_pct_of_sleeve="10",
    max_position_concentration_pct="25",
    max_drawdown_pct="20",
)
_N_BARS = 600
_N_TRAIN = 320
_REAL = "AlpacaHistoricalData"


def _dataset_bytes(*, source_adapter: str = _REAL, seed: int = 21) -> bytes:
    day, sessions = date(2019, 1, 2), list[datetime]()
    while len(sessions) < _N_BARS:
        if day.weekday() < 5:
            sessions.append(datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC))
        day += timedelta(days=1)
    rng, close, bars = random.Random(seed), 100.0, list[RlBar]()
    for s in sessions:
        close *= math.exp(rng.gauss(0.0003, 0.01))
        bars.append(RlBar(session_ts=s, known_at=session_known_at(s), close=close))
    return rl_dataset_to_json(
        RlDataset(
            symbol="SPY",
            bars=tuple(bars),
            adjustment="all",
            source_adapter=source_adapter,
            start=sessions[0] - timedelta(days=1),
            end=_NOW,
            generated_at=_NOW,
        )
    )


@dataclass(frozen=True)
class _Trained:
    root: Path
    dataset: Path
    model: Path


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> _Trained:
    root = tmp_path_factory.mktemp("rl_export_job")
    data = _dataset_bytes()
    dataset = root / "spy_daily.json"
    dataset.write_bytes(data)
    table = BarTable.from_dataset(rl_dataset_from_json(data))
    train = BarTable(symbol=table.symbol, bars=table.bars[:_N_TRAIN])
    result = train_ppo(
        train,
        EnvConfig(),
        TrainConfig(total_timesteps=1_024, seed=0),
        out_dir=root / "models",
        provenance={"dataset_sha256": hashlib.sha256(data).hexdigest()},
        run_date=date(2026, 10, 22),
    )
    return _Trained(root=root, dataset=dataset, model=result.model_path)


@dataclass
class _FakeDeps:
    event_store: InMemoryEventStore
    clock: FixedClock
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def _patch(monkeypatch: pytest.MonkeyPatch) -> _FakeDeps:
    clock = FixedClock(_NOW)
    deps = _FakeDeps(InMemoryEventStore(clock), clock)
    monkeypatch.setattr(job, "build_dependencies", lambda _settings: deps)
    monkeypatch.setattr(job, "Settings", lambda: _SETTINGS)
    monkeypatch.setattr(job, "configure_logging", lambda _level: None)
    return deps


def _copy_model(trained: _Trained, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    model = into / trained.model.name
    shutil.copy(trained.model, model)
    shutil.copy(trained.model.with_suffix(".json"), model.with_suffix(".json"))
    return model


def _argv(model: Path, dataset: Path, out: Path, *flags: str) -> list[str]:
    return ["--model", str(model), "--dataset", str(dataset), "--out", str(out), *flags]


async def _events(deps: _FakeDeps) -> list[Any]:
    return list(await deps.event_store.read_all())


async def test_dry_run_exports_with_parity_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    deps = _patch(monkeypatch)
    out = tmp_path / "policy.json"

    assert await job.run_export(job.parse_args(_argv(trained.model, trained.dataset, out))) == 0

    text = capsys.readouterr().out
    assert "DRY-RUN" in text
    assert "0 disagreements" in text
    assert "policy_id: spy-ppo-20261022-" in text
    assert not out.exists()
    assert await _events(deps) == []


async def test_confirm_writes_the_artifact_and_appends_one_event(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    deps = _patch(monkeypatch)
    out = tmp_path / "out" / "spy_ppo_v1.json"

    args = job.parse_args(_argv(trained.model, trained.dataset, out, "--confirm"))
    assert await job.run_export(args) == 0

    data = out.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert f"sha256 {sha}" in capsys.readouterr().out
    policy = load_policy(out)
    manifest = json.loads(trained.model.with_suffix(".json").read_text())
    assert policy.policy_id == f"spy-ppo-20261022-{manifest['policy_sha256'][:8]}"
    assert policy.actions == ENV_ACTION_ORDER

    document = json.loads(data)
    assert document["features"]["names"][0] == "r1"
    assert document["features"]["window"] == 50
    assert set(document["features"]["scales"]) == {
        "daily_vol",
        "sma10_gap",
        "sma50_gap",
        "trend_gap",
        "unrealized",
    }
    meta = document["meta"]
    assert meta["training"]["run_id"] == "spy_ppo_20261022_seed0"
    assert meta["training"]["seed"] == 0
    assert meta["training"]["policy_sha256"] == manifest["policy_sha256"]
    assert meta["dataset"]["sha256"] == hashlib.sha256(trained.dataset.read_bytes()).hexdigest()
    assert meta["dataset"]["adjustment"] == "all"
    assert meta["dataset"]["first_session"] == "2019-01-02"
    assert meta["dataset"]["n_bars"] == _N_BARS
    splits = meta["splits"]
    assert splits["train"]["bars"] == [0, _N_TRAIN]
    assert splits["val"]["bars"] == [_N_TRAIN, _N_TRAIN + 126]
    assert splits["purge"]["bars"] == [_N_TRAIN + 126, _N_TRAIN + 176]
    assert splits["test"]["bars"] == [_N_TRAIN + 176, _N_TRAIN + 239]
    assert splits["derived"] == ["val", "purge", "test"]
    assert meta["training"]["feature_scales_source"] == "manifest"
    assert meta["parity"]["logit_tol"] == 1e-3
    for versions in (meta["training"]["library_versions"], meta["export"]["library_versions"]):
        assert set(versions) == {"sb3", "torch", "gymnasium", "numpy"}
        assert "absent" not in versions.values()
    assert meta["parity"]["n_disagreements"] == 0
    assert meta["parity"]["n_observations"] == 1_000 + 2 * 126

    events = await _events(deps)
    assert len(events) == 1
    assert events[0].event.event_type == EventType.USER_ACTION
    payload = events[0].event.payload
    assert payload["action"] == "rl.export"
    assert payload["artifact_sha256"] == sha
    assert payload["policy_id"] == policy.policy_id
    assert payload["parity"]["n_disagreements"] == 0


async def test_parity_failure_writes_nothing_even_with_confirm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    deps = _patch(monkeypatch)
    bad = ParityReport(
        n_observations=10,
        n_agree=9,
        n_ties=0,
        n_disagreements=1,
        tie_margin=1e-4,
        max_abs_logit_diff=0.5,
        logit_tol=1e-3,
        disagreement_indices=(3,),
    )

    def failing(model: Any, policy: Any, observations: Any) -> ParityReport:
        raise ParityError(bad)

    monkeypatch.setattr(export_module, "assert_parity", failing)
    out = tmp_path / "policy.json"

    args = job.parse_args(_argv(trained.model, trained.dataset, out, "--confirm"))
    assert await job.run_export(args) == 1

    assert "PARITY FAILED" in capsys.readouterr().out
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp").exists()
    assert await _events(deps) == []


async def test_confirm_refuses_to_overwrite_unless_asked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    deps = _patch(monkeypatch)
    out = tmp_path / "policy.json"
    out.write_text("previous")

    args = job.parse_args(_argv(trained.model, trained.dataset, out, "--confirm"))
    assert await job.run_export(args) == 1
    assert "refusing to overwrite" in capsys.readouterr().out
    assert out.read_text() == "previous"

    args = job.parse_args(_argv(trained.model, trained.dataset, out, "--confirm", "--overwrite"))
    assert await job.run_export(args) == 0
    load_policy(out)
    assert len(await _events(deps)) == 1


async def test_dataset_must_be_the_one_the_model_trained_on(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    _patch(monkeypatch)
    other = tmp_path / "other.json"
    other.write_bytes(_dataset_bytes(seed=99))

    args = job.parse_args(_argv(trained.model, other, tmp_path / "p.json", "--confirm"))
    assert await job.run_export(args) == 1

    assert "is not the dataset this model trained on" in capsys.readouterr().out
    assert not (tmp_path / "p.json").exists()


async def test_manifest_must_describe_these_weights(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    _patch(monkeypatch)
    model = _copy_model(trained, tmp_path / "models")
    manifest_path = model.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    args = job.parse_args(_argv(model, trained.dataset, tmp_path / "p.json", "--confirm"))
    assert await job.run_export(args) == 1

    assert "does not match its manifest's policy_sha256" in capsys.readouterr().out
    assert not (tmp_path / "p.json").exists()


async def test_confirm_requires_a_real_data_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    _patch(monkeypatch)
    synthetic = tmp_path / "synthetic.json"
    synthetic.write_bytes(_dataset_bytes(source_adapter="SyntheticHistoricalPrices"))
    model = _copy_model(trained, tmp_path / "models")
    manifest_path = model.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["dataset_sha256"] = hashlib.sha256(synthetic.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    args = job.parse_args(_argv(model, synthetic, tmp_path / "p.json", "--confirm"))
    assert await job.run_export(args) == 1
    assert "REAL historical-bars adapter" in capsys.readouterr().out

    # A dry run on the same inputs is always allowed.
    assert await job.run_export(job.parse_args(_argv(model, synthetic, tmp_path / "p.json"))) == 0
    assert not (tmp_path / "p.json").exists()


async def test_non_default_feature_scales_are_carried_through_faithfully(
    monkeypatch: pytest.MonkeyPatch, trained: _Trained, tmp_path: Path
) -> None:
    # Review finding: train_ppo accepts any EnvConfig.spec (rl-10's per-fold harness calls it
    # directly). The manifest must record the scales and the export must use them — parity alone
    # cannot tell (both sides would see the same observations).
    _patch(monkeypatch)
    data = trained.dataset.read_bytes()
    table = BarTable.from_dataset(rl_dataset_from_json(data))
    scales = FeatureScales(daily_vol=0.02, sma10_gap=0.03, sma50_gap=0.07, trend_gap=0.04)
    spec = FeatureSpec(window=WARMUP_BARS, names=FEATURE_NAMES, scales=scales)
    result = train_ppo(
        BarTable(symbol=table.symbol, bars=table.bars[:_N_TRAIN]),
        EnvConfig(spec=spec),
        TrainConfig(total_timesteps=0, seed=1),
        out_dir=tmp_path / "models",
        provenance={"dataset_sha256": hashlib.sha256(data).hexdigest()},
        run_date=date(2026, 10, 22),
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["env"]["feature_scales"] == spec.to_dict()["scales"]
    out = tmp_path / "policy.json"

    args = job.parse_args(_argv(result.model_path, trained.dataset, out, "--confirm"))
    assert await job.run_export(args) == 0

    policy = load_policy(out)
    assert policy.features == spec
    assert policy.policy_id.startswith("spy-ppo-untrained-20261022-")
    meta = json.loads(out.read_bytes())["meta"]
    assert meta["training"]["feature_scales_source"] == "manifest"
    assert meta["training"]["untrained"] is True


async def test_a_scale_less_manifest_is_never_silently_default_stamped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    _patch(monkeypatch)
    model = _copy_model(trained, tmp_path / "models")
    manifest_path = model.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    del manifest["env"]["feature_scales"]
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "policy.json"

    # Not written by trading.jobs.rl_train (a direct train_ppo call could have used any spec).
    args = job.parse_args(_argv(model, trained.dataset, out, "--confirm"))
    assert await job.run_export(args) == 1
    assert "refusing to assume the default scales" in capsys.readouterr().out
    assert not out.exists()

    # A pre-rl-09 manifest from the rl_train job (which always trained on the default spec) is
    # accepted, and labeled as assumed.
    manifest["provenance"]["job"] = "trading.jobs.rl_train"
    manifest_path.write_text(json.dumps(manifest))
    assert await job.run_export(args) == 0
    meta = json.loads(out.read_bytes())["meta"]
    assert meta["training"]["feature_scales_source"] == "default_assumed"
    assert load_policy(out).features.scales == FeatureScales()


async def test_a_failed_read_back_leaves_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trained: _Trained,
    tmp_path: Path,
) -> None:
    deps = _patch(monkeypatch)

    def corrupt(path: Path) -> Any:
        raise PolicyArtifactError("simulated corrupt read-back")

    monkeypatch.setattr(job, "load_policy", corrupt)
    out = tmp_path / "policy.json"

    args = job.parse_args(_argv(trained.model, trained.dataset, out, "--confirm"))
    assert await job.run_export(args) == 1

    assert "did not read back" in capsys.readouterr().out
    assert not out.exists()
    assert await _events(deps) == []
