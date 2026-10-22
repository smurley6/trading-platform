"""RL-export job seams (task-rl-09) that need NO stable-baselines3 / torch: policy-id naming, the
manifest checks, the train / val / purge / test bounds, and the guards that run before any model is
loaded. The real export + parity gate is ``tests/unit/rl/test_export_job.py`` (``rl``-marked).
"""

import hashlib
import itertools
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from trading.application.rl_dataset import (
    RlBar,
    RlDataset,
    rl_dataset_from_json,
    rl_dataset_to_json,
    session_known_at,
)
from trading.config import Settings
from trading.domain.rl_features import FEATURE_NAMES, FeatureScales
from trading.jobs import rl_export as job
from trading.rl import train as train_module
from trading.rl.bars import BarTable
from trading.rl.train import RlExtraMissingError

_NY = ZoneInfo("America/New_York")
_NOW = datetime(2026, 10, 22, 23, 0, tzinfo=UTC)
_SETTINGS = Settings(
    _env_file=None,
    max_trade_pct_of_sleeve="10",
    max_position_concentration_pct="25",
    max_drawdown_pct="20",
)
_SHA = "d3888e7c9a261ec2dbf8cad207a0a879283b88a6a5e983049e230d4214c435c5"
_STEM = "spy_ppo_20261022_seed0"
_SCALES = {
    "daily_vol": 0.01,
    "sma10_gap": 0.02,
    "sma50_gap": 0.05,
    "trend_gap": 0.03,
    "unrealized": 0.05,
}


def _dataset(n: int = 300) -> bytes:
    day, sessions = date(2019, 1, 2), list[datetime]()
    while len(sessions) < n:
        if day.weekday() < 5:
            sessions.append(datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC))
        day += timedelta(days=1)
    bars = tuple(
        RlBar(session_ts=s, known_at=session_known_at(s), close=100.0 + 0.1 * i)
        for i, s in enumerate(sessions)
    )
    return rl_dataset_to_json(
        RlDataset(
            symbol="SPY",
            bars=bars,
            adjustment="all",
            source_adapter="AlpacaHistoricalData",
            start=sessions[0] - timedelta(days=1),
            end=_NOW,
            generated_at=_NOW,
        )
    )


def _table(data: bytes) -> BarTable:
    return BarTable.from_dataset(rl_dataset_from_json(data))


def _iso(table: BarTable, i: int) -> str:
    return table.bars[i].session_ts.isoformat().replace("+00:00", "Z")


def _manifest(table: BarTable, data: bytes, *, n_train: int = 40) -> dict[str, Any]:
    return {
        "kind": "rl.train.ppo",
        "schema_version": 1,
        "symbol": "SPY",
        "model_file": f"{_STEM}.zip",
        "policy_sha256": _SHA,
        "untrained": False,
        "env": _env(),
        "train_bars": {
            "n_bars": n_train,
            "first_session_ts": _iso(table, 0),
            "last_session_ts": _iso(table, n_train - 1),
        },
        "provenance": {"dataset_sha256": hashlib.sha256(data).hexdigest()},
    }


def test_policy_id_is_symbol_ppo_date_sha8() -> None:
    assert (
        job.make_policy_id("SPY", "20261022", _SHA, untrained=False) == "spy-ppo-20261022-d3888e7c"
    )
    assert (
        job.make_policy_id("SPY", "20261022", _SHA, untrained=True)
        == "spy-ppo-untrained-20261022-d3888e7c"
    )
    with pytest.raises(ValueError, match="sha256"):
        job.make_policy_id("SPY", "20261022", "xyz", untrained=False)


@pytest.mark.parametrize(
    ("model_file", "expected"),
    [
        ("spy_ppo_20261022_seed0.zip", ("spy_ppo_20261022_seed0", "20261022")),
        ("spy_ppo_untrained_20261022_seed4.zip", ("spy_ppo_untrained_20261022_seed4", "20261022")),
    ],
)
def test_run_identity_reads_the_training_artifact_stem(
    model_file: str, expected: tuple[str, str]
) -> None:
    assert job.run_identity({"model_file": model_file}) == expected


@pytest.mark.parametrize("model_file", ["model.zip", "spy_ppo_2026_seed0.zip", 7, None])
def test_run_identity_rejects_other_names(model_file: object) -> None:
    with pytest.raises(ValueError, match="model_file"):
        job.run_identity({"model_file": model_file})


def test_split_bounds_follow_the_train_window_with_the_5_3_geometry() -> None:
    data = _dataset()
    table = _table(data)

    bounds = job.split_bounds(table, _manifest(table, data, n_train=40))

    assert bounds["train"]["bars"] == [0, 40]
    assert bounds["val"]["bars"] == [40, 166]
    assert bounds["purge"]["bars"] == [166, 216]
    assert bounds["test"]["bars"] == [216, 279]
    assert bounds["train"]["first_session"] == "2019-01-02"
    assert bounds["val"]["first_session"] == table.bars[40].session_ts.date().isoformat()
    # Contiguous, non-overlapping, half-open, in order.
    edges = [bounds[k]["bars"] for k in ("train", "val", "purge", "test")]
    assert all(a[1] == b[0] for a, b in itertools.pairwise(edges))


def test_split_bounds_fail_closed() -> None:
    data = _dataset()
    table = _table(data)
    with pytest.raises(ValueError, match="too short"):
        job.split_bounds(table, _manifest(table, data, n_train=100))
    manifest = _manifest(table, data)
    manifest["train_bars"]["first_session_ts"] = "2000-01-03T05:00:00Z"
    with pytest.raises(ValueError, match="not in the dataset"):
        job.split_bounds(table, manifest)
    manifest = _manifest(table, data)
    manifest["train_bars"]["last_session_ts"] = _iso(table, 5)
    with pytest.raises(ValueError, match="does not hold"):
        job.split_bounds(table, manifest)


def _env(**overrides: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "feature_window": 50,
        "feature_names": list(FEATURE_NAMES),
        "feature_scales": dict(_SCALES),
    }
    env.update(overrides)
    return env


def test_feature_spec_is_read_from_the_manifest_including_scales() -> None:
    custom = {**_SCALES, "daily_vol": 0.02}
    spec, source = job.feature_spec_for({"env": _env(feature_scales=custom)})

    assert source == "manifest"
    assert spec.names == FEATURE_NAMES
    assert spec.scales.daily_vol == 0.02
    with pytest.raises(ValueError, match="feature spec"):
        job.feature_spec_for({"env": _env(feature_names=["r1"])})
    with pytest.raises(ValueError, match="feature spec"):
        job.feature_spec_for({"env": _env(feature_scales={"daily_vol": 0.01})})
    with pytest.raises(ValueError, match="feature spec"):
        job.feature_spec_for({"env": _env(feature_scales={**_SCALES, "daily_vol": -1.0})})
    with pytest.raises(ValueError, match="feature_window"):
        job.feature_spec_for({"env": _env(feature_window=True)})


def test_scale_less_manifests_are_default_assumed_only_from_the_rl_train_job() -> None:
    env = _env()
    del env["feature_scales"]
    spec, source = job.feature_spec_for(
        {"env": env, "provenance": {"job": "trading.jobs.rl_train"}}
    )
    assert source == "default_assumed"
    assert spec.scales == FeatureScales()

    for provenance in ({"job": "something.else"}, {}, None):
        with pytest.raises(ValueError, match="refusing to assume the default scales"):
            job.feature_spec_for({"env": env, "provenance": provenance})


def test_read_manifest_checks_kind_and_model(tmp_path: Path) -> None:
    model = tmp_path / f"{_STEM}.zip"
    path = model.with_suffix(".json")
    path.write_text(json.dumps({"kind": "rl.train.ppo", "schema_version": 1, "model_file": "x"}))
    with pytest.raises(ValueError, match="describes 'x'"):
        job.read_manifest(path, model=model)
    path.write_text(json.dumps({"kind": "other", "schema_version": 1, "model_file": model.name}))
    with pytest.raises(ValueError, match=r"not a rl\.train\.ppo"):
        job.read_manifest(path, model=model)
    # schema_version must be exactly the int 1 -- a float 1.0 (== 1 in Python) is a malformed
    # manifest, matching the policy store's strict rule.
    path.write_text(
        json.dumps({"kind": "rl.train.ppo", "schema_version": 1.0, "model_file": model.name})
    )
    with pytest.raises(ValueError, match=r"not a rl\.train\.ppo"):
        job.read_manifest(path, model=model)
    with pytest.raises(ValueError, match="cannot read"):
        job.read_manifest(tmp_path / "missing.json", model=model)


def _patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "Settings", lambda: _SETTINGS)
    monkeypatch.setattr(job, "configure_logging", lambda _level: None)

    def no_deps(_settings: Settings) -> None:
        raise AssertionError("no dependencies may be built before the export succeeds")

    monkeypatch.setattr(job, "build_dependencies", no_deps)


def _layout(tmp_path: Path, *, n: int = 300) -> tuple[Path, Path]:
    data = _dataset(n)
    dataset = tmp_path / "spy.json"
    dataset.write_bytes(data)
    model = tmp_path / f"{_STEM}.zip"
    model.write_bytes(b"not read before the guards")
    model.with_suffix(".json").write_text(json.dumps(_manifest(_table(data), data)))
    return model, dataset


async def test_out_may_not_be_the_model_or_its_manifest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _patch_env(monkeypatch)
    model, dataset = _layout(tmp_path)
    for out in (model, model.with_suffix(".json")):
        args = job.ExportArgs(model=model, dataset=dataset, out=out, confirm=True, overwrite=True)
        assert await job.run_export(args) == 1
        assert "must not be the model" in capsys.readouterr().out


async def test_dataset_sha_mismatch_is_refused_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _patch_env(monkeypatch)
    model, dataset = _layout(tmp_path)
    dataset.write_bytes(_dataset(301))

    args = job.ExportArgs(model=model, dataset=dataset, out=tmp_path / "p.json", confirm=False)
    assert await job.run_export(args) == 1
    assert "is not the dataset this model trained on" in capsys.readouterr().out


async def test_missing_rl_extra_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _patch_env(monkeypatch)

    def missing() -> None:
        raise RlExtraMissingError("the optional [rl] extra is not installed (stub)")

    monkeypatch.setattr(train_module, "require_rl_extra", missing)
    model, dataset = _layout(tmp_path)

    args = job.ExportArgs(model=model, dataset=dataset, out=tmp_path / "p.json", confirm=False)
    assert await job.run_export(args) == 1
    assert "[rl] extra is not installed" in capsys.readouterr().out
    assert not (tmp_path / "p.json").exists()


def test_usage_errors_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert job.main(["--model", "m.zip"]) == 1
