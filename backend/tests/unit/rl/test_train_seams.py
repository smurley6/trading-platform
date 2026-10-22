"""The SB3-free seams of ``rl/train.py`` (task-rl-08): config validation, artifact naming, the
manifest schema, and the SB3 -> ``Policy`` bridge driven by a STUB model through the real
``rl/scoring.py`` rollout. Runs in the default ``pytest`` without the ``[rl]`` extra (train.py
imports stable-baselines3 / torch lazily); the real trainer is ``test_train_smoke.py`` (``rl``).
"""

import json
import math
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tests.unit.rl.conftest import bar_table, synthetic_random_walk
from trading.domain.rl_features import OBS_DIM
from trading.rl.env import EnvConfig
from trading.rl.metrics import Segment
from trading.rl.scoring import rollout
from trading.rl.train import (
    EP_REW_WINDOW,
    REQUIRED_MANIFEST_KEYS,
    TrainConfig,
    TrainError,
    _check_effective,
    artifact_stem,
    build_manifest,
    manifest_to_json,
    passes_over_bars,
    policy_factory,
    policy_state_sha256,
    training_summary,
)

_CREATED = datetime(2026, 10, 22, 12, tzinfo=UTC)
# What effective_hyperparameters reads back from a real model (stubbed: no SB3 here).
_EFFECTIVE: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "net_arch": [64, 64],
    "activation_fn": "Tanh",
    "device": "cpu",
}


class _StubModel:
    """Records every predict call; acts by a fixed rule on the observation (no torch)."""

    def __init__(self, action: Any = None) -> None:
        self.calls: list[tuple[npt.NDArray[np.float32], bool]] = []
        self._action = action

    def predict(
        self,
        observation: npt.NDArray[np.float32],
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[Any, Any]:
        self.calls.append((observation, deterministic))
        if self._action is not None:
            return self._action, None
        # Long iff momentum feature (index 0) is positive: a deterministic function of obs.
        return np.int64(1 if observation[0] > 0 else 2), None


# --- TrainConfig ----------------------------------------------------------------------------------


def test_train_config_defaults_are_the_first_run_pins() -> None:
    config = TrainConfig()
    assert (config.total_timesteps, config.n_steps, config.batch_size, config.n_epochs) == (
        200_000,
        1024,
        64,
        10,
    )
    assert (config.learning_rate, config.ent_coef, config.gamma) == (3e-4, 0.01, 0.99)
    assert config.net_arch == (64, 64)
    assert config.untrained is False
    assert TrainConfig(total_timesteps=0).untrained is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_timesteps": -1},
        {"seed": -1},
        {"seed": True},
        {"n_steps": 0},
        {"batch_size": 2048},
        {"learning_rate": 0.0},
        {"ent_coef": math.nan},
        {"gamma": 1.5},
        {"net_arch": ()},
        {"net_arch": (64, 0)},
    ],
)
def test_train_config_rejects(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TrainError):
        TrainConfig(**kwargs)


# --- naming -------------------------------------------------------------------------------------


def test_artifact_stem_names_trained_and_untrained_distinctly() -> None:
    day = date(2026, 10, 22)
    assert artifact_stem("SPY", day, 3, untrained=False) == "spy_ppo_20261022_seed3"
    assert artifact_stem("SPY", day, 3, untrained=True) == "spy_ppo_untrained_20261022_seed3"
    with pytest.raises(TrainError):
        artifact_stem("../x", day, 0, untrained=False)


def test_passes_over_bars_is_timesteps_per_train_bar() -> None:
    assert passes_over_bars(200_000, 500) == 400.0
    assert passes_over_bars(0, 504) == 0.0
    with pytest.raises(TrainError):
        passes_over_bars(10, 0)


# --- manifest schema ----------------------------------------------------------------------------


def _manifest(**overrides: Any) -> dict[str, Any]:
    bars = bar_table(synthetic_random_walk(400, seed=1))
    window = [{"r": 1.0, "equity": 1.1, "turnover": 4.0, "n_trades": 4}] * 3
    kwargs: dict[str, Any] = {
        "bars": bars,
        "env_config": EnvConfig(),
        "config": TrainConfig(total_timesteps=2048, seed=7),
        "model_file": "spy_ppo_20261022_seed7.zip",
        "model_sha256": "ab" * 32,
        "policy_sha256": "ef" * 32,
        "effective": _EFFECTIVE,
        "num_timesteps": 2048,
        "wall_seconds": 1.25,
        "created_at": _CREATED,
        "training": training_summary([0.5, 1.0, 1.0, 1.0], [252, 252, 252, 252], window),
        "versions": {"sb3": "2.9.0", "torch": "2.14.1", "gymnasium": "1.4.0", "numpy": "2.5.4"},
        "provenance": {"dataset_sha256": "cd" * 32, "git_commit": "deadbeef"},
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_manifest_has_exactly_the_required_keys_and_round_trips_as_strict_json() -> None:
    manifest = _manifest()
    assert set(manifest) == REQUIRED_MANIFEST_KEYS
    parsed = json.loads(manifest_to_json(manifest))
    assert parsed == json.loads(json.dumps(manifest))
    assert parsed["schema_version"] == 1
    assert parsed["seed"] == 7
    assert parsed["untrained"] is False
    assert parsed["passes_over_train_bars"] == 2048 / 400
    assert parsed["policy_sha256"] == "ef" * 32
    assert parsed["model_sha256"] == "ab" * 32
    hyper = parsed["hyperparameters"]
    assert hyper["requested"]["n_steps"] == 1024
    assert hyper["requested"]["ent_coef"] == 0.01
    assert hyper["requested"]["net_arch"] == [64, 64]
    assert hyper["effective"] == _EFFECTIVE  # read back from the model, never assumed
    assert hyper["vec_normalize"] is False
    assert "changes EVERY run" in parsed["seeding"]["hashes"]
    assert parsed["env"]["cost_one_way_bps"] == 1.0
    assert parsed["train_bars"]["n_bars"] == 400
    assert parsed["train_bars"]["episode_start_bounds"] == [49, 400 - 1 - 252]
    assert parsed["seeding"]["torch_num_threads"] == 1
    assert parsed["library_versions"]["sb3"] == "2.9.0"
    assert parsed["provenance"]["dataset_sha256"] == "cd" * 32
    assert parsed["created_at"] == "2026-10-22T12:00:00Z"
    assert "NOT an evaluation result" in parsed["note"]


def test_manifest_json_is_deterministic() -> None:
    assert manifest_to_json(_manifest()) == manifest_to_json(_manifest())


def test_training_summary_first_and_final_windows() -> None:
    returns = [float(i) for i in range(150)]
    window = [{"r": r, "equity": 1.0, "turnover": 2.0, "n_trades": 2} for r in returns[-100:]]
    summary = training_summary(returns, [252] * 150, window)
    assert summary["n_episodes"] == 150
    assert summary["ep_rew_mean_first"] == pytest.approx(49.5)  # mean of episodes 0..99
    assert summary["final_ep_rew_mean"] == pytest.approx(99.5)  # mean of episodes 50..149
    assert summary["final_window"] == {
        "n_episodes": 100,
        "mean_equity": 1.0,
        "mean_turnover": 2.0,
        "mean_n_trades": 2.0,
    }


def test_untrained_summary_has_no_episode_statistics() -> None:
    summary = training_summary([], [], [])
    assert summary["ep_rew_mean_first"] is None
    assert summary["final_ep_rew_mean"] is None
    manifest = _manifest(config=TrainConfig(total_timesteps=0), num_timesteps=0, training=summary)
    parsed = json.loads(manifest_to_json(manifest))
    assert parsed["untrained"] is True
    assert parsed["training"]["final_ep_rew_mean"] is None


def test_manifest_refuses_nan_and_schema_drift() -> None:
    with pytest.raises(TrainError):
        training_summary([math.nan], [252], [])
    with pytest.raises(TrainError):
        _manifest(wall_seconds=math.inf)
    manifest = _manifest()
    manifest["provenance"] = {"x": math.nan}
    with pytest.raises(TrainError, match="strict JSON"):
        manifest_to_json(manifest)
    del manifest["provenance"]
    with pytest.raises(TrainError, match="missing"):
        manifest_to_json(manifest)


# --- the reproducible weights hash ---------------------------------------------------------------


def _state(scale: float = 1.0) -> dict[str, Any]:
    return {
        "action_net.weight": np.arange(6, dtype=np.float32).reshape(3, 2) * scale,
        "action_net.bias": np.zeros(3, dtype=np.float32),
    }


def test_policy_state_sha256_is_canonical_over_names_and_values() -> None:
    reference = policy_state_sha256(_state())
    assert len(reference) == 64
    # Insertion order does not matter; values, names, dtype and shape do.
    assert policy_state_sha256(dict(reversed(list(_state().items())))) == reference
    assert policy_state_sha256(_state(scale=2.0)) != reference
    renamed = {f"x.{k}": v for k, v in _state().items()}
    assert policy_state_sha256(renamed) != reference
    as_f64 = {k: v.astype(np.float64) for k, v in _state().items()}
    assert policy_state_sha256(as_f64) != reference
    reshaped = {**_state(), "action_net.weight": _state()["action_net.weight"].reshape(2, 3)}
    assert policy_state_sha256(reshaped) != reference
    # A one-ulp change to a single weight changes the hash.
    nudged = _state()
    nudged["action_net.bias"][0] = np.nextafter(np.float32(0), np.float32(1))
    assert policy_state_sha256(nudged) != reference


def test_policy_state_sha256_ignores_memory_layout_and_byte_order() -> None:
    weight = _state()["action_net.weight"]
    fortran = {**_state(), "action_net.weight": np.asfortranarray(weight)}
    big_endian = {**_state(), "action_net.weight": weight.astype(">f4")}
    assert policy_state_sha256(fortran) == policy_state_sha256(_state())
    assert policy_state_sha256(big_endian) == policy_state_sha256(_state())


# --- the SB3 -> Policy bridge -------------------------------------------------------------------


def test_policy_factory_calls_predict_deterministically_with_the_exact_observation() -> None:
    model = _StubModel()
    act = policy_factory(model)(123)
    observation = [0.25, -1.0, 0.0, 0.5, 0.1, 0.0, 1.0, 0.0, 0.0]

    assert act(observation, [100.0] * 50) == 1
    (obs, deterministic), *_ = model.calls
    assert deterministic is True
    assert obs.dtype == np.float32 and obs.shape == (OBS_DIM,)
    assert obs.tolist() == [float(np.float32(x)) for x in observation]


@pytest.mark.parametrize("bad", [np.array([3]), np.array([0.5]), np.array([1, 2]), -1])
def test_policy_factory_rejects_actions_outside_discrete_3(bad: Any) -> None:
    act = policy_factory(_StubModel(action=bad))(0)
    with pytest.raises(TrainError):
        act([0.0] * OBS_DIM, [100.0] * 50)


def test_bridged_policy_replays_through_scoring_like_any_baseline() -> None:
    bars = bar_table(synthetic_random_walk(400, seed=11))
    segment = Segment(start=60, length=200)
    model = _StubModel()

    first = rollout(policy_factory(model), bars, segment, seed=0)
    second = rollout(policy_factory(_StubModel()), bars, segment, seed=99)

    assert first.steps == 200
    assert first.n_trades > 0
    assert len(model.calls) == 200
    assert all(deterministic for _, deterministic in model.calls)
    # Deterministic policy: the seed changes nothing.
    assert (first.actions, first.equity) == (second.actions, second.equity)


# --- the fail-closed hyperparameter check (task-rl-08 re-review finding) ------------------------


def test_check_effective_passes_when_the_model_applied_every_requested_setting() -> None:
    effective = {**_EFFECTIVE, "stats_window_size": EP_REW_WINDOW}
    _check_effective(TrainConfig(), effective)  # must not raise


@pytest.mark.parametrize(
    ("key", "wrong"),
    [("n_steps", 2048), ("gae_lambda", 0.9), ("clip_range", 0.3), ("device", "mps")],
)
def test_check_effective_fails_closed_when_a_setting_was_not_applied(key: str, wrong: Any) -> None:
    effective = {**_EFFECTIVE, "stats_window_size": EP_REW_WINDOW, key: wrong}
    with pytest.raises(TrainError, match="did not apply"):
        _check_effective(TrainConfig(), effective)


def test_check_effective_fails_closed_when_a_setting_is_missing_entirely() -> None:
    effective = {**_EFFECTIVE}  # no stats_window_size at all
    with pytest.raises(TrainError, match="did not apply"):
        _check_effective(TrainConfig(), effective)
