"""The real PPO trainer end to end (task-rl-08 acceptance test 5 + the untrained baseline + the
scoring bridge + seeded determinism). ``rl``-marked; skipped without the ``[rl]`` extra."""

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.rl.conftest import bar_table, synthetic_random_walk
from trading.rl.bars import BarTable
from trading.rl.env import EnvConfig, SingleTickerTradingEnv
from trading.rl.metrics import Segment
from trading.rl.scoring import rollout
from trading.rl.train import (
    REQUIRED_MANIFEST_KEYS,
    TrainConfig,
    TrainError,
    TrainResult,
    load_model,
    policy_factory,
    policy_state_sha256,
    train_ppo,
)

pytestmark = pytest.mark.rl
sb3 = pytest.importorskip("stable_baselines3")
torch = pytest.importorskip("torch")

_DAY = date(2026, 10, 22)


def _bars() -> BarTable:
    return bar_table(synthetic_random_walk(400, seed=5))


def _train(out_dir: Path, *, timesteps: int, seed: int = 0, overwrite: bool = False) -> TrainResult:
    return train_ppo(
        _bars(),
        EnvConfig(),
        TrainConfig(total_timesteps=timesteps, seed=seed),
        out_dir=out_dir,
        provenance={"dataset_sha256": "synthetic"},
        run_date=_DAY,
        overwrite=overwrite,
    )


def _params(path: Path) -> dict[str, Any]:
    return {k: v.clone() for k, v in load_model(path).policy.state_dict().items()}


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_short_training_completes_saves_and_round_trips(tmp_path: Path) -> None:
    threads_before = torch.get_num_threads()
    result = _train(tmp_path, timesteps=2_000)

    assert result.final_ep_rew_mean is not None and math.isfinite(result.final_ep_rew_mean)
    assert result.num_timesteps == 2_048  # SB3 completes the last n_steps=1024 rollout
    assert result.model_path == tmp_path / "spy_ppo_20261022_seed0.zip"
    assert result.untrained is False
    model = sb3.PPO.load(str(result.model_path), device="cpu")
    env = SingleTickerTradingEnv(_bars(), EnvConfig())
    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in {0, 1, 2}

    manifest = json.loads(result.manifest_path.read_text())
    assert set(manifest) == REQUIRED_MANIFEST_KEYS
    assert manifest["model_sha256"] == hashlib.sha256(result.model_path.read_bytes()).hexdigest()
    # The weights hash survives the save/load round trip (it is computed from the weights, not
    # the file), so a reader can verify any saved .zip against its manifest.
    reloaded = policy_state_sha256(load_model(result.model_path).policy.state_dict())
    assert manifest["policy_sha256"] == result.policy_sha256 == reloaded
    effective = manifest["hyperparameters"]["effective"]  # read back from the trained model
    assert (effective["n_steps"], effective["batch_size"], effective["n_epochs"]) == (1024, 64, 10)
    assert (effective["gae_lambda"], effective["clip_range"], effective["ent_coef"]) == (
        0.95,
        0.2,
        0.01,
    )
    assert effective["net_arch"] == [64, 64]
    assert effective["activation_fn"] == "Tanh"
    assert effective["features_extractor"] == "FlattenExtractor"
    assert manifest["num_timesteps"] == 2_048
    assert manifest["training"]["final_ep_rew_mean"] == result.final_ep_rew_mean
    assert manifest["training"]["n_episodes"] == len(manifest["training"]["episode_returns"]) >= 1
    assert manifest["library_versions"]["sb3"] == result.sb3_version == sb3.__version__
    assert manifest["provenance"] == {"dataset_sha256": "synthetic"}
    assert torch.get_num_threads() == threads_before  # the training-time pin was restored


def test_same_seed_is_bit_identical_and_a_different_seed_is_not(tmp_path: Path) -> None:
    a = _train(tmp_path / "a", timesteps=1_024, seed=3)
    b = _train(tmp_path / "b", timesteps=1_024, seed=3)
    c = _train(tmp_path / "c", timesteps=1_024, seed=4)

    assert _same(_params(a.model_path), _params(b.model_path))
    assert a.final_ep_rew_mean == b.final_ep_rew_mean
    assert not _same(_params(a.model_path), _params(c.model_path))
    # policy_sha256 is THE reproducibility check: equal for the same seed, different otherwise,
    # and matching what each manifest records.
    assert a.policy_sha256 == b.policy_sha256 != c.policy_sha256
    for result in (a, b, c):
        recorded = json.loads(result.manifest_path.read_text())["policy_sha256"]
        assert recorded == result.policy_sha256
    # ... while the .zip's own hash differs even for bit-identical weights (SB3 embeds the
    # wall-clock start_time, Monitor elapsed times, ...): it is file integrity, not reproducibility.
    zip_a = json.loads(a.manifest_path.read_text())["model_sha256"]
    zip_b = json.loads(b.manifest_path.read_text())["model_sha256"]
    assert zip_a != zip_b


def test_untrained_baseline_is_the_seeds_initialization(tmp_path: Path) -> None:
    untrained = _train(tmp_path / "u", timesteps=0, seed=2)
    trained = _train(tmp_path / "t", timesteps=1_024, seed=2)

    assert untrained.untrained is True
    assert untrained.num_timesteps == 0
    assert untrained.final_ep_rew_mean is None
    assert untrained.model_path.name == "spy_ppo_untrained_20261022_seed2.zip"
    manifest = json.loads(untrained.manifest_path.read_text())
    assert manifest["untrained"] is True
    assert manifest["training"]["n_episodes"] == 0
    # Exactly the weights a seed-2 PPO is constructed with (the trained run's starting point) ...
    venv = sb3.common.vec_env.DummyVecEnv([lambda: SingleTickerTradingEnv(_bars(), EnvConfig())])
    fresh = sb3.PPO("MlpPolicy", venv, seed=2, device="cpu", policy_kwargs={"net_arch": [64, 64]})
    init = {k: v.clone() for k, v in fresh.policy.state_dict().items()}
    assert _same(_params(untrained.model_path), init)
    # ... and training moved them.
    assert not _same(_params(untrained.model_path), _params(trained.model_path))


def test_refuses_to_overwrite_an_existing_stem_unless_asked(tmp_path: Path) -> None:
    first = _train(tmp_path, timesteps=0, seed=1)
    before = first.model_path.read_bytes()

    with pytest.raises(TrainError, match="refusing to overwrite"):
        _train(tmp_path, timesteps=0, seed=1)
    assert first.model_path.read_bytes() == before

    again = _train(tmp_path, timesteps=0, seed=1, overwrite=True)
    assert again.policy_sha256 == first.policy_sha256


def test_deterministic_predict_is_the_argmax_of_the_policy_logits(tmp_path: Path) -> None:
    model = load_model(_train(tmp_path, timesteps=1_024).model_path)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.uniform(-3.0, 3.0, size=9).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        with torch.no_grad():
            # Any: a CategoricalDistribution here; SB3 types it as a Distribution union.
            dist: Any = model.policy.get_distribution(torch.as_tensor(obs).unsqueeze(0))
            logits = dist.distribution.logits[0]
        assert int(action) == int(torch.argmax(logits))


def test_trained_and_untrained_policies_replay_through_scoring(tmp_path: Path) -> None:
    trained = load_model(_train(tmp_path / "t", timesteps=1_024).model_path)
    untrained = load_model(_train(tmp_path / "u", timesteps=0).model_path)
    bars = _bars()
    segment = Segment(start=60, length=200)

    for model in (trained, untrained):
        first = rollout(policy_factory(model), bars, segment, seed=0)
        again = rollout(policy_factory(model), bars, segment, seed=17)
        assert first.steps == 200
        assert set(first.requested_actions) <= {0, 1, 2}
        assert all(math.isfinite(e) and e > 0 for e in first.equity)
        # Deterministic (argmax) policy: identical replay whatever the seed.
        assert (first.requested_actions, first.equity) == (again.requested_actions, again.equity)
