"""SB3 PPO -> torch-free MlpPolicy export + the parity gate (task-rl-09 acceptance tests 4 + 5).
``rl``-marked; skipped without the ``[rl]`` extra.

Hermetic: the models are trained (briefly) or initialized in-test on a synthetic random walk. The
real seed-0 exhibit artifact is checked too WHEN it is present (``backend/artifacts/`` is
git-ignored, so it is skipped on a fresh checkout).
"""

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.rl.conftest import bar_table, synthetic_random_walk
from trading.domain.rl_policy import ENV_ACTION_ORDER, DenseLayer, MlpPolicy, RlAction
from trading.infrastructure.rl.policy_store import load_policy, write_policy
from trading.rl.bars import BarTable
from trading.rl.env import DEFAULT_FEATURE_SPEC, EnvConfig, SingleTickerTradingEnv
from trading.rl.export import (
    LOGIT_TOL,
    TIE_MARGIN,
    ExportError,
    ParityError,
    ParityReport,
    assert_parity,
    mlp_policy_factory,
    sample_observations,
    sb3_logits,
    sb3_to_mlp_policy,
)
from trading.rl.metrics import Segment
from trading.rl.scoring import rollout
from trading.rl.train import TrainConfig, load_model, policy_factory, policy_state_sha256, train_ppo

pytestmark = pytest.mark.rl
sb3 = pytest.importorskip("stable_baselines3")
th = pytest.importorskip("torch")
torch_layers = pytest.importorskip("stable_baselines3.common.torch_layers")

_N_OBS = 1_000
_SEED0_ZIP = Path(__file__).parents[3] / "artifacts" / "rl" / "spy_ppo_20261022_seed0.zip"
_SEED0_POLICY_SHA256 = "d3888e7c9a261ec2dbf8cad207a0a879283b88a6a5e983049e230d4214c435c5"


def _bars() -> BarTable:
    return bar_table(synthetic_random_walk(400, seed=11))


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A briefly TRAINED PPO (one 1024-step rollout) through the real training path."""
    out = tmp_path_factory.mktemp("rl_export_model")
    result = train_ppo(
        _bars(),
        EnvConfig(),
        TrainConfig(total_timesteps=1_024, seed=3),
        out_dir=out,
        provenance={"dataset_sha256": "synthetic"},
        run_date=date(2026, 10, 22),
    )
    return load_model(result.model_path)


def _fresh_model(**policy_kwargs: Any) -> Any:
    env = SingleTickerTradingEnv(_bars(), EnvConfig())
    return sb3.PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=0, device="cpu", n_steps=64)


def _export(model: Any) -> MlpPolicy:
    return sb3_to_mlp_policy(
        model, policy_id="spy-ppo-20261022-test0000", spec=DEFAULT_FEATURE_SPEC
    )


def _observations(model: Any, n: int = _N_OBS, seed: int = 0) -> list[tuple[float, ...]]:
    return sample_observations(model.observation_space, n, seed=seed)


def test_parity_over_1000_random_observations(trained_model: Any) -> None:
    policy = _export(trained_model)
    observations = _observations(trained_model)
    assert len(observations) >= 1_000

    report = assert_parity(trained_model, policy, observations)

    # The spec's check, written out independently of assert_parity's loop — TWO-SIDED: logits must
    # match SB3's, and a mismatch is a tie only if BOTH sides' competing logits are within 1e-4.
    ties = 0
    for obs in observations:
        obs32 = np.asarray(obs, dtype=np.float32)
        action, _ = trained_model.predict(obs32, deterministic=True)
        theirs = int(action)
        ours_logits = policy.logits(obs)
        sb3 = sb3_logits(trained_model, obs)
        assert max(abs(a - b) for a, b in zip(ours_logits, sb3, strict=True)) < LOGIT_TOL
        ours = policy.action_index(obs)
        if ENV_ACTION_ORDER[theirs] != policy.act(obs):
            assert abs(ours_logits[ours] - ours_logits[theirs]) < 1e-4
            assert abs(sb3[ours] - sb3[theirs]) < 1e-4
            ties += 1
    assert isinstance(report, ParityReport)
    assert report.ok
    assert report.n_observations == len(observations)
    assert report.n_disagreements == 0
    assert report.n_ties == ties
    assert report.n_agree + report.n_ties == report.n_observations
    assert report.tie_margin == TIE_MARGIN == 1e-4
    assert report.logit_tol == LOGIT_TOL == 1e-3
    assert 0.0 <= report.max_abs_logit_diff < 1e-4  # float32 rounding only


def test_exported_weights_are_the_state_dict_values(trained_model: Any) -> None:
    policy = _export(trained_model)
    state = trained_model.policy.state_dict()

    assert [layer.activation for layer in policy.hidden] == ["tanh", "tanh"]
    assert [layer.out_features for layer in policy.hidden] == [64, 64]
    assert policy.action_head.activation == "identity"
    assert policy.actions == ENV_ACTION_ORDER
    assert policy.features == DEFAULT_FEATURE_SPEC
    for layer, key in zip(
        (*policy.hidden, policy.action_head),
        ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net"),
        strict=True,
    ):
        assert layer.weight == tuple(tuple(r) for r in state[f"{key}.weight"].tolist())
        assert layer.bias == tuple(state[f"{key}.bias"].tolist())


def test_sb3_logits_match_the_stdlib_forward_pass(trained_model: Any) -> None:
    policy = _export(trained_model)
    for obs in _observations(trained_model, n=50, seed=9):
        theirs = sb3_logits(trained_model, obs)
        ours = policy.logits(obs)
        assert max(abs(a - b) for a, b in zip(theirs, ours, strict=True)) < 1e-4


def test_a_corrupted_policy_fails_the_gate(trained_model: Any) -> None:
    # Negative control: swapping the BUY and SELL logit rows is a real defect; the gate must see it.
    policy = _export(trained_model)
    head = policy.action_head
    swapped = DenseLayer(
        weight=(head.weight[0], head.weight[2], head.weight[1]),
        bias=(head.bias[0], head.bias[2], head.bias[1]),
        activation="identity",
    )
    broken = MlpPolicy(
        policy_id=policy.policy_id,
        schema_version=policy.schema_version,
        features=policy.features,
        hidden=policy.hidden,
        action_head=swapped,
        actions=policy.actions,
    )

    with pytest.raises(ParityError) as caught:
        assert_parity(trained_model, broken, _observations(trained_model, n=300))
    report = caught.value.report
    assert report.n_disagreements > 0
    assert not report.ok
    assert report.disagreement_indices


def _with_head(policy: MlpPolicy, weight: Any, bias: Any) -> MlpPolicy:
    head = DenseLayer(weight=weight, bias=bias, activation="identity")
    return MlpPolicy(
        policy_id=policy.policy_id,
        schema_version=policy.schema_version,
        features=policy.features,
        hidden=policy.hidden,
        action_head=head,
        actions=policy.actions,
    )


def _zeroed_head(policy: MlpPolicy) -> MlpPolicy:
    """A DEAD policy: all logits 0 -> always HOLD (index 0 wins every exact tie)."""
    head = policy.action_head
    return _with_head(
        policy, tuple((0.0,) * len(row) for row in head.weight), (0.0,) * len(head.bias)
    )


def _scaled_sign_flip(policy: MlpPolicy) -> MlpPolicy:
    """An ARGMIN policy whose logits are all tiny (x -1e-7): every margin is far below 1e-4."""
    head = policy.action_head
    return _with_head(
        policy,
        tuple(tuple(-1e-7 * w for w in row) for row in head.weight),
        tuple(-1e-7 * b for b in head.bias),
    )


def test_a_dead_always_hold_policy_fails_the_gate(trained_model: Any) -> None:
    # Review finding: with a one-sided tie rule a zeroed head "passed" as agree + ties.
    with pytest.raises(ParityError) as caught:
        assert_parity(
            trained_model, _zeroed_head(_export(trained_model)), _observations(trained_model)
        )
    report = caught.value.report
    assert not report.ok
    assert report.max_abs_logit_diff >= LOGIT_TOL


def test_a_scaled_sign_flipped_argmin_policy_fails_the_gate(trained_model: Any) -> None:
    # Review finding: x -1e-7 made every stdlib margin < 1e-4, so all 1000 mismatches were "ties".
    with pytest.raises(ParityError) as caught:
        assert_parity(
            trained_model, _scaled_sign_flip(_export(trained_model)), _observations(trained_model)
        )
    report = caught.value.report
    assert report.n_ties == 0
    assert report.n_disagreements > 0


def test_matching_actions_with_drifted_logits_fail_the_gate(trained_model: Any) -> None:
    # Actions alone are not enough: a uniform +0.01 logit offset keeps every argmax but is not the
    # same function, so the logit tolerance must catch it.
    policy = _export(trained_model)
    head = policy.action_head
    shifted = _with_head(policy, head.weight, tuple(b + 0.01 for b in head.bias))
    observations = _observations(trained_model, n=200)
    assert [shifted.act(o) for o in observations] == [policy.act(o) for o in observations]

    with pytest.raises(ParityError) as caught:
        assert_parity(trained_model, shifted, observations)
    report = caught.value.report
    assert report.n_disagreements == 0
    assert report.max_abs_logit_diff >= LOGIT_TOL


def test_parity_needs_observations_of_the_right_shape(trained_model: Any) -> None:
    policy = _export(trained_model)
    with pytest.raises(ExportError, match="at least one"):
        assert_parity(trained_model, policy, [])
    with pytest.raises(ExportError, match="shape"):
        assert_parity(trained_model, policy, [(0.0,) * 8])


def test_export_refuses_unsupported_architecture() -> None:
    # ReLU is supported: it exports, and exactly.
    relu = _fresh_model(activation_fn=th.nn.ReLU, net_arch=[32])
    policy = _export(relu)
    assert [(layer.out_features, layer.activation) for layer in policy.hidden] == [(32, "relu")]
    report = assert_parity(relu, policy, _observations(relu))
    assert report.ok

    # An unsupported activation is refused.
    with pytest.raises(ExportError, match="unsupported module LeakyReLU"):
        _export(_fresh_model(activation_fn=th.nn.LeakyReLU, net_arch=[16]))

    # A non-FlattenExtractor features extractor MUST be refused (built with type() so no SB3 class
    # is subclassed in source).
    def init(self: Any, observation_space: Any, features_dim: int = 9) -> None:
        torch_layers.BaseFeaturesExtractor.__init__(self, observation_space, features_dim)
        self.linear = th.nn.Linear(9, features_dim)

    def forward(self: Any, observations: Any) -> Any:
        return self.linear(observations)

    extractor = type(
        "LinearExtractor",
        (torch_layers.BaseFeaturesExtractor,),
        {"__init__": init, "forward": forward},
    )
    with pytest.raises(ExportError, match="not FlattenExtractor"):
        _export(_fresh_model(features_extractor_class=extractor))


def test_export_refuses_an_unsupported_policy_net_module() -> None:
    # Any policy_net module the stdlib pass cannot reproduce is refused, wherever it appears.
    model = _fresh_model(net_arch=[8])
    model.policy.mlp_extractor.policy_net.append(th.nn.Dropout(0.5))
    with pytest.raises(ExportError, match="unsupported module Dropout"):
        _export(model)


def test_exported_policy_round_trips_through_the_store(trained_model: Any, tmp_path: Path) -> None:
    policy = _export(trained_model)
    path = tmp_path / "policy.json"
    write_policy(path, policy, meta={"source": "test"})

    reread = load_policy(path)
    assert reread == policy
    observations = _observations(trained_model, n=200, seed=4)
    assert [reread.act(o) for o in observations] == [policy.act(o) for o in observations]
    assert assert_parity(trained_model, reread, observations).ok
    assert 10_000 < path.stat().st_size < 200_000  # ~100 KB for 9 -> 64 -> 64 -> 3


def test_mlp_policy_factory_replays_like_the_sb3_bridge(trained_model: Any) -> None:
    # Through the identical env replay (rl/scoring.py), the stdlib policy and the SB3 model take
    # the same actions (barring float32 ties, which this replay does not hit).
    policy = _export(trained_model)
    bars = _bars()
    segment = Segment(start=60, length=200)
    ours = rollout(mlp_policy_factory(policy), bars, segment, seed=0)
    theirs = rollout(policy_factory(trained_model), bars, segment, seed=0)

    assert ours.requested_actions == theirs.requested_actions
    assert ours.equity == theirs.equity


def test_mlp_policy_factory_rejects_a_foreign_action_order(trained_model: Any) -> None:
    policy = _export(trained_model)
    reordered = MlpPolicy(
        policy_id=policy.policy_id,
        schema_version=policy.schema_version,
        features=policy.features,
        hidden=policy.hidden,
        action_head=policy.action_head,
        actions=(RlAction.SELL, RlAction.HOLD, RlAction.BUY),
    )
    with pytest.raises(ExportError, match="env's"):
        mlp_policy_factory(reordered)


@pytest.mark.skipif(not _SEED0_ZIP.exists(), reason="seed-0 exhibit artifact not present")
def test_real_seed0_model_exports_with_parity() -> None:
    model = load_model(_SEED0_ZIP)
    assert policy_state_sha256(model.policy.state_dict()) == _SEED0_POLICY_SHA256
    policy = _export(model)

    observations = _observations(model, n=2_000, seed=1)
    report = assert_parity(model, policy, observations)

    assert report.ok
    assert report.n_observations == 2_000
    assert math.isfinite(report.max_abs_logit_diff)
    assert report.max_abs_logit_diff < 1e-4
    # The review's demonstrations on THIS model must fail the gate.
    for broken in (_zeroed_head(policy), _scaled_sign_flip(policy)):
        with pytest.raises(ParityError):
            assert_parity(model, broken, observations[:1_000])
