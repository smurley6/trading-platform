"""The torch-free policy artifact store (task-rl-09). NO extras: this is the deployable read path,
so it runs in the default ``pytest`` with stdlib only (the import closure is proven torch / numpy
free by ``tests/unit/test_ml_deps_isolated.py``).
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from trading.domain.rl_features import FEATURE_NAMES, WARMUP_BARS, FeatureScales, FeatureSpec
from trading.domain.rl_policy import ENV_ACTION_ORDER, DenseLayer, MlpPolicy, RlAction
from trading.infrastructure.rl.policy_store import (
    POLICY_SCHEMA_VERSION,
    PolicyArtifactError,
    load_policy,
    policy_from_json,
    policy_to_json,
    write_policy,
)

_ZEROS7 = [0.0] * 7


def _document() -> dict[str, Any]:
    """A hand-built schema-v1 artifact: 9 -> 2 (tanh) -> 3 (logits). Hidden unit 0 reads r1,
    unit 1 reads r5; logit 0 (HOLD) is constant 0, logit 1 (BUY) = h0, logit 2 (SELL) = -h1."""
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "kind": "rl.policy.mlp",
        "policy_id": "spy-ppo-20261022-deadbeef",
        "features": FeatureSpec(
            window=WARMUP_BARS, names=FEATURE_NAMES, scales=FeatureScales()
        ).to_dict(),
        "hidden": [
            {
                "weight": [[1.0, 0.0, *_ZEROS7], [0.0, 1.0, *_ZEROS7]],
                "bias": [0.0, 0.0],
                "activation": "tanh",
            }
        ],
        "action_head": {
            "weight": [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]],
            "bias": [0.0, 0.0, 0.0],
            "activation": "identity",
        },
        "actions": ["hold", "buy", "sell"],
        "meta": {"note": "hand-built"},
    }


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_hand_built_artifact_round_trips_and_acts_on_the_hand_computed_argmax(
    tmp_path: Path,
) -> None:
    policy = load_policy(_write(tmp_path / "p.json", _document()))

    assert policy.policy_id == "spy-ppo-20261022-deadbeef"
    assert policy.schema_version == POLICY_SCHEMA_VERSION
    assert policy.actions == ENV_ACTION_ORDER
    # logits = (0, tanh(r1), -tanh(r5)), computed by hand:
    buy_obs = (0.5, -0.25, *_ZEROS7)  # (0, 0.4621.., 0.2449..) -> BUY
    sell_obs = (-0.5, -1.0, *_ZEROS7)  # (0, -0.4621.., 0.7615..) -> SELL
    hold_obs = (-0.5, 1.0, *_ZEROS7)  # (0, -0.4621.., -0.7615..) -> HOLD
    assert policy.logits(buy_obs) == (0.0, math.tanh(0.5), -math.tanh(-0.25))
    assert policy.act(buy_obs) is RlAction.BUY
    assert policy.act(sell_obs) is RlAction.SELL
    assert policy.act(hold_obs) is RlAction.HOLD
    assert policy.act((0.0,) * 9) is RlAction.HOLD  # all-zero logits: lowest index wins

    # write -> load is the identity, and the bytes are deterministic.
    out = tmp_path / "again.json"
    sha = write_policy(out, policy, meta={"note": "hand-built"})
    assert load_policy(out) == policy
    assert sha == hashlib.sha256(out.read_bytes()).hexdigest()
    assert policy_to_json(policy, meta={"note": "hand-built"}) == out.read_bytes()


def test_schema_version_mismatch_raises_policy_artifact_error(tmp_path: Path) -> None:
    for version in (POLICY_SCHEMA_VERSION + 1, 0, None, True, "1", 1.0):
        document = _document()
        document["schema_version"] = version
        with pytest.raises(PolicyArtifactError, match="schema_version"):
            load_policy(_write(tmp_path / "p.json", document))
    del document["schema_version"]
    with pytest.raises(PolicyArtifactError, match="schema_version"):
        load_policy(_write(tmp_path / "p.json", document))


def test_floats_round_trip_bit_exactly(tmp_path: Path) -> None:
    # float32 weights widened to float64 must read back identically (repr round-trip).
    awkward = [0.1, 1 / 3, -2.718281828459045, 1e-38, 3.4028234663852886e38, 5e-324]
    document = _document()
    document["hidden"][0]["bias"] = awkward[:2]
    document["action_head"]["bias"] = awkward[2:5]
    policy = load_policy(_write(tmp_path / "p.json", document))
    out = tmp_path / "out.json"
    write_policy(out, policy, meta={})

    reread = load_policy(out)
    assert reread.hidden[0].bias == tuple(awkward[:2])
    assert reread.action_head.bias == tuple(awkward[2:5])


def test_write_is_atomic_and_creates_parents(tmp_path: Path) -> None:
    policy = load_policy(_write(tmp_path / "p.json", _document()))
    out = tmp_path / "nested" / "dir" / "policy.json"

    write_policy(out, policy, meta={"a": [1, 2]})

    assert out.exists()
    assert not out.with_name(out.name + ".tmp").exists()
    assert json.loads(out.read_text())["meta"] == {"a": [1, 2]}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(kind="rl.policy.cnn"), "kind"),
        (lambda d: d.update(extra=1), "keys"),
        (lambda d: d.pop("meta"), "keys"),
        (lambda d: d.update(meta=[]), "meta"),
        (lambda d: d.update(actions=["hold", "buy", "short"]), "invalid policy artifact"),
        (lambda d: d.update(actions=["hold", "buy"]), "invalid policy artifact"),
        (lambda d: d["hidden"][0].update(activation="gelu"), "activation"),
        (lambda d: d["hidden"][0].pop("bias"), "exactly the keys"),
        (lambda d: d["hidden"][0].update(bias=[0.0]), "hidden\\[0\\]"),
        (lambda d: d["action_head"]["weight"].pop(), "hidden|action_head|invalid"),
        (lambda d: d["features"].update(window=10), "invalid policy artifact"),
        (lambda d: d["hidden"][0]["weight"][0].append(0.0), "same length"),
        (lambda d: d.update(policy_id=""), "invalid policy artifact"),
        (lambda d: d.update(hidden={}), "hidden"),
    ],
)
def test_malformed_artifacts_fail_closed(tmp_path: Path, mutate: Any, message: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(PolicyArtifactError, match=message):
        load_policy(_write(tmp_path / "p.json", document))


def test_non_finite_numbers_are_refused_both_ways(tmp_path: Path) -> None:
    text = json.dumps(_document()).replace('"bias": [0.0, 0.0]', '"bias": [NaN, 0.0]', 1)
    with pytest.raises(PolicyArtifactError, match="non-finite"):
        policy_from_json(text.encode())

    policy = load_policy(_write(tmp_path / "p.json", _document()))
    with pytest.raises(PolicyArtifactError, match="strict JSON"):
        policy_to_json(policy, meta={"x": math.inf})
    with pytest.raises(PolicyArtifactError, match="strict JSON"):
        policy_to_json(policy, meta={"x": object()})


def test_unreadable_or_invalid_json_raises_policy_artifact_error(tmp_path: Path) -> None:
    with pytest.raises(PolicyArtifactError, match="cannot read"):
        load_policy(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(PolicyArtifactError, match="not valid JSON"):
        load_policy(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[]")
    with pytest.raises(PolicyArtifactError, match="JSON object"):
        load_policy(tmp_path / "list.json")


def test_writer_refuses_another_schema_version() -> None:
    head = DenseLayer(weight=((0.0,) * 9,) * 3, bias=(0.0,) * 3, activation="identity")
    spec = FeatureSpec(window=WARMUP_BARS, names=FEATURE_NAMES, scales=FeatureScales())
    policy = MlpPolicy("p", POLICY_SCHEMA_VERSION + 1, spec, (), head, ENV_ACTION_ORDER)

    with pytest.raises(PolicyArtifactError, match="schema_version"):
        policy_to_json(policy, meta={})
