"""The torch-free MLP policy (task-rl-09; program doc §3.4). Stdlib only — no optional extras.

Pins the forward pass against a hand-computed 3 -> 2 -> 2 tanh net, the argmax-only (never
sampled) action rule, its lowest-index tie break, and the fail-closed validation.
"""

import math
import random

import pytest

from trading.domain.rl_features import FEATURE_NAMES, WARMUP_BARS, FeatureScales, FeatureSpec
from trading.domain.rl_policy import ENV_ACTION_ORDER, DenseLayer, MlpPolicy, RlAction

_SPEC = FeatureSpec(window=WARMUP_BARS, names=FEATURE_NAMES, scales=FeatureScales())

# The hand-written 3 -> 2 -> 2 net.
_W1 = ((0.5, -1.0, 0.25), (-0.75, 0.1, 2.0))
_B1 = (0.1, -0.2)
_W2 = ((1.5, -0.5), (-2.0, 0.75))
_B2 = (0.05, -0.3)
_X = (0.3, -0.7, 1.1)


def _hand_logits() -> tuple[float, float]:
    # Written out term by term, independent of DenseLayer's loops.
    h0 = math.tanh(0.5 * 0.3 + (-1.0) * (-0.7) + 0.25 * 1.1 + 0.1)  # tanh(1.225)
    h1 = math.tanh((-0.75) * 0.3 + 0.1 * (-0.7) + 2.0 * 1.1 + (-0.2))  # tanh(1.705)
    return (1.5 * h0 + (-0.5) * h1 + 0.05, (-2.0) * h0 + 0.75 * h1 + (-0.3))


def _embedded_policy(*, actions: tuple[RlAction, ...] = (RlAction.HOLD, RlAction.BUY)) -> MlpPolicy:
    """The 3 -> 2 -> 2 net inside a 9-input MlpPolicy: features 3..8 get zero weight."""
    first = DenseLayer(
        weight=tuple((*row, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for row in _W1),
        bias=_B1,
        activation="tanh",
    )
    head = DenseLayer(weight=_W2, bias=_B2, activation="identity")
    return MlpPolicy(
        policy_id="test-ppo-20261022-00000000",
        schema_version=1,
        features=_SPEC,
        hidden=(first,),
        action_head=head,
        actions=actions,
    )


def test_forward_pass_matches_hand_computed_tanh_mlp() -> None:
    expected = _hand_logits()
    hidden = DenseLayer(weight=_W1, bias=_B1, activation="tanh")
    head = DenseLayer(weight=_W2, bias=_B2, activation="identity")

    logits = head.forward(hidden.forward(_X))

    assert len(logits) == 2
    for got, want in zip(logits, expected, strict=True):
        assert abs(got - want) < 1e-12
    # The same net embedded in a full 9-input MlpPolicy gives the same logits.
    via_policy = _embedded_policy().logits((*_X, 9.0, -9.0, 3.0, 1.0, 1.0, -4.0))
    for got, want in zip(via_policy, expected, strict=True):
        assert abs(got - want) < 1e-12


def test_act_is_the_argmax_of_the_logits() -> None:
    policy = _embedded_policy()
    obs = (*_X, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected = _hand_logits()
    best = 0 if expected[0] >= expected[1] else 1

    assert policy.action_index(obs) == best
    assert policy.act(obs) is (RlAction.HOLD, RlAction.BUY)[best]


def test_act_never_samples() -> None:
    # Determinism contract: the action depends only on the observation — never on any RNG state.
    policy = _embedded_policy()
    rng = random.Random(7)
    observations = [tuple(rng.uniform(-10, 10) for _ in range(9)) for _ in range(200)]
    first = [policy.act(o) for o in observations]
    random.seed(12345)
    second = [policy.act(o) for o in observations]

    assert first == second


def test_exact_tie_resolves_to_the_lowest_index() -> None:
    head = DenseLayer(weight=((0.0,) * 9,) * 3, bias=(0.0, 0.0, 0.0), activation="identity")
    policy = MlpPolicy(
        policy_id="tie",
        schema_version=1,
        features=_SPEC,
        hidden=(),
        action_head=head,
        actions=ENV_ACTION_ORDER,
    )

    assert policy.logits((1.0,) * 9) == (0.0, 0.0, 0.0)
    assert policy.act((1.0,) * 9) is RlAction.HOLD


def test_relu_and_identity_activations() -> None:
    layer = DenseLayer(weight=((1.0, 0.0), (0.0, 1.0)), bias=(0.0, 0.0), activation="relu")
    linear = DenseLayer(weight=((1.0, 0.0), (0.0, 1.0)), bias=(0.5, 0.5), activation="identity")

    assert layer.forward((-2.0, 3.0)) == (0.0, 3.0)
    assert linear.forward((-2.0, 3.0)) == (-1.5, 3.5)


def test_env_action_order_is_hold_buy_sell() -> None:
    assert ENV_ACTION_ORDER == (RlAction.HOLD, RlAction.BUY, RlAction.SELL)
    assert [a.value for a in ENV_ACTION_ORDER] == ["hold", "buy", "sell"]


def test_lists_are_normalized_to_tuples() -> None:
    layer = DenseLayer(weight=[[1, 2], [3, 4]], bias=[0, 1], activation="identity")  # type: ignore[arg-type]

    assert layer.weight == ((1.0, 2.0), (3.0, 4.0))
    assert layer.bias == (0.0, 1.0)


@pytest.mark.parametrize(
    ("weight", "bias", "activation", "message"),
    [
        ((), (), "tanh", "non-empty"),
        (((1.0, 2.0), (3.0,)), (0.0, 0.0), "tanh", "same length"),
        (((1.0,),), (0.0, 0.0), "tanh", "bias must have 1"),
        (((math.nan,),), (0.0,), "tanh", "finite"),
        (((1.0,),), (math.inf,), "tanh", "finite"),
        (((True,),), (0.0,), "tanh", "number"),
        (((1.0,),), (0.0,), "sigmoid", "activation"),
    ],
)
def test_dense_layer_rejects_malformed_input(
    weight: object, bias: object, activation: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DenseLayer(weight=weight, bias=bias, activation=activation)  # type: ignore[arg-type]


def test_policy_rejects_dimension_mismatches() -> None:
    bad_first = DenseLayer(weight=((0.0,) * 8,) * 2, bias=(0.0, 0.0), activation="tanh")
    head = DenseLayer(weight=((0.0, 0.0),) * 2, bias=(0.0, 0.0), activation="identity")
    with pytest.raises(ValueError, match="layer 0 expects 8 inputs but receives 9"):
        MlpPolicy("p", 1, _SPEC, (bad_first,), head, (RlAction.HOLD, RlAction.BUY))

    with pytest.raises(ValueError, match="actions must be 2 distinct"):
        _embedded_policy(actions=(RlAction.HOLD, RlAction.HOLD))
    with pytest.raises(ValueError, match="actions must be 2 distinct"):
        _embedded_policy(actions=ENV_ACTION_ORDER)


def test_action_head_must_be_raw_logits() -> None:
    good = _embedded_policy()
    tanh_head = DenseLayer(weight=_W2, bias=_B2, activation="tanh")
    with pytest.raises(ValueError, match="identity"):
        MlpPolicy("p", 1, _SPEC, good.hidden, tanh_head, good.actions)


def test_logits_reject_wrong_length_and_non_finite_observations() -> None:
    policy = _embedded_policy()
    with pytest.raises(ValueError, match="9 floats"):
        policy.logits((0.0,) * 8)
    with pytest.raises(ValueError, match="finite"):
        policy.act((math.nan,) + (0.0,) * 8)


@pytest.mark.parametrize("version", [0, -1, True, "1"])
def test_schema_version_must_be_a_positive_int(version: object) -> None:
    good = _embedded_policy()
    with pytest.raises(ValueError, match="schema_version"):
        MlpPolicy("p", version, _SPEC, good.hidden, good.action_head, good.actions)  # type: ignore[arg-type]
