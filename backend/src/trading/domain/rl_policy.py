"""The torch-free RL policy: a plain-float MLP forward pass (task-rl-09; program doc §3.4).

PURE domain code: stdlib only (no numpy, no torch), zero I/O, no clock, NO RANDOMNESS. This is the
DEPLOYABLE half of a trained agent: ``rl/export.py`` copies the weights of SB3's default
``MlpPolicy`` actor into an :class:`MlpPolicy`, ``infrastructure/rl/policy_store.py`` round-trips
it through a small JSON file, and any consumer (scoring, the demo bundle, the in-browser
TypeScript mirror) runs it without the ``[rl]`` extra.

**The inference path** (SB3 2.x, ``ActorCriticPolicy`` with a 1-D ``Box`` observation and a
``Discrete`` action space): ``FlattenExtractor`` is a no-op on a 1-D float observation, the actor is
``mlp_extractor.policy_net`` (``Sequential(Linear, act, Linear, act, ...)``) and ``action_net`` is a
final ``Linear``; ``predict(obs, deterministic=True)`` is ``CategoricalDistribution.mode()`` — the
argmax of those logits. So::

    logits = W_k @ act(... act(W_1 @ obs + b_1) ...) + b_k
    action = actions[argmax(logits)]

**Deterministic by contract.** :meth:`MlpPolicy.act` is the argmax ONLY; it never samples. SB3's
``predict`` defaults to ``deterministic=False`` (a sample from the categorical); sampling would
make a replay depend on an RNG and break the "same inputs => same decision" property the event log
and every evaluation rely on. Ties (exactly equal logits) resolve to the LOWEST index, as
``torch.argmax`` / ``numpy.argmax`` do.

**Input contract: round observations to float32 first.** The policy was trained, and its parity
proven, on float32 observations (the env's ``Box(..., float32)``; SB3 casts with ``obs.float()``).
A consumer that builds the observation in float64 (``rl_features.build_observation``) should round
each component to float32 BEFORE calling :meth:`MlpPolicy.logits` / :meth:`MlpPolicy.act`
(``numpy.float32`` / ``struct`` round-trip in Python, ``Math.fround`` in TypeScript) — the exact
inputs the parity gate checked. Unrounded float64 inputs differ by <= 1 float32 ulp per feature,
which can flip only a genuine near-tie. This module does not round for you: it accepts any finite
floats, so tests and hand-built nets can use exact values.

**Arithmetic, specified for a mirror.** Everything is IEEE float64 (SB3 runs in float32, so the
two agree only up to float32 rounding — the export parity gate requires every logit within
``1e-3`` of SB3's and tolerates an action mismatch only when the two competing logits are within
``1e-4`` on BOTH sides, a genuine tie). Each output unit is
``(sum over j, left to right, of weight[i][j] * x[j]) + bias[i]``, then the activation: ``tanh``
(``math.tanh``), ``relu`` (``max(0.0, v)``) or ``identity``. ``tanh`` is NOT guaranteed
bit-identical across libm / V8 (~1 ulp, like ``ln`` in ``rl_features``), so cross-language parity
checks use a tolerance, never bit equality.

If the net ever outgrows a plain MLP, the documented fallback is ONNX + ``onnxruntime`` (an export
of the full torch graph). It is deliberately NOT a dependency: the stdlib MLP keeps the deployable
artifact free of every optional package.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, get_args

from trading.domain.rl_features import FeatureSpec

Activation = Literal["tanh", "relu", "identity"]
ACTIVATIONS: Final[frozenset[str]] = frozenset(get_args(Activation))


class RlAction(StrEnum):
    """What the policy asks for at a decision. The exported index order is the env's
    ``Discrete(3)``: ``0 = HOLD``, ``1 = BUY``, ``2 = SELL`` (:data:`ENV_ACTION_ORDER`)."""

    SELL = "sell"
    HOLD = "hold"
    BUY = "buy"


# Index -> action for the single-ticker env's Discrete(3) (rl/env.py ACTION_HOLD/BUY/SELL; the
# export asserts the equality, since this module may not import trading.rl).
ENV_ACTION_ORDER: Final[tuple[RlAction, ...]] = (RlAction.HOLD, RlAction.BUY, RlAction.SELL)


def _finite_float(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite, got {result}")
    return result


@dataclass(frozen=True, slots=True)
class DenseLayer:
    """One affine layer + activation. ``weight`` is ``[out][in]``, row-major — torch's
    ``Linear.weight`` layout — and ``bias`` has ``out`` entries. Values are normalized to tuples of
    finite floats on construction (lists are accepted)."""

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    activation: Activation

    def __post_init__(self) -> None:
        weight: object = self.weight
        if not isinstance(weight, tuple | list) or not weight:
            raise ValueError("weight must be a non-empty [out][in] matrix")
        rows: list[tuple[float, ...]] = []
        for i, row in enumerate(weight):
            if not isinstance(row, tuple | list) or not row:
                raise ValueError(f"weight row {i} must be a non-empty sequence")
            rows.append(tuple(_finite_float(v, f"weight[{i}][{j}]") for j, v in enumerate(row)))
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("weight rows must all have the same length")
        bias: object = self.bias
        if not isinstance(bias, tuple | list) or len(bias) != len(rows):
            raise ValueError(f"bias must have {len(rows)} entries (one per weight row)")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}"
            )
        object.__setattr__(self, "weight", tuple(rows))
        object.__setattr__(
            self, "bias", tuple(_finite_float(v, f"bias[{i}]") for i, v in enumerate(bias))
        )

    @property
    def in_features(self) -> int:
        return len(self.weight[0])

    @property
    def out_features(self) -> int:
        return len(self.weight)

    def forward(self, x: Sequence[float]) -> tuple[float, ...]:
        """``activation(weight @ x + bias)`` in float64 (see the module docstring's order)."""
        if len(x) != self.in_features:
            raise ValueError(f"layer expects {self.in_features} inputs, got {len(x)}")
        out: list[float] = []
        for row, b in zip(self.weight, self.bias, strict=True):
            total = 0.0
            for w, v in zip(row, x, strict=True):
                total += w * v
            out.append(_activate(total + b, self.activation))
        return tuple(out)


def _activate(value: float, activation: Activation) -> float:
    if activation == "tanh":
        return math.tanh(value)
    if activation == "relu":
        return value if value > 0.0 else 0.0
    return value


@dataclass(frozen=True, slots=True)
class MlpPolicy:
    """A deterministic MLP policy: ``hidden`` (``mlp_extractor.policy_net``) then ``action_head``
    (``action_net``, identity activation); ``actions[i]`` is the action for logit ``i``.
    ``features`` is the observation spec the net was trained on (``rl_features``)."""

    policy_id: str
    schema_version: int
    features: FeatureSpec
    hidden: tuple[DenseLayer, ...]
    action_head: DenseLayer
    actions: tuple[RlAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a non-empty string")
        version: object = self.schema_version
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f"schema_version must be a positive int, got {version!r}")
        if not isinstance(self.features, FeatureSpec):
            raise ValueError("features must be a FeatureSpec")
        hidden: object = self.hidden
        if not isinstance(hidden, tuple | list) or not all(
            isinstance(layer, DenseLayer) for layer in hidden
        ):
            raise ValueError("hidden must be a sequence of DenseLayer")
        object.__setattr__(self, "hidden", tuple(hidden))
        if not isinstance(self.action_head, DenseLayer):
            raise ValueError("action_head must be a DenseLayer")
        if self.action_head.activation != "identity":
            raise ValueError("action_head must have the identity activation (raw logits)")
        width = len(self.features.names)
        for i, layer in enumerate((*self.hidden, self.action_head)):
            if layer.in_features != width:
                raise ValueError(
                    f"layer {i} expects {layer.in_features} inputs but receives {width}"
                )
            width = layer.out_features
        actions: object = self.actions
        if not isinstance(actions, tuple | list) or not all(
            isinstance(a, RlAction) for a in actions
        ):
            raise ValueError("actions must be a sequence of RlAction")
        if len(actions) != self.action_head.out_features or len(set(actions)) != len(actions):
            raise ValueError(
                f"actions must be {self.action_head.out_features} distinct RlAction values, "
                f"got {actions!r}"
            )
        object.__setattr__(self, "actions", tuple(actions))

    @property
    def obs_dim(self) -> int:
        return len(self.features.names)

    def logits(self, observation: Sequence[float]) -> tuple[float, ...]:
        """The raw action logits for one observation (float64; finite inputs only)."""
        if len(observation) != self.obs_dim:
            raise ValueError(f"observation must have {self.obs_dim} floats, got {len(observation)}")
        x: tuple[float, ...] = tuple(
            _finite_float(v, f"observation[{i}]") for i, v in enumerate(observation)
        )
        for layer in self.hidden:
            x = layer.forward(x)
        return self.action_head.forward(x)

    def action_index(self, observation: Sequence[float]) -> int:
        """``argmax(logits)``; an exact tie resolves to the lowest index. Never samples."""
        values = self.logits(observation)
        best = 0
        for i in range(1, len(values)):
            if values[i] > values[best]:
                best = i
        return best

    def act(self, observation: Sequence[float]) -> RlAction:
        """The deterministic action (argmax of the logits). NEVER samples."""
        return self.actions[self.action_index(observation)]
