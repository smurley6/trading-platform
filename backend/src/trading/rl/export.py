"""SB3 PPO -> torch-free :class:`~trading.domain.rl_policy.MlpPolicy`, plus the parity gate
(task-rl-09; docs/RL-SLEEVE-PROGRAM.md §3.4).

HEAVY half of ``trading.rl`` and, with ``rl/train.py``, the only module that touches
``stable_baselines3`` / ``torch`` (the optional ``[rl]`` extra) — imported INSIDE the functions that
need them, so this module itself loads with only gymnasium + numpy (``[dev]``). The output is the
deployable artifact; the ``.zip`` stays the retraining one and is never shipped.

**What is exported** (SB3 2.x ``ActorCriticPolicy``): the actor path only —
``mlp_extractor.policy_net`` (``Sequential(Linear, Tanh, Linear, Tanh)`` for the default
``net_arch=[64, 64]``: state_dict keys ``mlp_extractor.policy_net.{0,2}.{weight,bias}``) then
``action_net`` (``action_net.{weight,bias}``). The value head is not needed for deterministic
inference and is dropped.

**Refuses what the stdlib forward pass cannot reproduce** (:class:`ExportError`): a features
extractor that is not ``FlattenExtractor`` (anything else transforms the observation before the
MLP); a non-``Box`` / non-1-D / non-float32 observation space or a feature count that disagrees with
the ``FeatureSpec``; anything but ``Discrete(3)`` from 0 with a ``CategoricalDistribution``; a
``policy_net`` module that is not exactly ``nn.Linear`` or a supported activation (``nn.Tanh`` ->
``tanh``, ``nn.ReLU`` -> ``relu``), or an activation not directly after a ``Linear``. Exact type
checks, so a subclass with a custom ``forward`` cannot slip through.

**The parity gate.** :func:`assert_parity` runs every observation through BOTH
``model.predict(obs, deterministic=True)`` (float32, argmax of the logits) and
:meth:`MlpPolicy.act` (float64). The policy is fed the SAME float32-rounded values SB3 sees, so the
comparison isolates the network arithmetic. It guarantees, over the given observations:

1. **The logits match**, not just the actions: ``max |SB3 float32 logit - stdlib float64 logit|``
   must be ``< LOGIT_TOL`` (``1e-3``; float32 rounding of a 9 -> 64 -> 64 -> 3 net is ~1e-6).
   Action agreement alone is not enough — a dead or argmin policy can "agree" by near-ties.
2. **Every action mismatch is a genuine two-sided tie**: the two competing actions' logits are
   within :data:`TIE_MARGIN` (``1e-4``) in BOTH the stdlib float64 logits AND SB3's float32 logits.
   Only then can float32 rounding have flipped the argmax; such mismatches are counted as ties.

A violation of either raises :class:`ParityError` carrying the full :class:`ParityReport`; the
export job refuses to write the artifact on it. (Given 1, the stdlib and SB3 logits are within
``1e-3`` everywhere checked, so a mismatch whose margins are both ``< 1e-4`` is the only kind that
can pass.)

**Fallback.** If the net ever outgrows an MLP, the documented route is ONNX + ``onnxruntime``; that
dependency is deliberately not added.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from gymnasium import spaces

from trading.domain.rl_features import FeatureSpec
from trading.domain.rl_policy import (
    ENV_ACTION_ORDER,
    Activation,
    DenseLayer,
    MlpPolicy,
    RlAction,
)
from trading.infrastructure.rl.policy_store import POLICY_SCHEMA_VERSION
from trading.rl.baselines import BUY, HOLD, SELL, Policy, PolicyFactory
from trading.rl.train import require_rl_extra

if TYPE_CHECKING:
    from stable_baselines3 import PPO

# A float32/float64 disagreement within this logit gap (on BOTH sides) is a genuine tie (§3.4).
TIE_MARGIN: Final = 1e-4
# Hard cap on max |SB3 logit - stdlib logit| (observed ~1e-6 on the real seed-0 model).
LOGIT_TOL: Final = 1e-3
# How many disagreeing observation indices a report keeps (for the operator; counts are complete).
_MAX_REPORTED: Final = 20


class ExportError(ValueError):
    """The model cannot be exported to a stdlib MLP exactly (fails closed, never approximates)."""


@dataclass(frozen=True, slots=True)
class ParityReport:
    """SB3 ``predict(deterministic=True)`` vs ``MlpPolicy.act`` over a set of observations.

    ``n_agree + n_ties + n_disagreements == n_observations``. ``n_ties`` are mismatches whose two
    competing logits are within ``tie_margin`` in BOTH the stdlib float64 and SB3 float32 logits;
    ``n_disagreements`` are every other mismatch. ``max_abs_logit_diff`` is the largest |float32
    SB3 logit - float64 stdlib logit| seen (expected ~1e-6: float32 rounding). The gate passes
    (``ok``) only with zero disagreements AND ``max_abs_logit_diff < logit_tol``."""

    n_observations: int
    n_agree: int
    n_ties: int
    n_disagreements: int
    tie_margin: float
    max_abs_logit_diff: float
    logit_tol: float
    tie_indices: tuple[int, ...] = field(default=())
    disagreement_indices: tuple[int, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return (
            self.n_observations > 0
            and self.n_disagreements == 0
            and self.max_abs_logit_diff < self.logit_tol
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_observations": self.n_observations,
            "n_agree": self.n_agree,
            "n_ties": self.n_ties,
            "n_disagreements": self.n_disagreements,
            "tie_margin": self.tie_margin,
            "max_abs_logit_diff": self.max_abs_logit_diff,
            "logit_tol": self.logit_tol,
        }


class ParityError(AssertionError):
    """SB3 and the exported policy disagree beyond a float32 tie: the artifact must not ship."""

    def __init__(self, report: ParityReport) -> None:
        super().__init__(
            f"parity FAILED: {report.n_disagreements} non-tie disagreement(s) over "
            f"{report.n_observations} observations (first at indices "
            f"{list(report.disagreement_indices)}); ties={report.n_ties}; max |logit diff| "
            f"{report.max_abs_logit_diff:.3g} (limit {report.logit_tol:g})"
        )
        self.report = report


# --- export -------------------------------------------------------------------------------------


def _check_action_order() -> None:
    expected = {HOLD: RlAction.HOLD, BUY: RlAction.BUY, SELL: RlAction.SELL}
    if dict(enumerate(ENV_ACTION_ORDER)) != expected:
        raise ExportError(f"ENV_ACTION_ORDER {ENV_ACTION_ORDER} disagrees with the env's indices")


def _float32_rows(tensor: Any, name: str) -> list[Any]:
    """A float32 parameter as nested Python floats (float32 -> float64 widening is exact)."""
    import torch

    if tensor.dtype != torch.float32:
        raise ExportError(f"{name} must be float32, got {tensor.dtype}")
    values: list[Any] = tensor.detach().cpu().tolist()
    return values


def _dense(module: Any, key: str, state: dict[str, Any], activation: Activation) -> DenseLayer:
    import torch

    for suffix, tensor in (("weight", module.weight), ("bias", module.bias)):
        stored = state.get(f"{key}.{suffix}")
        if stored is None or not torch.equal(stored, tensor):
            raise ExportError(
                f"state_dict key {key}.{suffix} is missing or differs from the module"
            )
    weight = _float32_rows(module.weight, f"{key}.weight")
    bias = _float32_rows(module.bias, f"{key}.bias")
    try:
        return DenseLayer(
            weight=tuple(tuple(row) for row in weight), bias=tuple(bias), activation=activation
        )
    except ValueError as exc:
        raise ExportError(f"{key}: {exc}") from exc


def sb3_to_mlp_policy(model: "PPO", *, policy_id: str, spec: FeatureSpec) -> MlpPolicy:
    """Copy a PPO ``MlpPolicy`` actor into a torch-free :class:`MlpPolicy` (schema v1).

    Raises :class:`ExportError` for anything the stdlib forward pass cannot reproduce exactly (see
    the module docstring). Checks the result with :func:`assert_parity` before shipping it."""
    require_rl_extra()
    from stable_baselines3.common.distributions import CategoricalDistribution
    from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor
    from torch import nn

    _check_action_order()
    policy = model.policy
    for name in ("features_extractor", "pi_features_extractor"):
        extractor = getattr(policy, name, None)
        if type(extractor) is not FlattenExtractor:
            raise ExportError(
                f"policy.{name} is {type(extractor).__name__}, not FlattenExtractor: the stdlib "
                "forward pass cannot reproduce a feature transform"
            )
    obs_space = model.observation_space
    if (
        not isinstance(obs_space, spaces.Box)
        or obs_space.shape != (len(spec.names),)
        or obs_space.dtype != np.float32
    ):
        raise ExportError(
            f"observation space must be a float32 Box of shape ({len(spec.names)},), got "
            f"{obs_space!r}"
        )
    act_space = model.action_space
    if (
        not isinstance(act_space, spaces.Discrete)
        or int(act_space.n) != len(ENV_ACTION_ORDER)
        or int(act_space.start) != 0
    ):
        raise ExportError(f"action space must be Discrete(3) from 0, got {act_space!r}")
    if not isinstance(policy.action_dist, CategoricalDistribution):
        raise ExportError(
            f"action distribution must be Categorical, got {type(policy.action_dist).__name__}"
        )
    extractor_net = policy.mlp_extractor
    if type(extractor_net) is not MlpExtractor or not isinstance(
        extractor_net.policy_net, nn.Sequential
    ):
        raise ExportError("policy.mlp_extractor must be SB3's MlpExtractor with a Sequential actor")

    supported: dict[type[Any], Activation] = {nn.Tanh: "tanh", nn.ReLU: "relu"}
    state = dict(policy.state_dict())
    hidden: list[DenseLayer] = []
    pending: tuple[str, Any] | None = None
    for index, module in enumerate(extractor_net.policy_net):
        key = f"mlp_extractor.policy_net.{index}"
        if type(module) is nn.Linear:
            if pending is not None:
                hidden.append(_dense(pending[1], pending[0], state, "identity"))
            pending = (key, module)
        elif type(module) in supported:
            if pending is None:
                raise ExportError(f"{key} ({type(module).__name__}) does not follow a Linear")
            hidden.append(_dense(pending[1], pending[0], state, supported[type(module)]))
            pending = None
        else:
            raise ExportError(
                f"{key} is an unsupported module {type(module).__name__}; the stdlib forward pass "
                f"supports Linear followed by one of {sorted(t.__name__ for t in supported)}"
            )
    if pending is not None:
        hidden.append(_dense(pending[1], pending[0], state, "identity"))
    if type(policy.action_net) is not nn.Linear:
        raise ExportError(f"action_net must be nn.Linear, got {type(policy.action_net).__name__}")
    head = _dense(policy.action_net, "action_net", state, "identity")

    try:
        return MlpPolicy(
            policy_id=policy_id,
            schema_version=POLICY_SCHEMA_VERSION,
            features=spec,
            hidden=tuple(hidden),
            action_head=head,
            actions=ENV_ACTION_ORDER,
        )
    except ValueError as exc:
        raise ExportError(f"exported layers do not form a valid MlpPolicy: {exc}") from exc


# --- parity -------------------------------------------------------------------------------------


def sb3_logits(model: "PPO", observation: Sequence[float]) -> tuple[float, ...]:
    """SB3's raw float32 action logits for ONE observation — the same path ``predict`` takes
    (``preprocess_obs`` -> ``pi_features_extractor`` -> ``policy_net`` -> ``action_net``)."""
    require_rl_extra()
    import torch
    from stable_baselines3.common.preprocessing import preprocess_obs

    policy = model.policy
    obs = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
    with torch.no_grad():
        prepared = preprocess_obs(obs, policy.observation_space, policy.normalize_images)
        features = policy.pi_features_extractor(prepared)
        logits = policy.action_net(policy.mlp_extractor.forward_actor(features))
    return tuple(float(v) for v in logits.reshape(-1).tolist())


def assert_parity(
    model: "PPO", policy: MlpPolicy, observations: Sequence[Sequence[float]]
) -> ParityReport:
    """Compare ``model.predict(obs, deterministic=True)`` with ``policy.act(obs)`` on every
    observation (each rounded to float32 first — exactly what SB3 sees — and fed to both).

    Returns the :class:`ParityReport` when ``max |logit diff| < LOGIT_TOL`` and every action
    mismatch is a two-sided tie (see the module docstring); raises :class:`ParityError` (carrying
    the report) otherwise. Raises :class:`ExportError` if there are no observations, or if SB3's
    ``predict`` picks an action other than the argmax of SB3's own logits by more than a float32
    tie (the comparison would then be meaningless). A ``predict`` / own-argmax split WITHIN a tie
    is not an error: it goes through the same two-sided tie rule as any other mismatch."""
    if not observations:
        raise ExportError("parity needs at least one observation")
    n_agree = n_ties = 0
    tie_indices: list[int] = []
    bad_indices: list[int] = []
    n_bad = 0
    max_diff = 0.0
    for i, raw in enumerate(observations):
        obs32 = np.asarray(raw, dtype=np.float32)
        if obs32.shape != (policy.obs_dim,):
            raise ExportError(f"observation {i} has shape {obs32.shape}, need ({policy.obs_dim},)")
        seen = tuple(float(v) for v in obs32.tolist())
        action, _state = model.predict(obs32, deterministic=True)
        theirs = int(np.asarray(action).reshape(-1)[0])
        torch_logits = sb3_logits(model, seen)
        own_best = int(np.argmax(np.asarray(torch_logits, dtype=np.float32)))
        if abs(torch_logits[own_best] - torch_logits[theirs]) >= TIE_MARGIN:
            raise ExportError(f"observation {i}: SB3 predict disagrees with its own logits")
        ours_logits = policy.logits(seen)
        max_diff = max(
            max_diff,
            max(abs(a - b) for a, b in zip(ours_logits, torch_logits, strict=True)),
        )
        ours = policy.action_index(seen)
        if ours == theirs:
            n_agree += 1
        elif (
            abs(ours_logits[ours] - ours_logits[theirs]) < TIE_MARGIN
            and abs(torch_logits[ours] - torch_logits[theirs]) < TIE_MARGIN
        ):
            n_ties += 1
            if len(tie_indices) < _MAX_REPORTED:
                tie_indices.append(i)
        else:
            n_bad += 1
            if len(bad_indices) < _MAX_REPORTED:
                bad_indices.append(i)
    if not math.isfinite(max_diff):
        raise ExportError(f"non-finite logit difference {max_diff}")
    report = ParityReport(
        n_observations=len(observations),
        n_agree=n_agree,
        n_ties=n_ties,
        n_disagreements=n_bad,
        tie_margin=TIE_MARGIN,
        max_abs_logit_diff=max_diff,
        logit_tol=LOGIT_TOL,
        tie_indices=tuple(tie_indices),
        disagreement_indices=tuple(bad_indices),
    )
    if not report.ok:
        raise ParityError(report)
    return report


def sample_observations(space: object, n: int, *, seed: int) -> list[tuple[float, ...]]:
    """``n`` observations drawn uniformly from a ``Box`` observation ``space`` (seeded with
    ``space.seed(seed)``, so reproducible), as Python floats (float32 values widened exactly)."""
    if not isinstance(space, spaces.Box):
        raise ExportError(f"can only sample a Box observation space, got {space!r}")
    if n < 1:
        raise ExportError(f"n must be >= 1, got {n}")
    space.seed(seed)
    return [tuple(float(v) for v in space.sample().tolist()) for _ in range(n)]


# --- the scoring bridge ---------------------------------------------------------------------------


def mlp_policy_factory(policy: MlpPolicy) -> PolicyFactory:
    """Wrap an exported policy as ``rl/baselines.py``'s ``PolicyFactory`` (the same contract as
    ``rl/train.py::policy_factory`` for the SB3 model): ``(observation, history) -> int`` in
    ``Discrete(3)``. ``history`` and the seed are ignored; the action is the argmax, never a
    sample."""
    _check_action_order()
    if policy.actions != ENV_ACTION_ORDER:
        raise ExportError(f"policy actions {policy.actions} are not the env's {ENV_ACTION_ORDER}")

    def factory(seed: int) -> Policy:
        del seed

        def act(observation: Sequence[float], history: Sequence[float]) -> int:
            del history
            return policy.action_index(observation)

        return act

    return factory
