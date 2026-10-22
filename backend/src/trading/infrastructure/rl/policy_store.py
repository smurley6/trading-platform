"""Read / write the deployable torch-free policy artifact (task-rl-09; program doc §3.4).

The JSON file is the DEPLOYABLE form of a trained agent: an :class:`~trading.domain.rl_policy.
MlpPolicy` (feature spec, layer weights, action mapping) plus a free-form ``meta`` block of
provenance (training run, dataset, splits, library versions, parity report). Pure stdlib — no
numpy, no torch — so a server or the demo bundle reads it with zero optional dependencies. The SB3
``.zip`` stays the RETRAINING artifact (it needs the ``[rl]`` extra) and is never shipped.

**Format (schema v1).** One JSON object with exactly the keys ``schema_version``, ``kind``
(``"rl.policy.mlp"``), ``policy_id``, ``features`` (``FeatureSpec.to_dict``), ``hidden`` (list of
layers), ``action_head`` (a layer), ``actions`` (``RlAction`` values, index order) and ``meta``.
A layer is ``{"weight": [[...] x out], "bias": [...], "activation": "tanh"|"relu"|"identity"}``.
Bytes are deterministic: sorted keys, compact separators, Python's shortest round-trip float
``repr`` (so every float32 weight, widened exactly to float64, reads back bit-identically), strict
JSON (NaN / Infinity refused both ways), UTF-8, one trailing newline. Equal policy + equal meta =>
equal bytes => equal sha256.

:func:`load_policy` fails closed (:class:`PolicyArtifactError`) on anything unexpected: a different
``schema_version``, an unknown ``kind``, a missing / extra key, a malformed layer, or dimensions
that do not chain from the feature count to the action count.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, NoReturn

from trading.domain.rl_features import FeatureSpec
from trading.domain.rl_policy import Activation, DenseLayer, MlpPolicy, RlAction

POLICY_SCHEMA_VERSION: Final[int] = 1
POLICY_KIND: Final = "rl.policy.mlp"
_KEYS: Final = frozenset(
    {"schema_version", "kind", "policy_id", "features", "hidden", "action_head", "actions", "meta"}
)
_LAYER_KEYS: Final = frozenset({"weight", "bias", "activation"})
_ACTIVATION_BY_NAME: Final[dict[str, Activation]] = {
    "tanh": "tanh",
    "relu": "relu",
    "identity": "identity",
}


class PolicyArtifactError(RuntimeError):
    """A policy artifact cannot be written or read as schema v1 (fails closed, never guesses)."""


def _layer_to_dict(layer: DenseLayer) -> dict[str, object]:
    return {
        "weight": [list(row) for row in layer.weight],
        "bias": list(layer.bias),
        "activation": layer.activation,
    }


def policy_to_json(policy: MlpPolicy, *, meta: Mapping[str, object]) -> bytes:
    """The artifact's exact bytes (deterministic; see the module docstring)."""
    if policy.schema_version != POLICY_SCHEMA_VERSION:
        raise PolicyArtifactError(
            f"cannot write schema_version {policy.schema_version}; this store writes "
            f"v{POLICY_SCHEMA_VERSION}"
        )
    if not isinstance(meta, Mapping) or not all(isinstance(k, str) for k in meta):
        raise PolicyArtifactError("meta must be a mapping with string keys")
    document = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "kind": POLICY_KIND,
        "policy_id": policy.policy_id,
        "features": policy.features.to_dict(),
        "hidden": [_layer_to_dict(layer) for layer in policy.hidden],
        "action_head": _layer_to_dict(policy.action_head),
        "actions": [action.value for action in policy.actions],
        "meta": dict(meta),
    }
    try:
        text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PolicyArtifactError(f"policy artifact is not strict JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def write_policy(path: Path, policy: MlpPolicy, *, meta: Mapping[str, object]) -> str:
    """Write the artifact atomically (temp file + rename, parents created); return the sha256 hex
    of the bytes written."""
    data = policy_to_json(policy, meta=meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def _fail(message: str) -> NoReturn:
    raise PolicyArtifactError(message)


def _reject_constant(token: str) -> NoReturn:
    _fail(f"non-finite number {token} in policy artifact")


def _layer_from(raw: object, where: str) -> DenseLayer:
    if not isinstance(raw, dict) or set(raw) != _LAYER_KEYS:
        _fail(f"{where} must be an object with exactly the keys {sorted(_LAYER_KEYS)}")
    weight, bias, activation = raw["weight"], raw["bias"], raw["activation"]
    if not isinstance(weight, list) or not all(isinstance(row, list) for row in weight):
        _fail(f"{where}.weight must be a list of lists")
    if not isinstance(bias, list):
        _fail(f"{where}.bias must be a list")
    if not isinstance(activation, str) or activation not in _ACTIVATION_BY_NAME:
        _fail(
            f"{where}.activation must be one of {sorted(_ACTIVATION_BY_NAME)}, got {activation!r}"
        )
    try:
        return DenseLayer(
            weight=tuple(tuple(row) for row in weight),
            bias=tuple(bias),
            activation=_ACTIVATION_BY_NAME[activation],
        )
    except ValueError as exc:
        raise PolicyArtifactError(f"{where}: {exc}") from exc


def policy_from_json(data: bytes) -> MlpPolicy:
    """Parse + validate artifact bytes (the inverse of :func:`policy_to_json`; ``meta`` is not
    interpreted). Raises :class:`PolicyArtifactError`."""
    try:
        document: object = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyArtifactError(f"policy artifact is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        _fail("policy artifact must be a JSON object")
    version = document.get("schema_version")
    # type() is int: a bool (True == 1) or a float (1.0 == 1) is not schema version 1.
    if type(version) is not int or version != POLICY_SCHEMA_VERSION:
        _fail(f"unsupported schema_version {version!r}; expected {POLICY_SCHEMA_VERSION}")
    if set(document) != _KEYS:
        _fail(f"policy artifact keys must be {sorted(_KEYS)}, got {sorted(document)}")
    if document["kind"] != POLICY_KIND:
        _fail(f"kind must be {POLICY_KIND!r}, got {document['kind']!r}")
    policy_id, hidden, actions = document["policy_id"], document["hidden"], document["actions"]
    if not isinstance(policy_id, str):
        _fail("policy_id must be a string")
    if not isinstance(hidden, list):
        _fail("hidden must be a list of layers")
    if not isinstance(actions, list) or not all(isinstance(a, str) for a in actions):
        _fail("actions must be a list of strings")
    if not isinstance(document["meta"], dict):
        _fail("meta must be an object")
    features_raw = document["features"]
    if not isinstance(features_raw, dict):
        _fail("features must be an object")
    try:
        features = FeatureSpec.from_dict(features_raw)
        return MlpPolicy(
            policy_id=policy_id,
            schema_version=POLICY_SCHEMA_VERSION,
            features=features,
            hidden=tuple(_layer_from(layer, f"hidden[{i}]") for i, layer in enumerate(hidden)),
            action_head=_layer_from(document["action_head"], "action_head"),
            actions=tuple(RlAction(a) for a in actions),
        )
    except ValueError as exc:
        raise PolicyArtifactError(f"invalid policy artifact: {exc}") from exc


def load_policy(path: Path) -> MlpPolicy:
    """Read + validate a schema-v1 policy artifact. Raises :class:`PolicyArtifactError` (including
    when the file cannot be read)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PolicyArtifactError(f"cannot read policy artifact {path}: {exc}") from exc
    return policy_from_json(data)
