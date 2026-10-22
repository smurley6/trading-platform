"""Train a PPO agent on the single-ticker env (task-rl-08; docs/RL-SLEEVE-PROGRAM.md §4.9, §3.1).

HEAVY half of ``trading.rl`` and, with the future ``rl/export.py``, the ONLY module that touches
``stable_baselines3`` / ``torch`` (the optional ``[rl]`` extra). Deliberately thin: one seeded
``PPO("MlpPolicy", ...)`` over :class:`~trading.rl.env.SingleTickerTradingEnv`, saved as SB3's
``.zip`` plus a sidecar JSON manifest. No SB3 type is subclassed (``disallow_subclassing_any``);
``VecNormalize`` is never used (the env's fixed feature scales are the only normalization).

**Import posture.** SB3 and torch are imported INSIDE the functions that need them, so this
module imports with only gymnasium + numpy (both in ``[dev]``): the manifest schema and the
SB3 -> ``Policy`` bridge are covered by the default ``pytest`` without the extra, and
``trading.jobs.rl_train`` can report a missing extra with a clear message
(:func:`require_rl_extra`) instead of an ImportError at module load.

**Hyperparameters** (§4.9, unchanged here — do NOT tune in this module): SB3 defaults except
exactly three deviations — ``n_steps=1024`` (the default 2048 is ~8 episodes per rollout at
``episode_length=252``), ``ent_coef=0.01`` (stops the discrete policy collapsing to always-HOLD)
and an explicit ``seed``. ``device="cpu"``. ``Monitor(info_keywords=("equity", "turnover",
"n_trades"))`` wraps the env BEFORE any other wrapper, then a one-env ``DummyVecEnv``.

**Seeding / determinism.** ``PPO(seed=s)`` calls SB3's ``set_random_seed`` (Python ``random``,
numpy's global RNG — minibatch shuffling — and ``torch.manual_seed`` — weight init and action
sampling), seeds the action space, and seeds the env's FIRST reset with ``s`` (gymnasium
``np_random`` -> every random episode start). Training also pins ``torch.set_num_threads(1)`` for
its duration (restored afterwards) so intra-op reduction order cannot vary with the host's core
count. Same seed + same platform + same library versions => bit-identical weights (pinned by
``tests/unit/rl/test_train_smoke.py``). Residual nondeterminism, honestly: bit-identity is NOT
promised across CPU architectures / BLAS builds / torch versions (float summation order inside
MKL/OpenBLAS/oneDNN kernels may differ), and the manifest's ``wall_seconds`` / ``created_at`` are
wall-clock by nature. The manifest records the exact versions so a run is reproducible on a
matching stack (program doc R11).

**Two hashes, and which one reproduces.** ``policy_sha256`` (:func:`policy_state_sha256`) hashes
the policy WEIGHTS — a canonical serialization of ``model.policy.state_dict()`` (sorted names,
dtype, shape, little-endian bytes) — and IS stable across same-seed runs on a matching stack:
check a rerun against it. ``model_sha256`` is only a FILE-INTEGRITY hash of the ``.zip`` and
changes on EVERY run, even when the weights are bit-identical, because SB3's zip embeds run-varying
content: the ``data`` member's wall-clock ``start_time``, the ``ep_info_buffer``'s Monitor
elapsed-time ``t`` fields, and cloudpickled class blobs (``policy_class``,
``rollout_buffer_class``) whose bytes are not stable run to run, plus per-entry zip timestamps on
``data`` / ``_stable_baselines3_version`` / ``system_info.txt``. (``policy.pth``,
``policy.optimizer.pth`` and ``pytorch_variables.pth`` are byte-identical across same-seed runs.)

**Artifacts are not committed by default.** ``backend/artifacts/`` is git-ignored so a training run
never leaves the tree dirty. The DELIBERATE exhibit artifacts are committed later, at task-rl-12
bundle time, with an explicit ``git add -f``. :func:`train_ppo` refuses to overwrite an existing
artifact stem unless ``overwrite=True`` (the job's ``--overwrite``).

**What the training reward is — and is not.** ``ep_rew_mean`` is the IN-SAMPLE mean episode return
(the env's x100 log reward) of the STOCHASTIC training policy over the most recent 100 episodes,
on the very bars it trains on. It shows the optimizer working (T0: "ep_rew_mean rises on the train
split"); it is never an evaluation result. Overfitting is structurally guaranteed (200k timesteps
over ~500 train bars is ~400 passes), and the manifest records that pass count.

**Untrained baseline** (§5.2, owned here since the task-rl-07 review): ``total_timesteps=0`` saves
the policy at initialization — byte-for-byte the same initial weights the seed's trained run starts
from — through the identical save / load / scoring path, so "PPO learned" can be separated from
"this env is trivially profitable".

**Scoring bridge.** :func:`policy_factory` wraps a loaded model as ``rl/baselines.py``'s
``PolicyFactory`` (``seed -> (observation, history) -> int``) calling
``predict(obs, deterministic=True)`` — ``CategoricalDistribution.mode()``, the argmax of the
logits, never a sample — so a trained (or untrained) policy replays through
``rl/scoring.py::rollout`` exactly like the baselines.
"""

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from trading.domain.rl_features import MAX_LOOKBACK
from trading.rl.bars import BarTable
from trading.rl.baselines import Policy, PolicyFactory
from trading.rl.env import N_ACTIONS, EnvConfig, SingleTickerTradingEnv

if TYPE_CHECKING:
    from stable_baselines3 import PPO

MANIFEST_SCHEMA_VERSION: Final = 1
MONITOR_INFO_KEYWORDS: Final = ("equity", "turnover", "n_trades")
# SB3's ``stats_window_size`` default: ep_rew_mean is the mean over the last 100 episodes.
EP_REW_WINDOW: Final = 100
# Distribution names whose exact versions every artifact records (program doc R11).
_VERSIONED_PACKAGES: Final = {
    "sb3": "stable_baselines3",
    "torch": "torch",
    "gymnasium": "gymnasium",
    "numpy": "numpy",
}
# Training pins torch to one intra-op thread (deterministic reduction order; also faster for a
# 64x64 MLP). Recorded in the manifest.
TORCH_NUM_THREADS: Final = 1
# Passed EXPLICITLY, as in program doc §4.9's PPO(...) call (they equal SB3 2.9's defaults, which
# are NOT deviations; passing them means a future SB3 default change cannot alter the run). Every
# value the manifest reports is read back from the constructed model (effective_hyperparameters).
GAE_LAMBDA: Final = 0.95
CLIP_RANGE: Final = 0.2

REQUIRED_MANIFEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "note",
        "symbol",
        "seed",
        "untrained",
        "model_file",
        "model_sha256",
        "policy_sha256",
        "requested_timesteps",
        "num_timesteps",
        "passes_over_train_bars",
        "wall_seconds",
        "created_at",
        "hyperparameters",
        "env",
        "train_bars",
        "seeding",
        "library_versions",
        "python_version",
        "training",
        "provenance",
    }
)
_IN_SAMPLE_NOTE: Final = (
    "IN-SAMPLE training diagnostic on the train split only (stochastic training policy, the bars "
    "it trained on). NOT an evaluation result; out-of-sample numbers come from the walk-forward "
    "evaluation (task-rl-10) via rl/scoring.py."
)


class TrainError(ValueError):
    """Training cannot run as configured (fails closed, never guesses)."""


class RlExtraMissingError(RuntimeError):
    """The optional ``[rl]`` extra (stable-baselines3 + torch) is not installed."""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """PPO hyperparameters for the first run (§4.9). SB3 defaults except ``n_steps``, ``ent_coef``
    and the explicit ``seed``; ``net_arch`` keeps SB3's default so task-rl-09's export is trivial.
    ``total_timesteps=0`` is the untrained (initialization) baseline."""

    total_timesteps: int = 200_000
    seed: int = 0
    n_steps: int = 1024  # default 2048 is ~8 episodes/rollout at episode_length=252
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    ent_coef: float = 0.01  # default 0.0 lets the discrete policy collapse to always-HOLD
    gamma: float = 0.99
    net_arch: tuple[int, ...] = (64, 64)  # keep the SB3 default: makes task-rl-09 export trivial

    def __post_init__(self) -> None:
        for name in ("total_timesteps", "seed", "n_steps", "batch_size", "n_epochs"):
            value: object = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TrainError(f"{name} must be an int, got {value!r}")
        if self.total_timesteps < 0:
            raise TrainError(f"total_timesteps must be >= 0, got {self.total_timesteps}")
        if self.seed < 0:
            raise TrainError(f"seed must be >= 0, got {self.seed}")
        for name in ("n_steps", "batch_size", "n_epochs"):
            if getattr(self, name) < 1:
                raise TrainError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.batch_size > self.n_steps:
            raise TrainError(
                f"batch_size ({self.batch_size}) must not exceed n_steps ({self.n_steps})"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise TrainError(f"learning_rate must be finite and > 0, got {self.learning_rate}")
        if not math.isfinite(self.ent_coef) or self.ent_coef < 0:
            raise TrainError(f"ent_coef must be finite and >= 0, got {self.ent_coef}")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise TrainError(f"gamma must be in (0, 1], got {self.gamma}")
        if not self.net_arch or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in self.net_arch
        ):
            raise TrainError(
                f"net_arch must be a non-empty tuple of positive ints, got {self.net_arch!r}"
            )

    @property
    def untrained(self) -> bool:
        """True for the §5.2 untrained baseline: the policy at initialization."""
        return self.total_timesteps == 0


@dataclass(frozen=True, slots=True)
class TrainResult:
    """One finished (or, at ``total_timesteps=0``, initialized) training run.

    ``final_ep_rew_mean`` is SB3's logged ``ep_rew_mean`` at the end of training (mean return of
    the last <= 100 episodes, IN-SAMPLE); None when no episode completed (the untrained baseline,
    or a run shorter than one episode). ``policy_sha256`` is the reproducible weights hash (see
    the module docstring); the ``.zip``'s own hash differs every run."""

    seed: int
    model_path: Path
    final_ep_rew_mean: float | None
    sb3_version: str
    torch_version: str
    gymnasium_version: str
    numpy_version: str
    manifest_path: Path
    num_timesteps: int
    wall_seconds: float
    untrained: bool
    policy_sha256: str


class _Predictor(Protocol):
    """The slice of an SB3 model the scoring bridge uses."""

    def predict(
        self,
        observation: npt.NDArray[np.float32],
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[Any, Any]: ...


# --- the [rl] extra ----------------------------------------------------------------------------


def require_rl_extra() -> None:
    """Raise :class:`RlExtraMissingError` (with the install command) unless SB3 + torch import."""
    try:
        importlib.import_module("stable_baselines3")
        importlib.import_module("torch")
    except ImportError as exc:
        raise RlExtraMissingError(
            f"the optional [rl] extra is not installed ({exc}). Install it with "
            '`pip install -e ".[dev,rl]"` (on Linux install the CPU torch wheel first: '
            "`pip install --index-url https://download.pytorch.org/whl/cpu torch`)."
        ) from exc


def library_versions() -> dict[str, str]:
    """Exact installed versions of SB3 / torch / gymnasium / numpy (``"absent"`` if missing)."""
    versions: dict[str, str] = {}
    for key, distribution in _VERSIONED_PACKAGES.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "absent"
    return versions


def policy_state_sha256(state: Mapping[str, Any]) -> str:
    """THE reproducible artifact hash: sha256 over a canonical serialization of a policy state
    dict — for each entry in sorted-name order, ``name``, dtype, shape, then the values as
    C-contiguous little-endian bytes. Accepts torch tensors (detached to CPU) or anything
    ``numpy.asarray`` takes, so it is torch-free for numpy inputs. Bit-identical weights <=> equal
    hash; independent of the ``.zip``'s run-varying metadata and of torch's pickle format."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        header = f"{name}\0{array.dtype.str}\0{list(array.shape)}\0"
        digest.update(header.encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def effective_hyperparameters(model: Any) -> dict[str, Any]:
    """The PPO settings READ BACK from a constructed model (never assumed from SB3's defaults),
    so a manifest cannot silently disagree with what actually trained."""
    policy = model.policy
    return {
        "learning_rate": float(model.learning_rate),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_vf": None if model.clip_range_vf is None else float(model.clip_range_vf(1.0)),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "normalize_advantage": bool(model.normalize_advantage),
        "target_kl": None if model.target_kl is None else float(model.target_kl),
        "use_sde": bool(model.use_sde),
        "net_arch": list(policy.net_arch),
        "activation_fn": policy.activation_fn.__name__,
        "features_extractor": type(policy.features_extractor).__name__,
        "ortho_init": bool(policy.ortho_init),
        "stats_window_size": int(model._stats_window_size),
        "device": str(model.device),
    }


def _check_effective(config: "TrainConfig", effective: Mapping[str, Any]) -> None:
    """Fail closed if SB3 did not apply a requested setting (e.g. a renamed kwarg)."""
    expected: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "ent_coef": config.ent_coef,
        "net_arch": list(config.net_arch),
        "stats_window_size": EP_REW_WINDOW,
        "device": "cpu",
    }
    drift = {k: (v, effective.get(k)) for k, v in expected.items() if effective.get(k) != v}
    if drift:
        raise TrainError(
            f"PPO did not apply the requested settings (requested, effective): {drift}"
        )


# --- naming + manifest (pure; covered without the extra) ---------------------------------------


def artifact_stem(symbol: str, run_date: date, seed: int, *, untrained: bool) -> str:
    """``{symbol}_ppo_{YYYYMMDD}_seed{n}`` (trained) or
    ``{symbol}_ppo_untrained_{YYYYMMDD}_seed{n}`` (``total_timesteps=0``), symbol lower-cased; the
    model is ``<stem>.zip`` and the manifest ``<stem>.json``."""
    if not symbol or not symbol.isalnum():
        raise TrainError(f"symbol must be a non-empty alphanumeric ticker, got {symbol!r}")
    kind = "ppo_untrained" if untrained else "ppo"
    return f"{symbol.lower()}_{kind}_{run_date:%Y%m%d}_seed{seed}"


def passes_over_bars(total_timesteps: int, n_bars: int) -> float:
    """How many times training walks the train bars: ``timesteps / n_bars`` (200k over ~500 bars
    is ~400 passes — overfitting is structurally guaranteed; always report it)."""
    if n_bars < 1:
        raise TrainError(f"n_bars must be >= 1, got {n_bars}")
    return total_timesteps / n_bars


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise TrainError(f"non-finite training statistic {value!r}")
    return float(value)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def training_summary(
    episode_returns: Sequence[float],
    episode_lengths: Sequence[int],
    final_window: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The manifest's ``training`` block: every completed episode's return + length (the full
    in-sample learning curve, in order), ``ep_rew_mean_first`` (mean of the first <= 100 episodes,
    the curve's start), ``final_ep_rew_mean`` (SB3's ``ep_rew_mean`` over the last <= 100 episodes,
    from its own ``ep_info_buffer``) and the ``Monitor`` info keywords averaged over that window."""
    if len(episode_returns) != len(episode_lengths):
        raise TrainError("episode_returns and episode_lengths must be the same length")
    returns = [_finite_or_none(float(r)) for r in episode_returns]
    window_returns = [float(entry["r"]) for entry in final_window]
    keyword_means = {
        f"mean_{key}": _finite_or_none(_mean_or_none([float(e[key]) for e in final_window]))
        for key in MONITOR_INFO_KEYWORDS
    }
    return {
        "ep_rew_window": EP_REW_WINDOW,
        "n_episodes": len(returns),
        "episode_returns": returns,
        "episode_lengths": [int(n) for n in episode_lengths],
        "ep_rew_mean_first": _finite_or_none(
            _mean_or_none([float(r) for r in episode_returns[:EP_REW_WINDOW]])
        ),
        "final_ep_rew_mean": _finite_or_none(_mean_or_none(window_returns)),
        "final_window": {"n_episodes": len(final_window), **keyword_means},
    }


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_manifest(
    *,
    bars: BarTable,
    env_config: EnvConfig,
    config: TrainConfig,
    model_file: str,
    model_sha256: str,
    policy_sha256: str,
    effective: Mapping[str, Any],
    num_timesteps: int,
    wall_seconds: float,
    created_at: datetime,
    training: Mapping[str, Any],
    versions: Mapping[str, str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The sidecar manifest (schema v1; exactly :data:`REQUIRED_MANIFEST_KEYS`). Pure: every value
    is passed in, so the schema is unit-tested without the ``[rl]`` extra. ``effective`` is
    :func:`effective_hyperparameters` of the trained model (read back, never assumed);
    ``model_sha256`` is a file-integrity hash that changes every run, ``policy_sha256`` the
    reproducible weights hash."""
    if not bars.bars:
        raise TrainError("cannot describe an empty bar table")
    if created_at.tzinfo is None:
        raise TrainError("created_at must be timezone-aware")
    first, last = bars.bars[0], bars.bars[-1]
    lo = MAX_LOOKBACK  # the env's first legal decision index (rl/env.py)
    hi = len(bars) - 1 - env_config.episode_length
    hyperparameters: dict[str, Any] = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "requested": {**asdict(config), "net_arch": list(config.net_arch)},
        "effective": dict(effective),
        "vec_env": "DummyVecEnv(n_envs=1)",
        "monitor_info_keywords": list(MONITOR_INFO_KEYWORDS),
        "vec_normalize": False,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "rl.train.ppo",
        "note": _IN_SAMPLE_NOTE,
        "symbol": bars.symbol,
        "seed": config.seed,
        "untrained": config.untrained,
        "model_file": model_file,
        "model_sha256": model_sha256,
        "policy_sha256": policy_sha256,
        "requested_timesteps": config.total_timesteps,
        "num_timesteps": num_timesteps,
        "passes_over_train_bars": passes_over_bars(num_timesteps, len(bars)),
        "wall_seconds": _finite_or_none(wall_seconds),
        "created_at": _iso(created_at),
        "hyperparameters": hyperparameters,
        "env": {
            "symbol": env_config.symbol,
            "episode_length": env_config.episode_length,
            "cost_one_way_bps": env_config.cost.one_way_bps,
            "reward": env_config.reward,
            "execution_lag": env_config.execution_lag,
            "invalid_action_penalty": env_config.invalid_action_penalty,
            "ruin_fraction": env_config.ruin_fraction,
            "random_start": env_config.random_start,
            "feature_window": env_config.spec.window,
            "feature_names": list(env_config.spec.names),
        },
        "train_bars": {
            "n_bars": len(bars),
            "first_session_ts": _iso(first.session_ts),
            "last_session_ts": _iso(last.session_ts),
            "first_known_at": _iso(first.known_at),
            "last_known_at": _iso(last.known_at),
            "episode_start_bounds": [lo, hi],
        },
        "seeding": {
            "seed": config.seed,
            "seeded": [
                "random",
                "numpy.random (global)",
                "torch.manual_seed",
                "action_space",
                "env first reset (np_random)",
            ],
            "torch_num_threads": TORCH_NUM_THREADS,
            "residual_nondeterminism": (
                "bit-identical only on a matching platform + library versions; float summation "
                "order in BLAS/oneDNN kernels may differ across CPUs/builds/torch versions"
            ),
            "hashes": (
                "policy_sha256 hashes the weights and reproduces across same-seed runs; "
                "model_sha256 is a file-integrity hash of the .zip and changes EVERY run (SB3 "
                "embeds start_time, Monitor elapsed times, pickled class blobs, zip timestamps)"
            ),
        },
        "library_versions": dict(versions),
        "python_version": platform.python_version(),
        "training": dict(training),
        "provenance": dict(provenance),
    }


def manifest_to_json(manifest: Mapping[str, Any]) -> bytes:
    """Deterministic, strict-JSON bytes (sorted keys; NaN/inf refused)."""
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    extra = set(manifest) - REQUIRED_MANIFEST_KEYS
    if missing or extra:
        raise TrainError(
            f"manifest keys mismatch: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    try:
        text = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)
    except ValueError as exc:
        raise TrainError(f"manifest is not strict JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


# --- the SB3 -> Policy bridge --------------------------------------------------------------------


def policy_factory(model: _Predictor) -> PolicyFactory:
    """Wrap a PPO model as an ``rl/baselines.py`` ``PolicyFactory`` for ``rl/scoring.py``.

    Each decision calls ``model.predict(obs, deterministic=True)`` on the env observation (the
    scoring loop passes it as Python floats widened from the env's float32, so the float32 cast
    here is exact) and returns the action as an int in ``Discrete(3)``. ``history`` is ignored —
    the agent's policy is a function of the observation alone — and so is the seed: the policy
    is deterministic (argmax of the logits, never a sample).
    """

    def factory(seed: int) -> Policy:
        del seed

        def act(observation: Sequence[float], history: Sequence[float]) -> int:
            del history
            obs = np.asarray(observation, dtype=np.float32)
            action, _state = model.predict(obs, deterministic=True)
            value = np.asarray(action)
            if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
                raise TrainError(f"policy returned a non-integer action {action!r}")
            result = int(value.reshape(-1)[0])
            if not 0 <= result < N_ACTIONS:
                raise TrainError(f"policy returned an action outside Discrete(3): {result}")
            return result

        return act

    return factory


def load_model(path: Path) -> "PPO":
    """Load a saved PPO ``.zip`` on the CPU (requires the ``[rl]`` extra)."""
    require_rl_extra()
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


# --- training ------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_ppo(
    bars: BarTable,
    env_config: EnvConfig,
    config: TrainConfig,
    *,
    out_dir: Path,
    tensorboard_log: Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    run_date: date | None = None,
    overwrite: bool = False,
) -> TrainResult:
    """Train (or, at ``total_timesteps=0``, initialize) one seeded PPO policy on ``bars`` and save
    ``<stem>.zip`` + ``<stem>.json`` under ``out_dir`` (see :func:`artifact_stem`).

    ``bars`` must already be the TRAIN split (``BarTable.split`` or an index slice); this function
    never splits. ``env_config.random_start`` must be True (random episode starts are mandatory in
    training, §4.7). ``provenance`` (dataset path / sha256 / split / git commit, supplied by the
    job) is embedded verbatim in the manifest. ``run_date`` (default: today, UTC) names the files.
    Raises :class:`TrainError` BEFORE training if ``<stem>.zip`` or ``<stem>.json`` already exists,
    unless ``overwrite=True``.
    """
    require_rl_extra()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    if not env_config.random_start:
        raise TrainError("training requires env_config.random_start=True (§4.7)")
    if bars.symbol != env_config.symbol:
        raise TrainError(
            f"bars are for {bars.symbol!r} but env_config is for {env_config.symbol!r}"
        )
    when = run_date if run_date is not None else datetime.now(UTC).date()
    stem = artifact_stem(bars.symbol, when, config.seed, untrained=config.untrained)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{stem}.zip"
    manifest_path = out_dir / f"{stem}.json"
    existing = [path.name for path in (model_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise TrainError(
            f"refusing to overwrite existing artifact(s) {existing} in {out_dir} "
            "(pass overwrite=True / --overwrite)"
        )

    # Monitor FIRST (SB3 docs), so its info keywords see the raw env's info dicts.
    monitor = Monitor(SingleTickerTradingEnv(bars, env_config), info_keywords=MONITOR_INFO_KEYWORDS)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        venv = DummyVecEnv([lambda: monitor])
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=config.ent_coef,
            policy_kwargs={"net_arch": list(config.net_arch)},
            tensorboard_log=None if tensorboard_log is None else str(tensorboard_log),
            seed=config.seed,
            device="cpu",
            verbose=1,
        )
        effective = effective_hyperparameters(model)
        _check_effective(config, effective)
        started = time.perf_counter()
        model.learn(total_timesteps=config.total_timesteps)
        wall_seconds = time.perf_counter() - started
        model.save(str(model_path))
    finally:
        torch.set_num_threads(previous_threads)

    weights_sha256 = policy_state_sha256(model.policy.state_dict())
    final_window = list(model.ep_info_buffer or [])
    training = training_summary(
        monitor.get_episode_rewards(), monitor.get_episode_lengths(), final_window
    )
    versions = library_versions()
    manifest = build_manifest(
        bars=bars,
        env_config=env_config,
        config=config,
        model_file=model_path.name,
        model_sha256=_sha256(model_path),
        policy_sha256=weights_sha256,
        effective=effective,
        num_timesteps=int(model.num_timesteps),
        wall_seconds=wall_seconds,
        created_at=datetime.now(UTC),
        training=training,
        versions=versions,
        provenance=dict(provenance or {}),
    )
    data = manifest_to_json(manifest)
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(manifest_path)
    return TrainResult(
        seed=config.seed,
        model_path=model_path,
        final_ep_rew_mean=training["final_ep_rew_mean"],
        sb3_version=versions["sb3"],
        torch_version=versions["torch"],
        gymnasium_version=versions["gymnasium"],
        numpy_version=versions["numpy"],
        manifest_path=manifest_path,
        num_timesteps=int(model.num_timesteps),
        wall_seconds=wall_seconds,
        untrained=config.untrained,
        policy_sha256=weights_sha256,
    )
