"""Export a trained PPO ``.zip`` to the torch-free policy JSON, behind the parity gate (task-rl-09).

Thin driver around ``trading.rl.export`` (docs/RL-SLEEVE-PROGRAM.md §3.4): loads the model and its
training manifest (``<model>.json``, written next to the ``.zip`` by ``trading.jobs.rl_train``),
copies the actor into an :class:`~trading.domain.rl_policy.MlpPolicy`, proves parity against
``PPO.predict(obs, deterministic=True)``, and writes ONE JSON artifact through
``trading.infrastructure.rl.policy_store``. Imports ``trading.rl.export`` only INSIDE
:func:`run_export`, after the cheap guards, so this module loads no ML at import time.

USAGE::

    pip install -e ".[dev,rl]"          # the offline extra (export reads the SB3 .zip)
    python -m trading.jobs.rl_export --model artifacts/rl/spy_ppo_20261022_seed0.zip \\
        --dataset data/rl/spy_daily.json --out artifacts/rl/spy_ppo_v1.json \\
        [--confirm] [--overwrite]

  * DRY-RUN (default): export + parity in memory; prints the parity report and the sha256 / size
    the artifact WOULD have. Nothing is written and no event is appended.
  * ``--confirm``: writes the artifact (atomically) and appends ONE ``user.action`` audit event
    (``action="rl.export"``: policy_id, artifact sha256, model + dataset hashes, parity counts).
    Requires a dataset frozen from the REAL historical adapter (as ``rl_train --confirm`` does).
  * ``--overwrite``: without it, ``--confirm`` refuses when ``--out`` already exists. ``--out`` may
    never be the model's own ``.zip`` or training manifest.
  * **The parity gate.** ``1000`` observations sampled from the observation space (seed ``0``)
    plus every decision of the validation slice (flat AND held) go through both SB3 and the stdlib
    policy. Any disagreement that is not a float32 tie (competing logits within ``1e-4``) fails the
    job with exit 1 and NOTHING is written, with or without ``--confirm``.
  * **Fail-closed provenance checks.** The manifest must belong to this ``.zip`` (its
    ``policy_sha256`` must equal the loaded weights' hash) and ``--dataset`` must be the file the
    model trained on (its sha256 must equal the manifest's ``provenance.dataset_sha256``).

``policy_id = f"{symbol}-ppo-{YYYYMMDD}-{sha8}"`` (``{symbol}-ppo-untrained-{YYYYMMDD}-{sha8}`` for
the §5.2 untrained baseline) with the symbol lower-cased (as in the artifact
file names), ``YYYYMMDD`` the training run's date (from its artifact stem) and ``sha8`` the first 8
hex digits of the weights hash ``policy_sha256`` — so the id is content-addressed.

The feature spec (window, names AND scales) is read from the manifest's ``env`` block, never
assumed — parity cannot detect a wrong spec, since both sides would see the same observations. A
manifest without ``env.feature_scales`` (pre-task-rl-09) is accepted only from
``trading.jobs.rl_train``, which always trained on the default spec, and is stamped
``feature_scales_source="default_assumed"``; any other scale-less manifest is refused.

The artifact's ``meta`` records the feature spec's provenance (the policy body itself carries the
feature names, window, scales and the index -> action mapping), the training run (``run_id`` = the
artifact stem, seed, hashes, timesteps), the dataset (sha256, adjustment, source, date range), the
train / val / purge / test bounds, the exact sb3 / torch / gymnasium / numpy versions (training
AND export), and the parity report. The train window is the manifest's; val / purge / test are the
program doc §5.3 fold geometry immediately after it (126 / 50 / 63 bars; listed in
``meta.splits.derived``) — task-rl-10's
``make_folds`` is authoritative for evaluation. No price LEVEL is recorded: the closes are
``adjustment=all`` (dividend back-adjusted; §4.2 returns-only rule). Exit 0 clean / 1 on failure.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from trading.application.rl_dataset import RlDataset, RlDatasetError, rl_dataset_from_json
from trading.config import Settings
from trading.domain.events import DomainEvent, EventType
from trading.domain.rl_features import WARMUP_BARS, FeatureScales, FeatureSpec, build_observation
from trading.infrastructure.rl.policy_store import (
    PolicyArtifactError,
    load_policy,
    policy_to_json,
    write_policy,
)
from trading.interface.app import AppDeps
from trading.jobs.rl_train import REAL_SOURCE_ADAPTERS, _git_state
from trading.main import build_dependencies
from trading.observability import configure_logging
from trading.rl.bars import BarTable, BarTableError

logger = logging.getLogger(__name__)

PARITY_SAMPLES: Final = 1_000
PARITY_SEED: Final = 0
# Program doc §5.3 fold geometry following a train window (rl-10 make_folds is authoritative).
VAL_BARS: Final = 126
PURGE_BARS: Final = WARMUP_BARS
TEST_BARS: Final = 63
_MANIFEST_KIND: Final = "rl.train.ppo"
# The one job whose manifests may omit env.feature_scales (it always trained on the default spec).
_DEFAULT_SPEC_JOB: Final = "trading.jobs.rl_train"
_SCALES_RECORDED: Final = "manifest"
_SCALES_ASSUMED: Final = "default_assumed"
_MANIFEST_SCHEMA_VERSION: Final = 1
_STEM_RE: Final = re.compile(r"^(?P<symbol>[a-z0-9]+)_ppo(?:_untrained)?_(?P<date>\d{8})_seed\d+$")
_USER_ACTION_STREAM: Final = "user-actions"


class _UsageError(ValueError):
    """A CLI argument or input file is invalid; the message is printed verbatim (exit 1)."""


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    sys.stdout.write(text + "\n")


@dataclass(frozen=True, slots=True)
class ExportArgs:
    model: Path
    dataset: Path
    out: Path
    confirm: bool
    overwrite: bool = False


def parse_args(argv: Sequence[str]) -> ExportArgs:
    """Parse the CLI (``SystemExit`` on an argparse usage error)."""
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.rl_export",
        description="Export a trained PPO .zip to the torch-free policy JSON (parity-gated).",
    )
    parser.add_argument("--model", type=Path, required=True, help="trained PPO .zip")
    parser.add_argument("--dataset", type=Path, required=True, help="the frozen RL dataset JSON")
    parser.add_argument("--out", type=Path, required=True, help="policy JSON path (--confirm)")
    parser.add_argument("--confirm", action="store_true", help="write the artifact + audit event")
    parser.add_argument(
        "--overwrite", action="store_true", help="allow --confirm to replace an existing --out"
    )
    args = parser.parse_args(list(argv))
    return ExportArgs(
        model=Path(args.model),
        dataset=Path(args.dataset),
        out=Path(args.out),
        confirm=bool(args.confirm),
        overwrite=bool(args.overwrite),
    )


# --- pure helpers (covered without the [rl] extra) ---------------------------------------------


def _require(mapping: Mapping[str, Any], key: str, kind: type[Any]) -> Any:
    value = mapping.get(key)
    if isinstance(value, bool) and kind is not bool:
        raise _UsageError(f"error: training manifest field {key!r} has the wrong type")
    if not isinstance(value, kind):
        raise _UsageError(f"error: training manifest field {key!r} is missing or mistyped")
    return value


def read_manifest(path: Path, *, model: Path) -> dict[str, Any]:
    """The training manifest next to ``model``, checked to be an ``rl.train.ppo`` v1 manifest
    naming this ``.zip``."""
    try:
        manifest: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _UsageError(f"error: cannot read the training manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _UsageError(f"error: training manifest {path} is not a JSON object")
    version = manifest.get("schema_version")
    if manifest.get("kind") != _MANIFEST_KIND or (
        type(version) is not int or version != _MANIFEST_SCHEMA_VERSION
    ):
        raise _UsageError(
            f"error: {path} is not a {_MANIFEST_KIND} v{_MANIFEST_SCHEMA_VERSION} training manifest"
        )
    if manifest.get("model_file") != model.name:
        raise _UsageError(
            f"error: manifest {path} describes {manifest.get('model_file')!r}, not {model.name!r}"
        )
    return manifest


def run_identity(manifest: Mapping[str, Any]) -> tuple[str, str]:
    """``(run_id, YYYYMMDD)``: the training artifact stem
    (``{symbol}_ppo[_untrained]_{date}_seed{n}``) and its run date."""
    model_file = str(_require(manifest, "model_file", str))
    run_id = model_file.removesuffix(".zip")
    match = _STEM_RE.match(run_id)
    if match is None:
        raise _UsageError(f"error: model_file {model_file!r} is not a training artifact name")
    return run_id, match["date"]


def make_policy_id(symbol: str, run_date: str, policy_sha256: str, *, untrained: bool) -> str:
    """``f"{symbol}-ppo-{YYYYMMDD}-{sha8}"`` (symbol lower-cased; ``sha8`` = weights hash), with an
    ``-untrained`` infix (``{symbol}-ppo-untrained-...``) for the §5.2 untrained baseline so the id
    alone tells the two apart."""
    if not re.fullmatch(r"[0-9a-f]{64}", policy_sha256):
        raise _UsageError(f"error: policy_sha256 {policy_sha256!r} is not a sha256 hex digest")
    kind = "ppo-untrained" if untrained else "ppo"
    return f"{symbol.lower()}-{kind}-{run_date}-{policy_sha256[:8]}"


def feature_spec_for(manifest: Mapping[str, Any]) -> tuple[FeatureSpec, str]:
    """``(spec, scales_source)``: the observation spec the model trained with, as RECORDED.

    Parity cannot catch a wrong spec (both sides would see the same, wrong observations), so the
    scales are never assumed when the manifest can say:

    - ``env.feature_scales`` present (every manifest since task-rl-09): window, names and scales
      are read from the manifest and validated (``FeatureSpec.from_dict``) -> ``"manifest"``.
    - absent (older manifests): accepted ONLY when ``provenance.job`` is
      ``trading.jobs.rl_train``, whose code always built ``EnvConfig()`` with the default spec ->
      ``"default_assumed"``. A scale-less manifest from anywhere else (e.g. a direct ``train_ppo``
      call, which accepts any ``EnvConfig.spec``) is REFUSED.
    """
    env = _require(manifest, "env", dict)
    window = _require(env, "feature_window", int)
    names = _require(env, "feature_names", list)
    scales = env.get("feature_scales")
    if scales is None:
        provenance = manifest.get("provenance")
        job = provenance.get("job") if isinstance(provenance, dict) else None
        if job != _DEFAULT_SPEC_JOB:
            raise _UsageError(
                "error: the training manifest does not record env.feature_scales and was not "
                f"written by {_DEFAULT_SPEC_JOB} (provenance.job={job!r}); refusing to assume "
                "the default scales"
            )
        scales, source = asdict(FeatureScales()), _SCALES_ASSUMED
    else:
        source = _SCALES_RECORDED
    try:
        spec = FeatureSpec.from_dict({"window": window, "names": names, "scales": scales})
    except ValueError as exc:
        raise _UsageError(f"error: manifest feature spec is not valid: {exc}") from exc
    return spec, source


def _session(table: BarTable, index: int) -> str:
    return table.bars[index].session_ts.date().isoformat()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def split_bounds(table: BarTable, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Half-open bar-index windows ``train`` / ``val`` / ``purge`` / ``test`` with session dates.
    ``train`` is located in ``table`` from the manifest (first / last session must match exactly);
    the rest is the §5.3 geometry right after it. Fails closed if it does not fit the dataset."""
    train_bars = _require(manifest, "train_bars", dict)
    n_train = _require(train_bars, "n_bars", int)
    first_ts = _require(train_bars, "first_session_ts", str)
    last_ts = _require(train_bars, "last_session_ts", str)
    stamps = [_iso(bar.session_ts) for bar in table.bars]
    try:
        start = stamps.index(first_ts)
    except ValueError as exc:
        raise _UsageError(
            f"error: the train window's first session {first_ts} is not in the dataset"
        ) from exc
    stop = start + n_train
    if n_train < 1 or stop > len(table) or stamps[stop - 1] != last_ts:
        raise _UsageError(
            f"error: the dataset does not hold the manifest's {n_train}-bar train window "
            f"{first_ts} .. {last_ts}"
        )
    windows = {
        "train": (start, stop),
        "val": (stop, stop + VAL_BARS),
        "purge": (stop + VAL_BARS, stop + VAL_BARS + PURGE_BARS),
        "test": (stop + VAL_BARS + PURGE_BARS, stop + VAL_BARS + PURGE_BARS + TEST_BARS),
    }
    if windows["test"][1] > len(table):
        raise _UsageError(
            f"error: the dataset ({len(table)} bars) is too short for the §5.3 val/purge/test "
            f"windows after the train window (need {windows['test'][1]})"
        )
    return {
        name: {
            "bars": [lo, hi],
            "first_session": _session(table, lo),
            "last_session": _session(table, hi - 1),
        }
        for name, (lo, hi) in windows.items()
    }


def validation_observations(
    table: BarTable, bounds: Mapping[str, Mapping[str, Any]], spec: FeatureSpec
) -> list[tuple[float, ...]]:
    """Every decision of the validation slice, twice: flat, and held since the slice's first close
    (an ``entry_close`` already public at each decision — same series, so level-free)."""
    lo, hi = bounds["val"]["bars"]
    closes = table.closes()
    entry = closes[lo]
    observations: list[tuple[float, ...]] = []
    for t in range(lo, hi):
        window = closes[: t + 1]
        observations.append(build_observation(window, held=False, entry_close=None, spec=spec))
        observations.append(build_observation(window, held=True, entry_close=entry, spec=spec))
    return observations


def build_meta(
    *,
    manifest: Mapping[str, Any],
    run_id: str,
    dataset: RlDataset,
    table: BarTable,
    dataset_path: Path,
    dataset_sha256: str,
    bounds: Mapping[str, Any],
    parity: Mapping[str, Any],
    export_versions: Mapping[str, str],
    scales_source: str,
) -> dict[str, object]:
    """The artifact's provenance block (JSON types only; no price level anywhere)."""
    hyper = _require(manifest, "hyperparameters", dict)
    effective = _require(hyper, "effective", dict)
    provenance = _require(manifest, "provenance", dict)
    return {
        "symbol": table.symbol,
        "inference": (
            "deterministic: action = actions[argmax(logits)], exactly SB3 "
            "predict(obs, deterministic=True); never sampled"
        ),
        "training": {
            "run_id": run_id,
            "seed": _require(manifest, "seed", int),
            "untrained": _require(manifest, "untrained", bool),
            "model_file": _require(manifest, "model_file", str),
            "model_sha256": _require(manifest, "model_sha256", str),
            "policy_sha256": _require(manifest, "policy_sha256", str),
            "num_timesteps": _require(manifest, "num_timesteps", int),
            "created_at": _require(manifest, "created_at", str),
            "net_arch": effective.get("net_arch"),
            "activation_fn": effective.get("activation_fn"),
            "features_extractor": effective.get("features_extractor"),
            "feature_scales_source": scales_source,
            "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"),
            "library_versions": dict(_require(manifest, "library_versions", dict)),
            "python_version": manifest.get("python_version"),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "adjustment": dataset.adjustment,
            "source_adapter": dataset.source_adapter,
            "generated_at": _iso(dataset.generated_at),
            "n_bars": len(table),
            "first_session": _session(table, 0),
            "last_session": _session(table, len(table) - 1),
            "price_basis": (
                f"adjustment={dataset.adjustment} closes: consumed only as ratios (program doc "
                "§4.2 returns-only rule); no price level is recorded in this artifact"
            ),
        },
        "splits": {
            "source": (
                "train = the training manifest's train window; val / purge / test = program doc "
                f"§5.3 fold geometry immediately after it ({VAL_BARS} / {PURGE_BARS} / "
                f"{TEST_BARS} bars); task-rl-10 make_folds is authoritative for evaluation"
            ),
            "half_open": True,
            "derived": ["val", "purge", "test"],
            **bounds,
        },
        "parity": dict(parity),
        "export": {
            "job": "trading.jobs.rl_export",
            "library_versions": dict(export_versions),
            "python_version": platform.python_version(),
            **_git_state(),
        },
    }


# --- the job -------------------------------------------------------------------------------------


async def run_export(args: ExportArgs) -> int:
    """Guard, export, prove parity, and (``--confirm``) write + audit. Teardown always runs."""
    settings = Settings()
    configure_logging(settings.log_level)

    manifest_path = args.model.with_suffix(".json")
    if args.out.resolve() in {args.model.resolve(), manifest_path.resolve()}:
        _emit("error: --out must not be the model .zip or its training manifest")
        return 1
    try:
        raw = args.dataset.read_bytes()
    except OSError as exc:
        _emit(f"error: cannot read --dataset {args.dataset}: {exc}")
        return 1
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        dataset = rl_dataset_from_json(raw)
        table = BarTable.from_dataset(dataset)
        manifest = read_manifest(manifest_path, model=args.model)
        provenance = _require(manifest, "provenance", dict)
        if provenance.get("dataset_sha256") != dataset_sha256:
            raise _UsageError(
                f"error: --dataset sha256 {dataset_sha256} is not the dataset this model trained "
                f"on ({provenance.get('dataset_sha256')!r}, from {manifest_path.name})"
            )
        if manifest.get("symbol") != table.symbol:
            raise _UsageError(
                f"error: the model is for {manifest.get('symbol')!r}, the dataset for "
                f"{table.symbol!r}"
            )
        run_id, run_date = run_identity(manifest)
        policy_sha256 = str(_require(manifest, "policy_sha256", str))
        untrained = bool(_require(manifest, "untrained", bool))
        policy_id = make_policy_id(table.symbol, run_date, policy_sha256, untrained=untrained)
        spec, scales_source = feature_spec_for(manifest)
        bounds = split_bounds(table, manifest)
    except (RlDatasetError, BarTableError) as exc:
        _emit(f"error: invalid dataset {args.dataset}: {exc}")
        return 1
    except _UsageError as exc:
        _emit(str(exc))
        return 1

    if args.confirm and dataset.source_adapter not in REAL_SOURCE_ADAPTERS:
        _emit(
            "error: --confirm requires a dataset frozen from the REAL historical-bars adapter "
            f"({', '.join(sorted(REAL_SOURCE_ADAPTERS))}); this file's header records "
            f"source_adapter={dataset.source_adapter!r}. A dry run is always allowed."
        )
        return 1
    if args.confirm and args.out.exists() and not args.overwrite:
        _emit(f"error: refusing to overwrite {args.out}. Pass --overwrite to replace it.")
        return 1

    # Heavy imports only now, after every cheap guard.
    try:
        from trading.rl import export as rl_export
        from trading.rl import train as rl_train

        rl_train.require_rl_extra()
    except (ImportError, RuntimeError) as exc:
        _emit(f"error: {exc}")
        return 1

    model = rl_train.load_model(args.model)
    loaded_sha256 = rl_train.policy_state_sha256(model.policy.state_dict())
    if loaded_sha256 != policy_sha256:
        _emit(
            f"error: {args.model.name}'s weights hash {loaded_sha256} does not match its "
            f"manifest's policy_sha256 {policy_sha256}; refusing to export"
        )
        return 1
    try:
        policy = rl_export.sb3_to_mlp_policy(model, policy_id=policy_id, spec=spec)
    except rl_export.ExportError as exc:
        _emit(f"error: cannot export {args.model.name}: {exc}")
        return 1

    sampled = rl_export.sample_observations(
        model.observation_space, PARITY_SAMPLES, seed=PARITY_SEED
    )
    val_obs = validation_observations(table, bounds, spec)
    _emit("=== RL export (PPO -> torch-free MLP) ===")
    _emit(f"mode:      {'CONFIRM (writes artifact)' if args.confirm else 'DRY-RUN (no changes)'}")
    _emit(f"model:     {args.model}  (run {run_id}, policy_sha256 {policy_sha256})")
    _emit(f"dataset:   {args.dataset}  (sha256 {dataset_sha256}, adjustment={dataset.adjustment})")
    _emit(f"policy_id: {policy_id}")
    _emit(
        "layers:    "
        + " -> ".join(
            [str(policy.obs_dim)]
            + [f"{layer.out_features}({layer.activation})" for layer in policy.hidden]
            + [f"{policy.action_head.out_features}(logits)"]
        )
    )
    try:
        report = rl_export.assert_parity(model, policy, [*sampled, *val_obs])
    except rl_export.ParityError as exc:
        _emit(f"PARITY FAILED: {exc}")
        _emit("nothing was written.")
        return 1
    _emit(
        f"parity:    {report.n_agree}/{report.n_observations} agree, {report.n_ties} tie(s) "
        f"(< {report.tie_margin:g} on both sides), 0 disagreements; max |logit diff| "
        f"{report.max_abs_logit_diff:.3g} (limit {report.logit_tol:g})"
    )

    parity = {
        **report.to_dict(),
        "n_sampled": len(sampled),
        "sample_seed": PARITY_SEED,
        "n_validation": len(val_obs),
        "validation_states": "every val decision, flat and held",
    }
    meta = build_meta(
        manifest=manifest,
        run_id=run_id,
        dataset=dataset,
        table=table,
        dataset_path=args.dataset,
        dataset_sha256=dataset_sha256,
        bounds=bounds,
        parity=parity,
        export_versions=rl_train.library_versions(),
        scales_source=scales_source,
    )
    try:
        data = policy_to_json(policy, meta=meta)
    except PolicyArtifactError as exc:
        _emit(f"error: {exc}")
        return 1
    would_sha = hashlib.sha256(data).hexdigest()

    if not args.confirm:
        _emit(f"artifact:  {len(data)} bytes, sha256 {would_sha} (not written)")
        _emit("")
        _emit("DRY-RUN: nothing was written. Re-run with --confirm to write the artifact.")
        return 0

    deps: AppDeps | None = None
    try:
        deps = build_dependencies(settings)
        sha256 = write_policy(args.out, policy, meta=meta)
        try:
            reads_back = load_policy(args.out) == policy
        except PolicyArtifactError:
            reads_back = False
        if not reads_back:
            args.out.unlink(missing_ok=True)  # never leave an unverified artifact behind
            _emit(f"error: {args.out} did not read back as the exported policy; removed it")
            return 1
        size = args.out.stat().st_size
        await deps.event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": "rl.export",
                    "policy_id": policy_id,
                    "artifact": args.out.name,
                    "artifact_sha256": sha256,
                    "artifact_bytes": size,
                    "run_id": run_id,
                    "model_file": args.model.name,
                    "policy_sha256": policy_sha256,
                    "dataset_sha256": dataset_sha256,
                    "parity": parity,
                    "versions": rl_train.library_versions(),
                },
                occurred_at=deps.clock.now(),
                correlation_id=str(uuid4()),
            )
        )
        _emit(f"wrote {args.out}: {size} bytes, sha256 {sha256}")
        return 0
    finally:
        if deps is not None:
            await _teardown(deps)


async def _teardown(deps: AppDeps) -> None:
    """Release every async resource the composition root registered (mirrors app shutdown)."""
    for aclose in deps.aclosers:
        try:
            await aclose()
        except Exception:
            logger.exception("error closing a resource during job teardown")


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entrypoint: parse args, export, return a process exit code (0 ok / 1 failure)."""
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    try:
        return asyncio.run(run_export(args))
    except Exception:
        logger.exception("rl-export job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
