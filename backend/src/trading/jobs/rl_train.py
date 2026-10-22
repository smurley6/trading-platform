"""Offline PPO training over the frozen RL dataset's TRAIN split (task-rl-08).

Thin driver around ``trading.rl.train.train_ppo`` (docs/RL-SLEEVE-PROGRAM.md §4.9): reads the frozen
dataset (``jobs/rl_dataset.py``'s file, through ``rl_dataset_from_json`` -> ``BarTable``), cuts the
train split by knowledge time, and trains one seeded PPO policy per seed. Imports
``trading.rl.train`` only INSIDE :func:`run_training`, after the guards, so this module loads no ML
at import time and a missing ``[rl]`` extra is reported with a clear message.

USAGE::

    pip install -e ".[dev,rl]"          # the offline training extra (never installed on a server)
    # Linux: install the CPU-only torch wheel FIRST (the default Linux wheel is the CUDA build):
    #   pip install --index-url https://download.pytorch.org/whl/cpu torch
    python -m trading.jobs.rl_train --dataset data/rl/spy_daily.json --seeds 0,1,2,3,4 \\
        --train-end 2020-10-22 --timesteps 200000 --episode-length 252 --cost-bps 1.0 \\
        --out artifacts/rl/ [--confirm] [--overwrite]

  * DRY-RUN (default): a ``min(--timesteps, 5000)``-step smoke per seed into a throwaway temporary
    directory — nothing is written under ``--out`` and no event is appended.
  * ``--confirm``: the full run; writes ``{symbol}_ppo_{YYYYMMDD}_seed{n}.zip`` + its ``.json``
    manifest per seed under ``--out`` and appends ONE ``user.action`` audit event
    (``action="rl.train"``: symbol, seeds, timesteps, dataset sha256, per-seed final ep_rew_mean,
    exact sb3/torch/gymnasium/numpy versions). Requires a dataset frozen from the REAL historical
    adapter (its header's ``source_adapter``). The event is durable only when DATABASE_URL is
    configured (otherwise the in-memory store discards it at exit; the manifests remain).
  * ``--overwrite``: without it, ``--confirm`` refuses (before training anything) when any seed's
    ``<stem>.zip`` / ``<stem>.json`` already exists under ``--out`` — a repeat run never silently
    clobbers a prior run's artifacts.
  * ``backend/artifacts/`` is git-ignored: a training run never dirties the tree. The deliberate
    exhibit artifacts are committed at task-rl-12 bundle time with an explicit ``git add -f``.
  * Each manifest's ``policy_sha256`` hashes the WEIGHTS and reproduces across same-seed runs;
    its ``model_sha256`` (the ``.zip`` file) changes every run (see ``trading.rl.train``).
  * ``--timesteps 0`` is the §5.2 UNTRAINED baseline: each seed's policy at initialization (the
    same initial weights its trained run starts from), saved as
    ``{symbol}_ppo_untrained_{YYYYMMDD}_seed{n}.zip`` through the identical path.
  * ``--seeds`` needs at least 5 distinct seeds: a single-seed number is noise (program doc §0).
  * ``--train-end YYYY-MM-DD`` (required — there is no default that could train on the holdout):
    train = the bars whose close was public strictly before 00:00 America/New_York on that date
    (``BarTable.split(at, embargo=0)``; split by ``known_at``, never by ``session_ts``). On the
    frozen 2018-10-22.. SPY file, ``2020-10-22`` selects bars ``[0, 504)`` — the first walk-forward
    fold's train window (§5.3).

The printed ``ep_rew_mean`` numbers are IN-SAMPLE training diagnostics of the stochastic training
policy on the bars it trained on — never a result. The job prints the pass count (``timesteps /
train bars``): at 200k timesteps over ~500 bars it is ~400 passes, so overfitting is guaranteed.
Evaluation is task-rl-10's walk-forward job. Exit 0 clean / 1 on failure.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from trading.application.rl_dataset import MARKET_TZ, RlDatasetError, rl_dataset_from_json
from trading.config import Settings
from trading.domain.events import DomainEvent, EventType
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging
from trading.rl.bars import BarTable, BarTableError
from trading.rl.costs import CostModel

logger = logging.getLogger(__name__)

MIN_SEEDS: Final = 5
SMOKE_TIMESTEPS: Final = 5_000
SEEDS_ERROR: Final = "error: at least 5 training seeds are required; single-seed results are noise"
# Datasets frozen through the REAL historical-bars adapter (header `source_adapter`). An allowlist,
# so an unknown or synthetic provider fails closed under --confirm.
REAL_SOURCE_ADAPTERS: Final = frozenset({"AlpacaHistoricalData"})
_USER_ACTION_STREAM: Final = "user-actions"


class _UsageError(ValueError):
    """A CLI argument is invalid; the message is printed verbatim and the job exits 1."""


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    sys.stdout.write(text + "\n")


@dataclass(frozen=True, slots=True)
class TrainArgs:
    dataset: Path
    seeds: tuple[int, ...]
    timesteps: int
    episode_length: int
    cost_bps: float
    train_end: date
    out: Path
    confirm: bool
    overwrite: bool = False


def parse_seeds(raw: str) -> tuple[int, ...]:
    """``"0,1,2,3,4"`` -> ``(0, 1, 2, 3, 4)``: distinct non-negative ints, at least
    :data:`MIN_SEEDS` of them (else :data:`SEEDS_ERROR`)."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise _UsageError(f"error: --seeds must be comma-separated integers, got {raw!r}") from exc
    if any(seed < 0 for seed in seeds):
        raise _UsageError(f"error: --seeds must be non-negative, got {raw!r}")
    if len(set(seeds)) != len(seeds):
        raise _UsageError(f"error: --seeds must be distinct, got {raw!r}")
    if len(seeds) < MIN_SEEDS:
        raise _UsageError(SEEDS_ERROR)
    return seeds


def parse_args(argv: Sequence[str]) -> TrainArgs:
    """Parse + validate the CLI. ``SystemExit`` on an argparse usage error, :class:`_UsageError`
    (message printed verbatim) on a semantic one."""
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.rl_train",
        description="Train seeded PPO policies on the frozen RL dataset's train split (offline).",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="frozen RL dataset JSON")
    parser.add_argument("--seeds", required=True, help="comma-separated training seeds (>= 5)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="PPO timesteps per seed")
    parser.add_argument("--episode-length", type=int, default=252, help="decisions per episode")
    parser.add_argument("--cost-bps", type=float, default=1.0, help="one-way cost in bps")
    parser.add_argument(
        "--train-end",
        required=True,
        help="YYYY-MM-DD: train = bars public strictly before 00:00 New York on this date",
    )
    parser.add_argument("--out", type=Path, required=True, help="artifact directory (--confirm)")
    parser.add_argument(
        "--confirm", action="store_true", help="full run: write artifacts + one audit event"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="allow --confirm to replace existing artifacts"
    )
    args = parser.parse_args(list(argv))
    seeds = parse_seeds(str(args.seeds))
    if args.timesteps < 0:
        raise _UsageError("error: --timesteps must be >= 0 (0 = the untrained baseline)")
    if args.episode_length < 1:
        raise _UsageError("error: --episode-length must be >= 1")
    try:
        train_end = date.fromisoformat(str(args.train_end))
    except ValueError as exc:
        raise _UsageError(f"error: --train-end must be YYYY-MM-DD, got {args.train_end!r}") from exc
    return TrainArgs(
        dataset=Path(args.dataset),
        seeds=seeds,
        timesteps=int(args.timesteps),
        episode_length=int(args.episode_length),
        cost_bps=float(args.cost_bps),
        train_end=train_end,
        out=Path(args.out),
        confirm=bool(args.confirm),
        overwrite=bool(args.overwrite),
    )


def split_instant(train_end: date) -> datetime:
    """00:00 America/New_York on ``train_end``, in UTC — the ``BarTable.split`` boundary."""
    return datetime.combine(train_end, time(0), tzinfo=MARKET_TZ).astimezone(UTC)


def _git_state() -> dict[str, Any]:
    """The checkout's HEAD commit and whether the working tree differs from it — untracked files
    included, since a new, not-yet-committed module may be the very code that ran (None if no
    git). Conservative: any stray untracked file also marks the run dirty."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": head, "git_dirty": bool(status)}


def _fmt(value: float | None) -> str:
    return "n/a (no completed episode)" if value is None else f"{value:+.3f}"


async def run_training(args: TrainArgs) -> int:
    """Load + split the dataset, guard, train one policy per seed, and (``--confirm``) append the
    audit event. Teardown of any composition-root resources always runs."""
    settings = Settings()
    configure_logging(settings.log_level)

    try:
        raw = args.dataset.read_bytes()
    except OSError as exc:
        _emit(f"error: cannot read --dataset {args.dataset}: {exc}")
        return 1
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        dataset = rl_dataset_from_json(raw)
        table = BarTable.from_dataset(dataset)
    except (RlDatasetError, BarTableError) as exc:
        _emit(f"error: invalid dataset {args.dataset}: {exc}")
        return 1

    if args.confirm and dataset.source_adapter not in REAL_SOURCE_ADAPTERS:
        _emit(
            f"error: --confirm requires a dataset frozen from the REAL historical-bars adapter "
            f"({', '.join(sorted(REAL_SOURCE_ADAPTERS))}); this file's header records "
            f"source_adapter={dataset.source_adapter!r}. Refusing to train artifacts on synthetic "
            "data — a dry run is always allowed."
        )
        return 1

    at = split_instant(args.train_end)
    train, _ = table.split(at, embargo=timedelta(0))
    if not train.bars:
        _emit(f"error: no bar was public before {at.isoformat()} (--train-end {args.train_end})")
        return 1

    # Heavy imports only now, after every cheap guard.
    try:
        from trading.rl import train as rl_train
        from trading.rl.env import EnvConfig, SingleTickerTradingEnv

        rl_train.require_rl_extra()
    except (ImportError, RuntimeError) as exc:
        _emit(f"error: {exc}")
        return 1

    try:
        env_config = EnvConfig(
            symbol=table.symbol,
            episode_length=args.episode_length,
            cost=CostModel(one_way_bps=args.cost_bps),
            random_start=True,
        )
        SingleTickerTradingEnv(train, env_config)  # fail fast: the split must hold one episode
    except ValueError as exc:
        _emit(f"error: {exc}")
        return 1

    run_date = datetime.now(UTC).date()  # ONE date names every seed's artifacts
    collisions = [
        f"{stem}{suffix}"
        for seed in args.seeds
        for stem in [
            rl_train.artifact_stem(table.symbol, run_date, seed, untrained=args.timesteps == 0)
        ]
        for suffix in (".zip", ".json")
        if (args.out / f"{stem}{suffix}").exists()
    ]
    if collisions and args.confirm and not args.overwrite:
        _emit(
            f"error: refusing to overwrite existing artifact(s) under {args.out}: "
            f"{', '.join(collisions)}. Pass --overwrite to replace them."
        )
        return 1

    steps = args.timesteps if args.confirm else min(args.timesteps, SMOKE_TIMESTEPS)
    passes = rl_train.passes_over_bars(args.timesteps, len(train))
    first, last = train.bars[0], train.bars[-1]
    _emit("=== RL train (PPO) ===")
    _emit(f"mode:      {'CONFIRM (writes artifacts)' if args.confirm else 'DRY-RUN (no changes)'}")
    _emit(f"dataset:   {args.dataset}  (sha256 {dataset_sha256})")
    _emit(f"source:    {dataset.source_adapter}, adjustment={dataset.adjustment}")
    _emit(
        f"train:     {len(train)} bars, sessions {first.session_ts.date()} .. "
        f"{last.session_ts.date()} (known_at < {at.isoformat()})"
    )
    _emit(f"seeds:     {', '.join(str(s) for s in args.seeds)}")
    _emit(
        f"timesteps: {args.timesteps} per seed"
        + ("" if args.confirm else f" (dry-run smoke: {steps})")
        + (" — UNTRAINED baseline (policy at initialization)" if args.timesteps == 0 else "")
    )
    _emit(
        f"passes over the train bars: {args.timesteps} / {len(train)} = {passes:.1f} — "
        "overfitting is structurally guaranteed"
    )
    _emit(
        "NOTE: ep_rew_mean below is an IN-SAMPLE training diagnostic (stochastic policy, the bars "
        "it trained on). It is NOT a result."
    )
    if collisions:
        verdict = "will REPLACE them" if args.overwrite else "needs --overwrite"
        _emit(
            f"WARNING: {len(collisions)} artifact file(s) for this date already exist under "
            f"{args.out}; --confirm {verdict}."
        )
    _emit("")

    provenance: dict[str, Any] = {
        "job": "trading.jobs.rl_train",
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "dataset_adjustment": dataset.adjustment,
        "dataset_source_adapter": dataset.source_adapter,
        "dataset_n_bars": len(table),
        "train_end": args.train_end.isoformat(),
        "split": f"BarTable.split(at={at.isoformat()}, embargo=0): bars with known_at < at",
        **_git_state(),
    }

    deps: AppDeps | None = None
    try:
        if args.confirm:
            deps = build_dependencies(settings)
        results: list[Any] = []
        for seed in args.seeds:
            config = rl_train.TrainConfig(total_timesteps=steps, seed=seed)
            if args.confirm:
                result = rl_train.train_ppo(
                    train,
                    env_config,
                    config,
                    out_dir=args.out,
                    provenance=provenance,
                    run_date=run_date,
                    overwrite=args.overwrite,
                )
                first_mean = _manifest_first_mean(result.manifest_path)
            else:
                with tempfile.TemporaryDirectory(prefix="rl_train_dry_run_") as tmp:
                    result = rl_train.train_ppo(
                        train,
                        env_config,
                        config,
                        out_dir=Path(tmp),
                        provenance=provenance,
                        run_date=run_date,
                    )
                    first_mean = _manifest_first_mean(result.manifest_path)
            results.append(result)
            _emit(
                f"seed {seed}: {result.num_timesteps} steps in {result.wall_seconds:.1f}s; "
                f"in-sample ep_rew_mean first {_fmt(first_mean)} -> final "
                f"{_fmt(result.final_ep_rew_mean)}; policy_sha256 {result.policy_sha256[:16]}"
                + (f"; wrote {result.model_path}" if args.confirm else "")
            )

        if not args.confirm or deps is None:
            _emit("")
            _emit("DRY-RUN: nothing was written. Re-run with --confirm for the full run.")
            return 0

        versions = rl_train.library_versions()
        await deps.event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": "rl.train",
                    "symbol": table.symbol,
                    "seeds": list(args.seeds),
                    "timesteps": args.timesteps,
                    "untrained": args.timesteps == 0,
                    "dataset_sha256": dataset_sha256,
                    "train_end": args.train_end.isoformat(),
                    "n_train_bars": len(train),
                    "final_ep_rew_mean": {str(r.seed): r.final_ep_rew_mean for r in results},
                    "policy_sha256": {str(r.seed): r.policy_sha256 for r in results},
                    "versions": versions,
                    "models": [r.model_path.name for r in results],
                },
                occurred_at=deps.clock.now(),
                correlation_id=str(uuid4()),
            )
        )
        _emit("")
        _emit(f"wrote {len(results)} model(s) + manifest(s) under {args.out}.")
        return 0
    finally:
        if deps is not None:
            await _teardown(deps)


def _manifest_first_mean(path: Path) -> float | None:
    training = json.loads(path.read_text(encoding="utf-8"))["training"]
    value = training["ep_rew_mean_first"]
    return None if value is None else float(value)


async def _teardown(deps: AppDeps) -> None:
    """Release every async resource the composition root registered (mirrors app shutdown)."""
    for aclose in deps.aclosers:
        try:
            await aclose()
        except Exception:
            logger.exception("error closing a resource during job teardown")


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entrypoint: parse args, train, return a process exit code (0 ok / 1 failure)."""
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    except _UsageError as exc:
        _emit(str(exc))
        return 1
    try:
        return asyncio.run(run_training(args))
    except Exception:
        logger.exception("rl-train job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
