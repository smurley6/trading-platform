"""Build the static website demo bundle ``demo.json`` (task-rl-12; program doc retarget W0).

A pure READ/TRANSFORM of artifacts that already exist — the frozen dataset (task-rl-02), the
walk-forward per-fold policy exports (task-rl-10 ``--retrain``) and the rl-10 evaluation report.
No network, no training, no new evaluation: it appends NO event and never touches the look ledger
(``rl_eval_looks.jsonl``) — the eval table is the already-recorded look's numbers, transformed.

USAGE::

    python -m trading.jobs.rl_bundle --dataset data/rl/spy_daily.json \\
        --eval artifacts/rl/rl_eval.json --walk-forward artifacts/rl/walk_forward \\
        --out artifacts/rl/demo.json [--exhibit-seed 0] [--generated-at 2026-10-22T10:14:10Z]

``--walk-forward`` replaces the spec's single ``--policy`` (task-rl-10 review addendum, option
(a)): the exhibit is the EVALUATED agent — ONE seed's per-fold policies stitched across their own
test segments — never a single network trained on one window.

**Fail-closed provenance checks (nothing is written unless all pass).**

- The report must be about THIS dataset (``dataset_sha256``, symbol, adjustment) and its folds must
  be exactly ``make_folds`` of this dataset at the report's geometry, each passing
  ``assert_no_leakage``.
- Every exhibit policy goes through ``jobs/rl_eval.load_fold_policy`` — the SAME dataset-sha256 +
  train-window (bar index AND session date) + ``meta.walk_forward`` identity checks the evaluation
  applied — plus the env feature-spec check, and its file's sha256 must equal the hash the report
  recorded for that path (the bytes replayed here are the bytes evaluated).
- Every env replay (``trading.rl.scoring.rollout``: EVERY agent seed — the exhibit's curve and the
  5-seed envelope — buy-and-hold, always-flat, SMA, each uniform-random seed and each
  TURNOVER-MATCHED random seed (``matched_trade_probability`` over the fold's agent episodes,
  exactly rl-10's ``random_matched[ppo]``), at every cost) must reproduce the report's row for that
  (policy, fold, seed, cost) — ``final_equity`` and ``sharpe`` within 1e-9 relative. Every agent
  seed's policy passes the same identity + evaluated-sha checks; only the exhibit seed's ship.
- Budget priority: the matched band and the envelope always ship; the uniform band is dropped
  first (``curves.dropped`` says so) if the bundle would exceed the size budget.
- The stdlib replay (``rl.bundle.replay_fold``: what the page runs) must request the env's exact
  action and hold the env's exact position at every decision, and its equity accounting
  (``rl.bundle.equity_path``) must reproduce the env's equity at every cost (1e-12 relative).
- The assembled bytes must contain test-segment curve data only, no key-like string / account id /
  absolute local path, and fit the size budget (raw <= 2.5 MB, gzip <= 600 KB).

``meta.generated_at`` is DETERMINISTIC so that equal inputs give byte-identical bundles: it is the
evaluation report's ``as_of`` (when the bundled numbers were produced) unless ``--generated-at``
overrides it; ``meta.generated_at_source`` says which. Exit 0 clean / 1 on failure.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from trading.application.rl_dataset import RlDatasetError, rl_dataset_from_json
from trading.application.rl_evaluation import (
    LeakageError,
    RlEvaluationError,
    WalkForwardFold,
    make_folds,
)
from trading.domain.rl_features import WARMUP_BARS
from trading.domain.rl_policy import ENV_ACTION_ORDER, MlpPolicy
from trading.jobs.rl_eval import (
    AGENT_ID,
    ALWAYS_FLAT_ID,
    BUY_AND_HOLD_ID,
    HOLDOUT_BARS,
    RANDOM_ID,
    SMA_ID,
    WALK_FORWARD_DIR,
    load_fold_policy,
)
from trading.jobs.rl_eval import matched_id as rl_eval_matched_id
from trading.jobs.rl_train import _git_state
from trading.rl import bundle as rl_bundle
from trading.rl.bars import BarTable, BarTableError

logger = logging.getLogger(__name__)

ROW_TOLERANCE: Final = 1e-9
EQUITY_TOLERANCE: Final = 1e-12
_GENERATED_FROM_REPORT: Final = "the evaluation report's as_of (deterministic default)"
_GENERATED_FROM_FLAG: Final = "--generated-at (operator-supplied)"


class _UsageError(ValueError):
    """A CLI argument or input is invalid; the message is printed verbatim (exit 1)."""


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    sys.stdout.write(text + "\n")


@dataclass(frozen=True, slots=True)
class BundleArgs:
    dataset: Path
    eval_report: Path
    walk_forward: Path
    out: Path
    exhibit_seed: int = 0
    generated_at: str | None = None


def _iso_utc(raw: str) -> str:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise _UsageError(f"error: --generated-at must be ISO-8601, got {raw!r}") from exc
    if parsed.tzinfo is None:
        raise _UsageError(f"error: --generated-at must carry a UTC offset, got {raw!r}")
    return raw


def parse_args(argv: Sequence[str]) -> BundleArgs:
    """Parse + validate the CLI (``SystemExit`` on an argparse usage error, :class:`_UsageError`
    with a printable message on a semantic one)."""
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.rl_bundle",
        description="Assemble the static website demo bundle (offline, read-only).",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="frozen RL dataset JSON")
    parser.add_argument("--eval", type=Path, required=True, help="rl-10 evaluation report JSON")
    parser.add_argument(
        "--walk-forward", type=Path, required=True, help="the per-fold policy directory"
    )
    parser.add_argument("--out", type=Path, required=True, help="bundle JSON path")
    parser.add_argument("--exhibit-seed", type=int, default=0, help="the ONE replayed seed")
    parser.add_argument("--generated-at", default=None, help="override meta.generated_at")
    args = parser.parse_args(list(argv))
    if args.exhibit_seed < 0:
        raise _UsageError("error: --exhibit-seed must be >= 0")
    return BundleArgs(
        dataset=Path(args.dataset),
        eval_report=Path(args.eval),
        walk_forward=Path(args.walk_forward),
        out=Path(args.out),
        exhibit_seed=int(args.exhibit_seed),
        generated_at=None if args.generated_at is None else _iso_utc(str(args.generated_at)),
    )


# --- checks ---------------------------------------------------------------------------------------


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _UsageError(f"error: cannot read {what} {path}: {exc}") from exc


def report_folds(report: Mapping[str, Any], *, n_bars: int) -> tuple[WalkForwardFold, ...]:
    """The report's folds, proven to be ``make_folds`` of this dataset at the report's geometry
    and leak-free."""
    try:
        folds = tuple(
            WalkForwardFold(
                index=int(f["index"]),
                train=(int(f["train"][0]), int(f["train"][1])),
                val=(int(f["val"][0]), int(f["val"][1])),
                test=(int(f["test"][0]), int(f["test"][1])),
                purge_bars=int(f["purge_bars"]),
            )
            for f in report["folds"]
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise _UsageError(f"error: the report's folds are malformed ({exc})") from exc
    if not folds:
        raise _UsageError("error: the report has no folds")
    first = folds[0]
    step = folds[1].train[0] - first.train[0] if len(folds) > 1 else first.test[1] - first.test[0]
    expected = make_folds(
        n_bars=n_bars,
        train_bars=first.train[1] - first.train[0],
        val_bars=first.val[1] - first.val[0],
        test_bars=first.test[1] - first.test[0],
        step_bars=step,
        purge_bars=first.purge_bars,
    )
    if folds != expected:
        raise _UsageError(
            "error: the report's folds are not make_folds of this dataset at their own geometry "
            "— refusing (the report is about a different bar span)"
        )
    for fold in folds:
        fold.assert_no_leakage(obs_window=WARMUP_BARS)
    return folds


def _seeds(report: Mapping[str, Any], policy_id: str) -> tuple[int, ...]:
    return tuple(sorted({int(r["seed"]) for r in report["results"] if r["policy_id"] == policy_id}))


def _close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _row_index(report: Mapping[str, Any]) -> dict[tuple[str, int, int, float], Mapping[str, Any]]:
    return {
        (str(r["policy_id"]), int(r["fold_index"]), int(r["seed"]), float(r["cost_bps"])): r
        for r in report["results"]
    }


def _check_row(
    rows: Mapping[tuple[str, int, int, float], Mapping[str, Any]],
    key: tuple[str, int, int, float],
    *,
    final_equity: float,
    sharpe: float,
) -> None:
    row = rows.get(key)
    if row is None:
        raise _UsageError(f"error: the report has no row for {key} — refusing")
    if not (
        _close(final_equity, float(row["final_equity"]), ROW_TOLERANCE)
        and _close(sharpe, float(row["sharpe"]), ROW_TOLERANCE)
    ):
        raise _UsageError(
            f"error: the replay of {key} gives final_equity {final_equity!r} / sharpe {sharpe!r}, "
            f"but the report recorded {row['final_equity']!r} / {row['sharpe']!r} — the inputs "
            "are not the evaluated ones; refusing"
        )


# --- the build ------------------------------------------------------------------------------------


def build(args: BundleArgs) -> tuple[bytes, dict[str, Any]]:
    """Every check + the assembly; returns ``(bundle bytes, the bundle)``. Writes nothing."""
    from trading.rl.baselines import always_flat, buy_and_hold, random_policy, sma_cross
    from trading.rl.costs import CostModel
    from trading.rl.env import DEFAULT_FEATURE_SPEC
    from trading.rl.export import mlp_policy_factory
    from trading.rl.metrics import Segment, matched_trade_probability, summarize
    from trading.rl.scoring import rollout

    raw = _read(args.dataset, "--dataset")
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        dataset = rl_dataset_from_json(raw)
        table = BarTable.from_dataset(dataset)
    except (RlDatasetError, BarTableError) as exc:
        raise _UsageError(f"error: invalid dataset {args.dataset}: {exc}") from exc
    header = json.loads(raw.decode("utf-8"))["header"]
    report_bytes = _read(args.eval_report, "--eval")
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _UsageError(f"error: --eval {args.eval_report} is not JSON: {exc}") from exc
    if report.get("dataset_sha256") != dataset_sha256:
        raise _UsageError(
            f"error: the report evaluated dataset sha256 {report.get('dataset_sha256')!r}, not "
            f"--dataset's {dataset_sha256} — refusing"
        )
    if report.get("symbol") != table.symbol or report.get("adjustment") != dataset.adjustment:
        raise _UsageError("error: the report's symbol/adjustment differ from the dataset's")
    folds = report_folds(report, n_bars=len(table))
    agent_seeds = _seeds(report, AGENT_ID)
    random_seeds = _seeds(report, RANDOM_ID)
    if args.exhibit_seed not in agent_seeds:
        raise _UsageError(
            f"error: --exhibit-seed {args.exhibit_seed} is not an evaluated {AGENT_ID!r} seed "
            f"{agent_seeds}"
        )
    if len(random_seeds) < 2:
        raise _UsageError("error: the report has fewer than 2 random seeds (no band)")
    costs = tuple(float(c) for c in report["cost_bps_grid"])
    recorded_shas: Mapping[str, str] = report["artifact_sha256s"]

    closes = table.closes()
    dates = tuple(bar.session_ts.date().isoformat() for bar in table.bars)
    # Every agent seed's per-fold policy (the envelope + the turnover-matched null need them all);
    # the exhibit seed's are the ones shipped.
    loaded_policies: dict[tuple[int, int], MlpPolicy] = {}
    policies: list[rl_bundle.ExhibitPolicy] = []
    for fold in folds:
        for seed in agent_seeds:
            name = f"{table.symbol.lower()}_{AGENT_ID}_seed{seed}.policy.json"
            path = args.walk_forward / f"fold{fold.index:02d}" / name
            key = f"{WALK_FORWARD_DIR}/fold{fold.index:02d}/{name}"
            data = _read(path, "policy")
            try:
                loaded = load_fold_policy(
                    path,
                    fold,
                    table=table,
                    dataset_sha256=dataset_sha256,
                    seed=seed,
                    untrained=False,
                )
            except ValueError as exc:  # rl_eval's _UsageError: the message is the refusal
                raise _UsageError(str(exc)) from exc
            if hashlib.sha256(data).hexdigest() != loaded.sha256:
                raise _UsageError(f"error: {path} changed while it was being read — refusing")
            if loaded.policy.features != DEFAULT_FEATURE_SPEC:
                raise _UsageError(f"error: {path.name}'s feature spec differs from the env's")
            if loaded.policy.actions != ENV_ACTION_ORDER:
                raise _UsageError(f"error: {path.name}'s action order differs from the env's")
            if recorded_shas.get(key) != loaded.sha256:
                raise _UsageError(
                    f"error: {key} has sha256 {loaded.sha256}, but the report evaluated "
                    f"{recorded_shas.get(key)!r} — refusing (not the evaluated bytes)"
                )
            loaded_policies[(fold.index, seed)] = loaded.policy
            if seed != args.exhibit_seed:
                continue
            training = json.loads(data.decode("utf-8"))["meta"].get("training", {})
            policies.append(
                rl_bundle.ExhibitPolicy(
                    fold_index=fold.index,
                    policy=loaded.policy,
                    artifact_key=key,
                    artifact_sha256=loaded.sha256,
                    run_id=str(training.get("run_id", "")),
                    training_library_versions={
                        str(k): str(v) for k, v in training.get("library_versions", {}).items()
                    },
                )
            )

    rows = _row_index(report)
    matched_id = rl_eval_matched_id(AGENT_ID)
    replays = tuple(
        rl_bundle.replay_fold(closes, fold, item.policy)
        for fold, item in zip(folds, policies, strict=True)
    )
    curves: list[rl_bundle.CostCurves] = []
    for cost in costs:
        model = CostModel(one_way_bps=cost)
        agent: list[tuple[float, ...]] = []
        seeds_eq: dict[int, list[tuple[float, ...]]] = {seed: [] for seed in agent_seeds}
        baselines: dict[str, list[tuple[float, ...]]] = {
            BUY_AND_HOLD_ID: [],
            ALWAYS_FLAT_ID: [],
            SMA_ID: [],
        }
        uniforms: dict[int, list[tuple[float, ...]]] = {seed: [] for seed in random_seeds}
        matcheds: dict[int, list[tuple[float, ...]]] = {seed: [] for seed in random_seeds}
        for fold, replay in zip(folds, replays, strict=True):
            start, length = fold.test_segment()
            segment = Segment(start=start, length=length, cost=model)

            def run(
                factory: Any, seed: int, key: tuple[str, int, int, float], segment: Any = segment
            ) -> Any:
                result = rollout(factory, table, segment, seed=seed)
                if result.ruined:
                    raise _UsageError(f"error: {key} was ruined mid-episode (ragged curve)")
                _check_row(
                    rows, key, final_equity=result.final_equity, sharpe=summarize(result).sharpe
                )
                return result

            # rl-10's order: every agent seed, then the null matched to exactly those episodes.
            agent_results = [
                run(
                    mlp_policy_factory(loaded_policies[(fold.index, seed)]),
                    seed,
                    (AGENT_ID, fold.index, seed, cost),
                )
                for seed in agent_seeds
            ]
            for seed, seed_result in zip(agent_seeds, agent_results, strict=True):
                seeds_eq[seed].append(seed_result.equity)
            result = agent_results[agent_seeds.index(args.exhibit_seed)]
            requested = tuple(ENV_ACTION_ORDER[a] for a in result.requested_actions)
            if requested != tuple(d.requested for d in replay.decisions) or tuple(
                d.position for d in replay.decisions
            ) != tuple(result.positions[:-1]):
                raise _UsageError(
                    f"error: fold {fold.index}: the stdlib replay disagrees with the env replay "
                    "(actions or positions) — the page's path would not be the evaluated one"
                )
            recomputed = rl_bundle.equity_path(closes, start, requested, one_way_bps=cost)
            if len(recomputed) != len(result.equity) or not all(
                _close(a, b, EQUITY_TOLERANCE)
                for a, b in zip(recomputed, result.equity, strict=True)
            ):
                raise _UsageError(
                    f"error: fold {fold.index} at {cost:g} bps: rl.bundle.equity_path does not "
                    "reproduce the env's equity"
                )
            agent.append(result.equity)
            for policy_id, factory in (
                (BUY_AND_HOLD_ID, buy_and_hold),
                (ALWAYS_FLAT_ID, always_flat),
                (SMA_ID, sma_cross()),
            ):
                baselines[policy_id].append(
                    run(factory, 0, (policy_id, fold.index, 0, cost)).equity
                )
            uniform = random_policy()
            matched = random_policy(matched_trade_probability(agent_results))
            for seed in random_seeds:
                uniforms[seed].append(
                    run(uniform, seed, (RANDOM_ID, fold.index, seed, cost)).equity
                )
                matcheds[seed].append(
                    run(matched, seed, (matched_id, fold.index, seed, cost)).equity
                )
        curves.append(
            rl_bundle.CostCurves(
                cost_bps=cost,
                agent=tuple(agent),
                agent_seeds={k: tuple(v) for k, v in seeds_eq.items()},
                baselines={k: tuple(v) for k, v in baselines.items()},
                uniform={k: tuple(v) for k, v in uniforms.items()},
                matched={k: tuple(v) for k, v in matcheds.items()},
            )
        )

    versions = {"python": platform.python_version()}
    for package in ("gymnasium", "numpy"):
        versions[package] = importlib.metadata.version(package)
    holdout_start = max(0, len(table) - HOLDOUT_BARS)
    inputs = rl_bundle.BundleInputs(
        dataset_header=header,
        dataset_sha256=dataset_sha256,
        session_dates=dates,
        closes=closes,
        folds=folds,
        report=report,
        report_sha256=report_sha256,
        exhibit_seed=args.exhibit_seed,
        agent_seeds=agent_seeds,
        random_seeds=random_seeds,
        policies=tuple(policies),
        replays=replays,
        curves=tuple(curves),
        generated_at=args.generated_at or str(report["as_of"]),
        generated_at_source=(
            _GENERATED_FROM_FLAG if args.generated_at is not None else _GENERATED_FROM_REPORT
        ),
        git=_git_state(),
        library_versions=versions,
        holdout_expected=any(fold.test[1] > holdout_start for fold in folds),
        agent_id=AGENT_ID,
    )
    # Size-budget priority (review 2026-10-22): the matched band and the seed envelope always
    # ship; the uniform band is dropped first if the bundle does not fit.
    for include_uniform in (True, False):
        bundle = rl_bundle.assemble(inputs, include_uniform=include_uniform)
        rl_bundle.assert_curves_test_only(bundle)
        data = rl_bundle.bundle_to_json(bundle)
        leaks = rl_bundle.secret_like(data.decode("utf-8"))
        if leaks:
            raise _UsageError(
                f"error: the bundle contains {len(leaks)} key-like / account-like / "
                f"local-path-like string(s) (e.g. {leaks[0][:12]!r}...) — refusing to write it"
            )
        try:
            rl_bundle.check_size(data)
        except rl_bundle.BundleError:
            if include_uniform:
                continue
            raise
        return data, bundle
    raise AssertionError("unreachable")  # pragma: no cover


def run_bundle(args: BundleArgs) -> int:
    """Build, check and write the bundle (atomically); print its summary."""
    inputs = {args.dataset.resolve(), args.eval_report.resolve()}
    if args.out.resolve() in inputs or args.walk_forward.resolve() in args.out.resolve().parents:
        _emit("error: --out may not overwrite an input or write inside --walk-forward")
        return 1
    try:
        data, bundle = build(args)
    except _UsageError as exc:
        _emit(str(exc))
        return 1
    except (rl_bundle.BundleError, LeakageError, RlEvaluationError) as exc:
        _emit(f"error: {exc}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(args.out)
    raw, packed = len(data), rl_bundle.gzip_size(data)
    table = bundle["eval_table"]
    _emit("=== RL demo bundle (task-rl-12) ===")
    _emit(f"wrote {args.out} ({raw} bytes raw, {packed} bytes gzip)")
    _emit(f"sha256  {hashlib.sha256(data).hexdigest()}")
    _emit(
        f"exhibit seed {bundle['meta']['exhibit_seed']}: {len(bundle['policy']['policies'])} "
        f"per-fold policies shipped, {len(bundle['policy']['parity_vectors'])} parity vectors, "
        f"{len(bundle['replay']['records'])} replay decisions, {len(bundle['curves']['dates'])} "
        "curve marks per series"
    )
    _emit(f"eval table: look #{table['source']['look']}, K={table['n_folds']} x {table['n_seeds']}")
    for agent, line in table["beats_buy_and_hold"].items():
        _emit(f"  {agent}: {line}")
    for gate in table["gates"]:
        _emit(f"  {gate['policy_id']:<14} {gate['name']:<20} {gate['verdict']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entrypoint: parse args, build, return a process exit code (0 ok / 1 fail)."""
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    except _UsageError as exc:
        _emit(str(exc))
        return 1
    try:
        return run_bundle(args)
    except Exception:
        logger.exception("rl-bundle job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
