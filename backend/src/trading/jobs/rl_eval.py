"""Walk-forward evaluation of the PPO agent against the §5.2 baselines, with the §5.6 gates
(task-rl-10; docs/RL-SLEEVE-PROGRAM.md §5).

Mirrors ``jobs/diagnostic.py``: the report dataclass -> ``dataclasses.asdict`` -> JSON (stdout, or
``--out``), a printed table, and ONE ``user.action`` audit event. The statistics live in the pure
``trading.application.rl_evaluation`` (reached through the composition root's ``rl_evaluator``);
this job produces the replay results it consumes (Tier B: ``trading.rl.scoring.rollout``, imported
lazily, gymnasium + numpy only — no torch).

USAGE::

    python -m trading.jobs.rl_eval --dataset data/rl/spy_daily.json --models artifacts/rl/ \\
        --seeds 0,1,2,3,4 --random-seeds 30 --cost-bps 0,2,5 \\
        --train-bars 504 --val-bars 126 --test-bars 63 --step-bars 63 --purge-bars 50 \\
        [--out artifacts/rl/rl_eval.json] [--confirm] \\
        [--retrain [--timesteps 200000 --episode-length 252 --train-cost-bps 1.0]]

**Per-fold retraining (program doc §5.3: "Retrain PPO once per fold").** Each fold *k* has its own
policies, trained on exactly ``fold.train`` (a rolling 504-bar window) and exported torch-free. They
live at ``<models>/walk_forward/fold{k:02d}/{symbol}_ppo[_untrained]_seed{s}.policy.json`` (the SB3
``.zip`` + training manifest next to them under ``sb3/``). ``--retrain --confirm`` creates the
missing ones (needs the ``[rl]`` extra and a dataset frozen from the REAL adapter): rl-08's
``train_ppo`` with rl-08's settings on the fold's train slice, then rl-09's export behind the SAME
parity gate (any non-tie disagreement aborts the run). Existing policies are reused, so an
interrupted retrain resumes. Evaluation itself only LOADS the JSON policies (no torch). The
pre-existing ``artifacts/rl/spy_ppo_*`` run (train ``[0, 504)``) is fold 0's training and is not
read here.

**Honest-evidence checks (fail closed).** Every fold passes ``assert_no_leakage(obs_window=50)``;
every loaded policy must record the evaluated dataset's sha256 (bar indices only mean something
within one frozen file), a train window EQUAL to its fold's ``train`` by bar index and by session
date, a ``meta.walk_forward`` identity matching its file name, and the env's feature spec — so no
in-sample window can masquerade as out-of-sample. ``train_sharpe`` is the same frozen policy
REPLAYED over its own train window (GATE-OVERFIT's in-sample number), never a training curve and
never a result.

**Dry run vs ``--confirm``.** Looking at test-window numbers is a LOOK that the multiple-testing
counter must see, so the DRY RUN computes none: it validates the folds (leakage) and the policy
inventory and prints the plan. ``--confirm`` runs the evaluation, writes the JSON (``--out`` or
stdout), prints the table and appends exactly ONE ``user.action`` (``action="rl.eval"``) carrying
``n_configurations_evaluated`` = the distinct prior looks at the same test span (keyed by test
session DATES) in the append-only file ledger ``<models>/rl_eval_looks.jsonl`` plus the event
store, + 1. The look is appended to the ledger (fsync'd) BEFORE the report is written or printed;
if that fails the job exits 1 without reporting. A synthetic dataset is allowed
for evaluation (the report records its ``source_adapter``) but never for ``--retrain --confirm``.

Baselines (rl-07, identical env replay): buy-and-hold, always-flat, SMA(20, 50), uniform random
(``--random-seeds`` seeds), and per agent a random policy TURNOVER-MATCHED to it on each fold
(``matched_trade_probability``) — the GATE-LEARN null. Exit 0 clean / 1 on failure.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from trading.application.rl_dataset import RlDataset, RlDatasetError, rl_dataset_from_json
from trading.application.rl_evaluation import (
    CORR_NOTE,
    GATE_COST_BPS,
    GATE_OVERFIT,
    GATE_RULES_VERSION,
    VS_BUY_AND_HOLD,
    GateVerdict,
    LeakageError,
    PolicyFoldResult,
    PolicySummary,
    RlEvaluationError,
    RlEvaluationReport,
    RlEvaluationRequest,
    WalkForwardFold,
    buy_and_hold_line,
    make_folds,
)
from trading.config import Settings
from trading.domain import metrics as domain_metrics
from trading.domain.events import DomainEvent, EventType
from trading.domain.rl_features import WARMUP_BARS
from trading.domain.rl_policy import MlpPolicy
from trading.infrastructure.rl.policy_store import PolicyArtifactError, policy_from_json
from trading.interface.app import AppDeps
from trading.jobs.rl_train import REAL_SOURCE_ADAPTERS, _git_state
from trading.main import build_dependencies
from trading.observability import configure_logging
from trading.rl import metrics as rl_metrics
from trading.rl.bars import BarTable, BarTableError
from trading.rl.costs import CostModel

if TYPE_CHECKING:
    from trading.rl.baselines import PolicyFactory
    from trading.rl.metrics import EpisodeResult

logger = logging.getLogger(__name__)

MIN_SEEDS: Final = 5
SEEDS_ERROR: Final = "error: at least 5 training seeds are required; single-seed results are noise"
AGENT_ID: Final = "ppo"
UNTRAINED_ID: Final = "ppo_untrained"
BUY_AND_HOLD_ID: Final = "buy_and_hold"
ALWAYS_FLAT_ID: Final = "always_flat"
RANDOM_ID: Final = "random"
SMA_ID: Final = "sma_cross_20_50"
WALK_FORWARD_DIR: Final = "walk_forward"
_USER_ACTION_STREAM: Final = "user-actions"
_ACTION: Final = "rl.eval"
_PARITY_SAMPLES: Final = 1_000
_PARITY_SEED: Final = 0
# The append-only file ledger of every --confirm evaluation (one JSON line per look), next to the
# models. It is the durable multiple-testing counter: the event store is in-memory without
# DATABASE_URL and forgets between processes.
LEDGER_NAME: Final = "rl_eval_looks.jsonl"
# Program doc §5.6: "Freeze the final 2 years as a look-once holdout".
HOLDOUT_BARS: Final = 504

EVALUATION_MODE: Final = (
    "per-fold retraining (program doc §5.3): fold k's PPO seeds were trained on exactly "
    "fold.train (rolling window), exported torch-free behind the rl-09 parity gate, and replayed "
    "over fold.test through trading.rl.scoring.rollout (Tier B); train_sharpe = the same frozen "
    "policy REPLAYED over its own train window (in-sample, GATE-OVERFIT only)"
)
NOTES: Final = (
    "val windows are unused: training is the fixed-budget rl-08 configuration (no early "
    "stopping / model selection), so no validation number exists to quote",
    "ppo_untrained = each seed's PPO at initialization (total_timesteps=0), retrained and "
    "exported per fold through the identical path; it separates 'PPO learned' from 'the env is "
    "trivially profitable'",
    "GATE-LEARN null = the random policy TURNOVER-MATCHED per fold to the agent "
    "(rl-07 matched_trade_probability); 'random' rows are the uniform (1/3) baseline",
    "risk_clipped_fraction is None (n/a on this path, NOT 0%): the Tier B env path has no "
    "RiskGate, so nothing was measured",
    f"{CORR_NOTE}: corr(bh) is a closed-form function of exposure for 0/1 position paths, so it "
    "is not independent evidence of anything exposure does not already show",
    "IQM CIs are two-stage bootstraps (folds resampled, then seeds within each drawn fold); with "
    "one run per fold the inner stage is a no-op, so single-run baselines carry a fold-only CI of "
    "the same estimator. The gates read the fold-clustered CR0 SE with t(K-1)",
    "GATE-NOTDEGENERATE's corr(bh) rule was AMENDED after look #1 on the frozen SPY span "
    "(2026-10-22 PIT review): under the old rule (an undefined corr in ANY episode vetoes) look #1 "
    "gave ppo GATE-NOTDEGENERATE FAIL and GATE-LEARN VOID; the amended rule averages |corr| over "
    "the episodes where it is defined and vetoes on corr only when none is (gate_rules_version)",
    "n_configurations_evaluated counts every recorded --confirm look at this test span (the file "
    "ledger plus the event store); it is never reset by re-running",
)


class _UsageError(ValueError):
    """A CLI argument or input is invalid; the message is printed verbatim (exit 1)."""


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    sys.stdout.write(text + "\n")


@dataclass(frozen=True, slots=True)
class EvalArgs:
    dataset: Path
    models: Path
    seeds: tuple[int, ...]
    random_seeds: int
    cost_bps: tuple[float, ...]
    train_bars: int
    val_bars: int
    test_bars: int
    step_bars: int
    purge_bars: int
    out: Path | None
    confirm: bool
    retrain: bool = False
    timesteps: int = 200_000
    episode_length: int = 252
    train_cost_bps: float = 1.0


def parse_seeds(raw: str) -> tuple[int, ...]:
    """``"0,1,2,3,4"`` -> distinct non-negative ints, at least :data:`MIN_SEEDS` of them."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise _UsageError(f"error: --seeds must be comma-separated integers, got {raw!r}") from exc
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise _UsageError(f"error: --seeds must be distinct non-negative integers, got {raw!r}")
    if len(seeds) < MIN_SEEDS:
        raise _UsageError(SEEDS_ERROR)
    return seeds


def parse_costs(raw: str) -> tuple[float, ...]:
    """``"0,2,5"`` -> the one-way bps grid; must contain the 2 bps gate cost."""
    try:
        costs = tuple(float(part) for part in raw.split(",") if part.strip())
        for cost in costs:
            CostModel(one_way_bps=cost)
    except ValueError as exc:
        raise _UsageError(f"error: --cost-bps must be valid one-way bps values: {exc}") from exc
    if len(set(costs)) != len(costs) or not costs:
        raise _UsageError(f"error: --cost-bps must be distinct values, got {raw!r}")
    if GATE_COST_BPS not in costs:
        raise _UsageError(
            f"error: --cost-bps must include the {GATE_COST_BPS:g} bps gate cost, got {raw!r}"
        )
    return costs


def parse_args(argv: Sequence[str]) -> EvalArgs:
    """Parse + validate the CLI (``SystemExit`` on an argparse usage error, :class:`_UsageError`
    with a printable message on a semantic one)."""
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.rl_eval",
        description="Walk-forward evaluation + gates of the per-fold PPO policies (offline).",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="frozen RL dataset JSON")
    parser.add_argument("--models", type=Path, required=True, help="artifact root directory")
    parser.add_argument("--seeds", required=True, help="comma-separated training seeds (>= 5)")
    parser.add_argument("--random-seeds", type=int, default=30, help="random-baseline seeds")
    parser.add_argument("--cost-bps", default="0,2,5", help="one-way cost grid in bps")
    parser.add_argument("--train-bars", type=int, default=504)
    parser.add_argument("--val-bars", type=int, default=126)
    parser.add_argument("--test-bars", type=int, default=63)
    parser.add_argument("--step-bars", type=int, default=63)
    parser.add_argument("--purge-bars", type=int, default=WARMUP_BARS)
    parser.add_argument("--out", type=Path, default=None, help="report JSON path (else stdout)")
    parser.add_argument(
        "--confirm", action="store_true", help="run the evaluation + append one audit event"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="train + export missing per-fold policies"
    )
    parser.add_argument("--timesteps", type=int, default=200_000, help="--retrain only")
    parser.add_argument("--episode-length", type=int, default=252, help="--retrain only")
    parser.add_argument("--train-cost-bps", type=float, default=1.0, help="--retrain only")
    args = parser.parse_args(list(argv))
    seeds = parse_seeds(str(args.seeds))
    if args.random_seeds < 2:
        raise _UsageError("error: --random-seeds must be >= 2 (the null is a distribution)")
    if args.timesteps < 1 or args.episode_length < 1:
        raise _UsageError("error: --timesteps and --episode-length must be >= 1")
    return EvalArgs(
        dataset=Path(args.dataset),
        models=Path(args.models),
        seeds=seeds,
        random_seeds=int(args.random_seeds),
        cost_bps=parse_costs(str(args.cost_bps)),
        train_bars=int(args.train_bars),
        val_bars=int(args.val_bars),
        test_bars=int(args.test_bars),
        step_bars=int(args.step_bars),
        purge_bars=int(args.purge_bars),
        out=None if args.out is None else Path(args.out),
        confirm=bool(args.confirm),
        retrain=bool(args.retrain),
        timesteps=int(args.timesteps),
        episode_length=int(args.episode_length),
        train_cost_bps=float(args.train_cost_bps),
    )


# --- layout + honest-evidence checks (pure) ------------------------------------------------------


def fold_dir(models: Path, fold_index: int) -> Path:
    return models / WALK_FORWARD_DIR / f"fold{fold_index:02d}"


def policy_path(models: Path, symbol: str, fold_index: int, seed: int, *, untrained: bool) -> Path:
    """``<models>/walk_forward/fold{k:02d}/{symbol}_ppo[_untrained]_seed{s}.policy.json``."""
    kind = "ppo_untrained" if untrained else "ppo"
    return fold_dir(models, fold_index) / f"{symbol.lower()}_{kind}_seed{seed}.policy.json"


def _session(table: BarTable, index: int) -> str:
    return table.bars[index].session_ts.date().isoformat()


def fold_bounds(table: BarTable, fold: WalkForwardFold) -> dict[str, dict[str, Any]]:
    """Half-open bar windows of one fold with their session dates (the export's ``meta.splits``
    shape; an empty window has no sessions)."""
    windows = {"train": fold.train, "val": fold.val, "purge": fold.purge, "test": fold.test}
    return {
        name: {
            "bars": [lo, hi],
            "first_session": _session(table, lo) if hi > lo else None,
            "last_session": _session(table, hi - 1) if hi > lo else None,
        }
        for name, (lo, hi) in windows.items()
    }


@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    policy: MlpPolicy
    sha256: str
    parity_ties: int
    parity_observations: int


def load_fold_policy(
    path: Path,
    fold: WalkForwardFold,
    *,
    table: BarTable,
    dataset_sha256: str,
    seed: int,
    untrained: bool,
) -> LoadedPolicy:
    """Load one exported policy and prove it belongs to ``fold`` of THIS dataset (fails closed):

    - ``meta.dataset.sha256`` equals the evaluated dataset's sha256 — bar indices are only
      meaningful within one frozen file (a re-freeze with bars prepended would shift every index
      and put a reused policy's "OOS" window inside its training dates);
    - its recorded train window equals ``fold.train`` by bar index AND by first/last session DATE;
    - ``meta.walk_forward`` (fold / seed / untrained) agrees with the file it was loaded from.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _UsageError(f"error: cannot read policy {path}: {exc}") from exc
    try:
        policy = policy_from_json(data)
    except PolicyArtifactError as exc:
        raise _UsageError(f"error: invalid policy artifact {path}: {exc}") from exc
    meta = json.loads(data.decode("utf-8"))["meta"]
    try:
        recorded_train = meta["splits"]["train"]
        train = recorded_train["bars"]
        sessions = (recorded_train["first_session"], recorded_train["last_session"])
        recorded_sha = meta["dataset"]["sha256"]
        walk_forward = meta["walk_forward"]
        parity = meta["parity"]
        ties, observations = int(parity["n_ties"]), int(parity["n_observations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _UsageError(
            f"error: {path} does not record its dataset, train window, walk-forward identity and "
            f"parity report ({exc})"
        ) from exc
    if recorded_sha != dataset_sha256:
        raise _UsageError(
            f"error: {path.name} was trained on dataset sha256 {recorded_sha!r}, not the evaluated "
            f"{dataset_sha256} — bar indices are only comparable within one frozen file; refusing"
        )
    if not isinstance(train, list) or [int(v) for v in train] != list(fold.train):
        raise _UsageError(
            f"error: {path.name} was trained on bars {train}, not fold {fold.index}'s train "
            f"window {list(fold.train)} — refusing (its test window would not be out-of-sample "
            "for exactly the training it records)"
        )
    expected = (_session(table, fold.train[0]), _session(table, fold.train[1] - 1))
    if tuple(sessions) != expected:
        raise _UsageError(
            f"error: {path.name} records train sessions {sessions[0]} .. {sessions[1]}, not fold "
            f"{fold.index}'s {expected[0]} .. {expected[1]} — refusing"
        )
    identity = {"fold_index": fold.index, "seed": seed, "untrained": untrained}
    if walk_forward != identity:
        raise _UsageError(
            f"error: {path.name} records walk_forward {walk_forward!r}, but its file name says "
            f"{identity} — refusing"
        )
    return LoadedPolicy(
        policy=policy,
        sha256=hashlib.sha256(data).hexdigest(),
        parity_ties=ties,
        parity_observations=observations,
    )


def span_key(table: BarTable, folds: Sequence[WalkForwardFold]) -> dict[str, Any]:
    """The identity of what this run LOOKS at, by test-session DATES (a re-frozen dataset of the
    same market span keeps its counter): same span => same multiple-testing counter."""
    return {
        "symbol": table.symbol,
        "first_test_session": _session(table, folds[0].test[0]),
        "last_test_session": _session(table, folds[-1].test[1] - 1),
        "test_bars": folds[0].test[1] - folds[0].test[0],
        "n_folds": len(folds),
    }


def ledger_path(models: Path) -> Path:
    return models / LEDGER_NAME


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every recorded look (fails closed on an unreadable or malformed ledger)."""
    if not path.exists():
        return []
    looks: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise _UsageError(f"error: cannot read the look ledger {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            look = json.loads(line)
        except ValueError as exc:
            raise _UsageError(f"error: look ledger {path} line {number} is not JSON") from exc
        if not isinstance(look, dict) or not {"report_sha256", "test_span"} <= set(look):
            raise _UsageError(f"error: look ledger {path} line {number} is not a look record")
        looks.append(look)
    return looks


def append_look(path: Path, record: Mapping[str, Any]) -> None:
    """Append one look durably (flush + fsync). Raises ``OSError``; the caller fails closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def config_record(args: EvalArgs) -> dict[str, Any]:
    """Everything that makes this evaluation THIS configuration (hashed into the ledger)."""
    return {
        "seeds": list(args.seeds),
        "random_seeds": args.random_seeds,
        "cost_bps": list(args.cost_bps),
        "train_bars": args.train_bars,
        "val_bars": args.val_bars,
        "test_bars": args.test_bars,
        "step_bars": args.step_bars,
        "purge_bars": args.purge_bars,
        "retrain": {
            "timesteps": args.timesteps,
            "episode_length": args.episode_length,
            "train_cost_bps": args.train_cost_bps,
        },
        "gate_rules": GATE_RULES_VERSION,
    }


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def holdout_note(
    table: BarTable, folds: Sequence[WalkForwardFold], *, first_look: int
) -> str | None:
    """Name the folds whose test windows fall in the §5.6 final-``HOLDOUT_BARS`` look-once
    holdout, and which look spent it."""
    start = max(0, len(table) - HOLDOUT_BARS)
    inside = [fold for fold in folds if fold.test[1] > start]
    if not inside:
        return None
    partial = [fold.index for fold in inside if fold.test[0] < start]
    return (
        f"folds {inside[0].index}-{inside[-1].index} have test windows inside the §5.6 "
        f"final-{HOLDOUT_BARS}-bar look-once holdout [{start}, {len(table)}) "
        f"({_session(table, start)} ..)"
        + (f", fold {partial[0]} partially" if partial else "")
        + f"; the holdout was SPENT by look #{first_look}, the first recorded --confirm "
        f"evaluation of this span. The pooled {len(folds)}-fold headline stands: carving those "
        "folds out after a look would itself be a post-result analysis choice"
    )


# --- replay -> PolicyFoldResult -------------------------------------------------------------------


def to_row(
    policy_id: str,
    fold_index: int,
    seed: int,
    cost_bps: float,
    result: "EpisodeResult",
    *,
    buy_and_hold: "EpisodeResult",
    train_sharpe: float | None,
) -> PolicyFoldResult:
    """One replay as the report's row (rl-07 metrics; ``domain.metrics`` volatility)."""
    score = rl_metrics.summarize(result)
    returns = rl_metrics.daily_returns(result)
    return PolicyFoldResult(
        policy_id=policy_id,
        fold_index=fold_index,
        seed=seed,
        cost_bps=cost_bps,
        total_return_pct=score.total_return_pct,
        annualized_return_pct=score.cagr_pct,
        sharpe=score.sharpe,
        sortino=score.sortino,
        volatility=domain_metrics.return_volatility(returns),
        max_drawdown_pct=score.max_drawdown_pct,
        win_rate=rl_metrics.round_trip_win_rate(result),
        annualized_turnover=score.annualized_turnover,
        exposure_fraction=score.exposure_fraction,
        n_trades=score.n_trades,
        n_decisions=score.steps,
        risk_clipped_fraction=None,  # n/a: the env path has no RiskGate (never 0%)
        train_sharpe=train_sharpe,
        bh_position_corr=rl_metrics.position_correlation(result, buy_and_hold),
        action_entropy=score.action_entropy,
        longest_holding_streak=rl_metrics.longest_holding_streak(result),
        final_equity=score.final_equity,
        break_even_cost_bps=score.break_even_cost_bps,
        frictionless_growth=None if result.ruined else rl_metrics.frictionless_growth(result),
        fill_sizes=tuple(fill.traded for fill in result.fills),
        ruined=result.ruined,
    )


def matched_id(agent_id: str) -> str:
    return f"random_matched[{agent_id}]"


def replay_all(
    table: BarTable,
    folds: Sequence[WalkForwardFold],
    policies: Mapping[tuple[int, int, bool], MlpPolicy],
    *,
    seeds: Sequence[int],
    random_seeds: Sequence[int],
    costs: Sequence[float],
) -> list[PolicyFoldResult]:
    """Every policy x fold x seed x cost through the ONE env replay path (Tier B)."""
    from trading.rl.baselines import always_flat, buy_and_hold, random_policy, sma_cross
    from trading.rl.export import mlp_policy_factory
    from trading.rl.metrics import Segment
    from trading.rl.scoring import rollout

    factories: dict[tuple[int, int, bool], PolicyFactory] = {
        key: mlp_policy_factory(policy) for key, policy in policies.items()
    }
    rows: list[PolicyFoldResult] = []
    for fold in folds:
        test_start, test_length = fold.test_segment()
        train_start, train_length = fold.train_replay_segment()
        for cost in costs:
            model = CostModel(one_way_bps=cost)
            segment = Segment(start=test_start, length=test_length, cost=model)
            in_sample = Segment(start=train_start, length=train_length, cost=model)
            bh = rollout(buy_and_hold, table, segment, seed=0)
            for policy_id, factory in (
                (BUY_AND_HOLD_ID, buy_and_hold),
                (ALWAYS_FLAT_ID, always_flat),
                (SMA_ID, sma_cross()),
            ):
                result = (
                    bh if policy_id == BUY_AND_HOLD_ID else rollout(factory, table, segment, seed=0)
                )
                rows.append(
                    to_row(
                        policy_id, fold.index, 0, cost, result, buy_and_hold=bh, train_sharpe=None
                    )
                )
            uniform = random_policy()
            for rs in random_seeds:
                result = rollout(uniform, table, segment, seed=rs)
                rows.append(
                    to_row(
                        RANDOM_ID, fold.index, rs, cost, result, buy_and_hold=bh, train_sharpe=None
                    )
                )
            for agent_id, untrained in ((AGENT_ID, False), (UNTRAINED_ID, True)):
                agent_results = []
                for seed in seeds:
                    factory = factories[(fold.index, seed, untrained)]
                    result = rollout(factory, table, segment, seed=seed)
                    replay_is = rollout(factory, table, in_sample, seed=seed)
                    agent_results.append(result)
                    rows.append(
                        to_row(
                            agent_id,
                            fold.index,
                            seed,
                            cost,
                            result,
                            buy_and_hold=bh,
                            train_sharpe=rl_metrics.summarize(replay_is).sharpe,
                        )
                    )
                matched = random_policy(rl_metrics.matched_trade_probability(agent_results))
                for rs in random_seeds:
                    result = rollout(matched, table, segment, seed=rs)
                    rows.append(
                        to_row(
                            matched_id(agent_id),
                            fold.index,
                            rs,
                            cost,
                            result,
                            buy_and_hold=bh,
                            train_sharpe=None,
                        )
                    )
    return rows


# --- retraining (lazy [rl] imports) ---------------------------------------------------------------


def retrain_missing(
    args: EvalArgs,
    *,
    table: BarTable,
    dataset: RlDataset,
    dataset_sha256: str,
    missing: Sequence[tuple[WalkForwardFold, int, bool]],
) -> list[dict[str, Any]]:
    """Train (rl-08 ``train_ppo``) and export (rl-09, parity-gated) every missing per-fold policy.
    Raises on any failure; a parity failure writes nothing for that policy."""
    from trading.infrastructure.rl.policy_store import write_policy
    from trading.jobs import rl_export as export_job
    from trading.rl import export as rl_export
    from trading.rl import train as rl_train
    from trading.rl.env import EnvConfig

    rl_train.require_rl_extra()
    env_config = EnvConfig(
        symbol=table.symbol,
        episode_length=args.episode_length,
        cost=CostModel(one_way_bps=args.train_cost_bps),
        random_start=True,
    )
    run_date = datetime.now(UTC).date()
    trained: list[dict[str, Any]] = []
    for fold, seed, untrained in missing:
        lo, hi = fold.train
        train = BarTable(symbol=table.symbol, bars=table.bars[lo:hi])
        provenance: dict[str, Any] = {
            "job": "trading.jobs.rl_eval --retrain",
            "dataset_path": str(args.dataset),
            "dataset_sha256": dataset_sha256,
            "dataset_adjustment": dataset.adjustment,
            "dataset_source_adapter": dataset.source_adapter,
            "dataset_n_bars": len(table),
            "fold_index": fold.index,
            "train_window": [lo, hi],
            "split": f"positional slice bars[{lo}:{hi}] (walk-forward fold {fold.index} train)",
            **_git_state(),
        }
        result = rl_train.train_ppo(
            train,
            env_config,
            rl_train.TrainConfig(total_timesteps=0 if untrained else args.timesteps, seed=seed),
            out_dir=fold_dir(args.models, fold.index) / "sb3",
            provenance=provenance,
            run_date=run_date,
            overwrite=True,
        )
        manifest = export_job.read_manifest(result.manifest_path, model=result.model_path)
        recorded = manifest["train_bars"]
        first, last = recorded["first_session_ts"], recorded["last_session_ts"]
        if (first, last) != (_iso(table, lo), _iso(table, hi - 1)):
            raise RlEvaluationError(f"fold {fold.index}: manifest train window {first}..{last}")
        run_id, run_day = export_job.run_identity(manifest)
        policy_id = export_job.make_policy_id(
            table.symbol, run_day, str(manifest["policy_sha256"]), untrained=untrained
        )
        spec, scales_source = export_job.feature_spec_for(manifest)
        bounds = fold_bounds(table, fold)
        model = rl_train.load_model(result.model_path)
        policy = rl_export.sb3_to_mlp_policy(model, policy_id=policy_id, spec=spec)
        sampled = rl_export.sample_observations(
            model.observation_space, _PARITY_SAMPLES, seed=_PARITY_SEED
        )
        val_obs = export_job.validation_observations(table, bounds, spec)
        report = rl_export.assert_parity(model, policy, [*sampled, *val_obs])
        parity = {
            **report.to_dict(),
            "n_sampled": len(sampled),
            "sample_seed": _PARITY_SEED,
            "n_validation": len(val_obs),
            "validation_states": "every val decision, flat and held",
        }
        meta = export_job.build_meta(
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
        splits = meta["splits"]
        if isinstance(splits, dict):
            splits["source"] = (
                f"walk-forward fold {fold.index} of trading.jobs.rl_eval (application."
                "rl_evaluation.make_folds, authoritative): the policy trained on exactly train"
            )
        meta["walk_forward"] = {"fold_index": fold.index, "untrained": untrained, "seed": seed}
        path = policy_path(args.models, table.symbol, fold.index, seed, untrained=untrained)
        sha256 = write_policy(path, policy, meta=meta)
        trained.append(
            {
                "fold": fold.index,
                "seed": seed,
                "untrained": untrained,
                "policy_id": policy_id,
                "policy_sha256": manifest["policy_sha256"],
                "artifact_sha256": sha256,
                "parity_ties": report.n_ties,
            }
        )
        _emit(
            f"  fold {fold.index:02d} seed {seed} {'untrained' if untrained else 'trained'}: "
            f"{result.num_timesteps} steps in {result.wall_seconds:.1f}s, parity "
            f"{report.n_agree}/{report.n_observations} (+{report.n_ties} ties) -> {path.name}"
        )
    return trained


def _iso(table: BarTable, index: int) -> str:
    return table.bars[index].session_ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


# --- output ---------------------------------------------------------------------------------------


def report_to_json(report: RlEvaluationReport) -> bytes:
    """``dataclasses.asdict`` -> strict compact JSON (NaN / Infinity refused)."""
    payload = asdict(report)
    payload["as_of"] = report.as_of.isoformat()
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def _num(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _ci(summary: PolicySummary) -> str:
    if summary.sharpe_iqm_ci_low is None or summary.sharpe_iqm_ci_high is None:
        return summary.sharpe_iqm_ci_note or "n/a"
    return f"[{summary.sharpe_iqm_ci_low:+.2f}, {summary.sharpe_iqm_ci_high:+.2f}]"


def _label(summary: PolicySummary) -> str:
    if summary.n_seeds > 1:
        return f"{summary.policy_id} ({summary.n_seeds} seeds, IQM)"
    return summary.policy_id


def _verdict_word(verdict: GateVerdict) -> str:
    if verdict.passed is None:
        return "VOID"
    if verdict.name == GATE_OVERFIT:
        return "OK" if verdict.passed else "WARN"
    if verdict.name == VS_BUY_AND_HOLD:
        return "YES" if verdict.passed else "NO"
    return "PASS" if verdict.passed else "FAIL"


def format_table(report: RlEvaluationReport) -> str:
    """The printed block (always, even on FAIL): a row per policy per cost, then per agent the
    four gate lines, the literal beats-buy-and-hold line, break-even / configs / parity."""
    first, last = report.folds[0], report.folds[-1]
    lines = [
        "=== RL walk-forward evaluation (task-rl-10) ===",
        f"symbol: {report.symbol}  adjustment={report.adjustment}  dataset sha256 "
        f"{report.dataset_sha256}",
        f"folds:  K={report.n_folds} (test bars [{first.test[0]}, {last.test[1]}); "
        f"train {first.train[1] - first.train[0]} / val {first.val[1] - first.val[0]} / purge "
        f"{first.purge_bars} / test {first.test[1] - first.test[0]})  seeds={report.n_seeds}  "
        f"low_n={report.low_n}",
        f"mode:   {report.evaluation_mode}",
        "",
    ]
    header = (
        f"{'POLICY':<34}{'ann.ret%':>9}{'sharpe':>8}{'sortino':>9}{'maxDD%':>8}{'turnover':>10}"
        f"{'winrate':>9}{'exposure':>10}{'trades':>8}{'streak':>8}  sharpe mean +/- SE   "
        "IQM 95% CI"
    )
    for cost in report.cost_bps_grid:
        lines.append(
            f"-- cost {cost:g} bps one-way: IQM over (fold, seed); out-of-sample test windows --"
        )
        lines.append(header)
        for s in (s for s in report.summaries if s.cost_bps == cost):
            lines.append(
                f"{_label(s):<34}{s.annualized_return_pct_iqm:>9.2f}{s.sharpe_iqm:>8.2f}"
                f"{s.sortino_iqm:>9.2f}{s.max_drawdown_pct_iqm:>8.2f}"
                f"{s.annualized_turnover_iqm:>10.2f}{_num(s.win_rate_iqm, '.2f'):>9}"
                f"{s.exposure_fraction_iqm:>10.2f}{s.n_trades_iqm:>8.1f}"
                f"{s.longest_holding_streak_iqm:>8.1f}  {s.sharpe_mean:+.2f} +/- "
                f"{_num(s.sharpe_clustered_se, '.2f'):<6}     {_ci(s)}"
            )
        lines.append("")
    agents: list[str] = []
    for verdict in report.gates:
        if verdict.policy_id not in agents:
            agents.append(verdict.policy_id)
    for agent in agents:
        lines.append(f"-- gates for {agent} at {report.gate_cost_bps:g} bps --")
        for verdict in (v for v in report.gates if v.policy_id == agent):
            detail = verdict.detail
            if verdict.name == VS_BUY_AND_HOLD:
                detail = detail.rsplit(" | ", 1)[0]
            lines.append(f"{verdict.name:<20}{_verdict_word(verdict):<6}{detail}")
        bh = next(v for v in report.gates if v.policy_id == agent and v.name == VS_BUY_AND_HOLD)
        lines.append(buy_and_hold_line(bh))
        lines.append(
            f"break-even cost     {_num(report.break_even_cost_bps.get(agent), '.2f')} bps "
            f"(vs always-flat)  |  configs evaluated: {report.n_configurations_evaluated}  |  "
            f"parity ties: {report.parity_ties}/{report.parity_observations}"
        )
        edges = (report.edge_over_flat_log_growth or {}).get(agent, {})
        if edges:
            lines.append(
                "edge vs always-flat (mean log-growth per episode) "
                + "  ".join(f"@{cost} bps {_num(value, '+.4f')}" for cost, value in edges.items())
            )
        lines.append("")
    lines.append(
        f"configs evaluated: {report.n_configurations_evaluated} — counted from "
        f"{report.n_configurations_source}"
    )
    lines.append(f"gate rules: {report.gate_rules_version}")
    for warning in report.warnings:
        lines.append(warning)
    for note in report.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)


# --- the job --------------------------------------------------------------------------------------


async def run_eval(args: EvalArgs) -> int:
    """Guard, (optionally) retrain, evaluate, report and audit. Teardown always runs."""
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
        folds = make_folds(
            n_bars=len(table),
            train_bars=args.train_bars,
            val_bars=args.val_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            purge_bars=args.purge_bars,
        )
        if not folds:
            raise RlEvaluationError(f"the dataset ({len(table)} bars) is too short for one fold")
        for fold in folds:
            fold.assert_no_leakage(obs_window=WARMUP_BARS)
    except (RlDatasetError, BarTableError) as exc:
        _emit(f"error: invalid dataset {args.dataset}: {exc}")
        return 1
    except LeakageError as exc:
        _emit(f"LEAKAGE: {exc}")
        return 1
    except RlEvaluationError as exc:
        _emit(f"error: {exc}")
        return 1

    wanted = [
        (fold, seed, untrained)
        for fold in folds
        for seed in args.seeds
        for untrained in (False, True)
    ]
    missing = [
        (fold, seed, untrained)
        for fold, seed, untrained in wanted
        if not policy_path(
            args.models, table.symbol, fold.index, seed, untrained=untrained
        ).exists()
    ]
    synthetic = dataset.source_adapter not in REAL_SOURCE_ADAPTERS
    _emit("=== RL walk-forward evaluation (task-rl-10) ===")
    _emit(f"mode:     {'CONFIRM (evaluates + records a look)' if args.confirm else 'DRY-RUN'}")
    _emit(f"dataset:  {args.dataset}  (sha256 {dataset_sha256}, {len(table)} bars)")
    _emit(f"source:   {dataset.source_adapter}, adjustment={dataset.adjustment}")
    if synthetic:
        _emit("WARNING:  SYNTHETIC / non-allowlisted dataset — numbers describe fake prices.")
    _emit(
        f"folds:    K={len(folds)} — test decisions {_session(table, folds[0].test[0])} .. "
        f"{_session(table, folds[-1].test[1] - 1)} (leakage asserted per fold, obs window "
        f"{WARMUP_BARS})"
    )
    _emit(f"policies: {len(wanted) - len(missing)}/{len(wanted)} per-fold exports present")

    if missing and not args.retrain:
        fold, seed, untrained = missing[0]
        example = policy_path(args.models, table.symbol, fold.index, seed, untrained=untrained)
        _emit(
            f"error: {len(missing)} per-fold policy export(s) are missing under "
            f"{args.models / WALK_FORWARD_DIR} (e.g. {example}). Re-run with --retrain "
            "--confirm to train + export them (program doc §5.3)."
        )
        return 1
    if not args.confirm:
        if missing:
            _emit(f"--retrain would train + export {len(missing)} policies (needs --confirm).")
        for fold in folds:
            _emit(
                f"  fold {fold.index:02d}: train {list(fold.train)} val {list(fold.val)} purge "
                f"{list(fold.purge)} test {list(fold.test)} ({_session(table, fold.test[0])} .. "
                f"{_session(table, fold.test[1] - 1)})"
            )
        _emit("")
        _emit(
            "DRY-RUN: no test-window number was computed (each one is a LOOK the multiple-testing "
            "counter must record). Re-run with --confirm to evaluate."
        )
        return 0
    if missing and synthetic:
        _emit(
            "error: --retrain --confirm requires a dataset frozen from the REAL historical-bars "
            f"adapter ({', '.join(sorted(REAL_SOURCE_ADAPTERS))}); got "
            f"source_adapter={dataset.source_adapter!r}"
        )
        return 1

    deps: AppDeps | None = None
    try:
        retrained: list[dict[str, Any]] = []
        if missing:
            _emit(f"retraining {len(missing)} per-fold policies (rl-08 config, rl-09 parity) ...")
            try:
                retrained = retrain_missing(
                    args,
                    table=table,
                    dataset=dataset,
                    dataset_sha256=dataset_sha256,
                    missing=missing,
                )
            except (ImportError, RuntimeError) as exc:
                _emit(f"error: {exc}")
                return 1

        policies: dict[tuple[int, int, bool], MlpPolicy] = {}
        shas: dict[str, str] = {}
        model_ids: list[str] = []
        ties = observations = 0
        from trading.rl.env import DEFAULT_FEATURE_SPEC

        for fold, seed, untrained in wanted:
            path = policy_path(args.models, table.symbol, fold.index, seed, untrained=untrained)
            loaded = load_fold_policy(
                path,
                fold,
                table=table,
                dataset_sha256=dataset_sha256,
                seed=seed,
                untrained=untrained,
            )
            if loaded.policy.features != DEFAULT_FEATURE_SPEC:
                raise _UsageError(
                    f"error: {path.name}'s feature spec differs from the env's (the replay would "
                    "feed it observations it was not trained on)"
                )
            policies[(fold.index, seed, untrained)] = loaded.policy
            shas[str(path.relative_to(args.models))] = loaded.sha256
            model_ids.append(loaded.policy.policy_id)
            ties += loaded.parity_ties
            observations += loaded.parity_observations

        deps = build_dependencies(settings)
        if deps.rl_evaluator is None:  # defensive: the composition root always wires it
            raise RuntimeError("rl_evaluator is not configured")
        span = span_key(table, folds)
        ledger = ledger_path(args.models)
        ledger_looks = [look for look in read_ledger(ledger) if look["test_span"] == span]
        event_looks = [
            stored.event.payload.get("report_sha256")
            for stored in await deps.event_store.read_stream(_USER_ACTION_STREAM)
            if stored.event.payload.get("action") == _ACTION
            and stored.event.payload.get("test_span") == span
        ]
        prior = {str(look["report_sha256"]) for look in ledger_looks} | {
            str(sha) for sha in event_looks
        }
        n_configurations = len(prior) + 1
        counter_source = (
            f"the file ledger {ledger} ({len(ledger_looks)} prior look(s) on this span) + the "
            f"event store ({len(event_looks)}; "
            + ("durable Postgres" if settings.database_url else "IN-MEMORY: forgets on exit")
            + ")"
        )
        first_look = min(
            (int(look.get("look", n_configurations)) for look in ledger_looks),
            default=n_configurations,
        )
        holdout = holdout_note(table, folds, first_look=first_look)

        _emit(
            f"replaying {len(folds)} folds x {len(args.cost_bps)} costs "
            f"({len(args.seeds)} seeds x 2 PPO groups, {args.random_seeds} random seeds x 3) ..."
        )
        rows = replay_all(
            table,
            folds,
            policies,
            seeds=args.seeds,
            random_seeds=tuple(range(args.random_seeds)),
            costs=args.cost_bps,
        )
        request = RlEvaluationRequest(
            symbol=table.symbol,
            adjustment=dataset.adjustment,
            dataset_sha256=dataset_sha256,
            folds=folds,
            results=tuple(rows),
            agent_ids=(AGENT_ID, UNTRAINED_ID),
            matched_random_ids={a: matched_id(a) for a in (AGENT_ID, UNTRAINED_ID)},
            buy_and_hold_id=BUY_AND_HOLD_ID,
            model_ids=tuple(model_ids),
            artifact_sha256s=shas,
            cost_bps_grid=args.cost_bps,
            n_configurations_evaluated=n_configurations,
            parity_ties=ties,
            parity_observations=observations,
            evaluation_mode=EVALUATION_MODE,
            gate_cost_bps=GATE_COST_BPS,
            obs_window=WARMUP_BARS,
            n_configurations_source=counter_source,
            notes=NOTES
            + ((holdout,) if holdout is not None else ())
            + (
                (f"SYNTHETIC dataset (source_adapter={dataset.source_adapter!r})",)
                if synthetic
                else ()
            ),
        )
        report = await deps.rl_evaluator(request)
        data = report_to_json(report)
        report_sha256 = hashlib.sha256(data).hexdigest()
        config = config_record(args)
        # Record the LOOK before the report is written or printed; fail closed if it cannot be.
        try:
            append_look(
                ledger,
                {
                    "look": n_configurations,
                    "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "as_of": report.as_of.isoformat(),
                    "dataset_sha256": dataset_sha256,
                    "test_span": span,
                    "config": config,
                    "config_sha256": canonical_sha256(config),
                    "report_sha256": report_sha256,
                    "results_sha256": canonical_sha256(json.loads(data)["results"]),
                    "out": None if args.out is None else str(args.out),
                    "gates": {f"{v.policy_id}:{v.name}": _verdict_word(v) for v in report.gates},
                    "note": f"--confirm evaluation ({GATE_RULES_VERSION})",
                },
            )
        except (OSError, ValueError) as exc:
            _emit(f"error: cannot record this look in {ledger}: {exc} — refusing to report")
            return 1
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(data)
            _emit(f"wrote {args.out} ({len(data)} bytes, sha256 {report_sha256})")
        else:
            _emit(data.decode("utf-8").rstrip("\n"))
        _emit("")
        _emit(format_table(report))

        await deps.event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": _ACTION,
                    "symbol": table.symbol,
                    "dataset_sha256": dataset_sha256,
                    "test_span": span,
                    "n_configurations_evaluated": n_configurations,
                    "n_folds": report.n_folds,
                    "n_seeds": report.n_seeds,
                    "low_n": report.low_n,
                    "random_seeds": args.random_seeds,
                    "cost_bps_grid": list(args.cost_bps),
                    "gates": {f"{v.policy_id}:{v.name}": _verdict_word(v) for v in report.gates},
                    "break_even_cost_bps": {
                        key: value
                        for key, value in report.break_even_cost_bps.items()
                        if key in (AGENT_ID, UNTRAINED_ID)
                    },
                    "n_policies": len(model_ids),
                    "report_sha256": report_sha256,
                    "out": None if args.out is None else str(args.out),
                    "retrained": retrained,
                    "evaluation_mode": EVALUATION_MODE,
                },
                occurred_at=deps.clock.now(),
                correlation_id=str(uuid4()),
            )
        )
        return 0
    except _UsageError as exc:
        _emit(str(exc))
        return 1
    except (LeakageError, RlEvaluationError) as exc:
        _emit(f"error: {exc}")
        return 1
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
    """Synchronous entrypoint: parse args, evaluate, return a process exit code (0 ok / 1 fail)."""
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    except _UsageError as exc:
        _emit(str(exc))
        return 1
    try:
        return asyncio.run(run_eval(args))
    except Exception:
        logger.exception("rl-eval job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
