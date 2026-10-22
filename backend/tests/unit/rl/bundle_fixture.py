"""A complete, SYNTHETIC task-rl-12 input layout (shared by the bundle unit tests and the job e2e).

Plain helpers (the repo's ``conftest`` pattern), not fixtures. The layout mirrors the real one:

- a frozen dataset written by the real ``rl_dataset_to_json`` (``adjustment=all``, a distinctive
  ``generated_at`` with microseconds, so provenance must be copied VERBATIM to match);
- per-fold policies in ``<models>/walk_forward/foldNN/`` written through the real ``policy_store``,
  each carrying the provenance ``jobs/rl_eval.load_fold_policy`` demands; weights are float32
  values (as every real export's are: SB3 trains in float32);
- an rl-10 report produced by the REAL replay + statistics code (``jobs/rl_eval.replay_all`` ->
  ``application/rl_evaluation.build_rl_report`` -> ``report_to_json``), with the single-run
  baselines' sharpe IQM CIs then set to ``None`` exactly as look #2's committed report has them
  (the bundle's one allowed recompute); the CIs the current code computed are returned so a test
  can prove the recompute reproduces them;
- a look ledger next to the models (the bundle job must never touch it).

Imports ``trading.rl.scoring`` through ``replay_all`` (gymnasium + numpy, both in ``[dev]``).
"""

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trading.application.rl_dataset import RlBar, RlDataset, rl_dataset_to_json, session_known_at
from trading.application.rl_evaluation import (
    RlEvaluationRequest,
    build_rl_report,
    make_folds,
)
from trading.domain.rl_features import WARMUP_BARS
from trading.domain.rl_policy import ENV_ACTION_ORDER, DenseLayer, MlpPolicy
from trading.infrastructure.rl.policy_store import write_policy
from trading.jobs import rl_eval
from trading.rl.bars import BarTable
from trading.rl.bundle import fround

_NY = ZoneInfo("America/New_York")
GENERATED_AT = datetime(2026, 10, 22, 2, 14, 43, 134336, tzinfo=UTC)
AS_OF = datetime(2026, 10, 22, 10, 14, 10, 596488, tzinfo=UTC)
N_BARS = 400
GEOMETRY = {"train_bars": 100, "val_bars": 20, "test_bars": 20, "step_bars": 20}
SEEDS = (0, 1, 2, 3, 4)
RANDOM_SEEDS = (0, 1, 2, 3, 4)
COSTS = (0.0, 2.0, 5.0)
SINGLE_RUN = ("buy_and_hold", "always_flat", "sma_cross_20_50")


@dataclass(frozen=True, slots=True)
class Layout:
    root: Path
    dataset: Path
    report: Path
    models: Path
    walk_forward: Path
    ledger: Path
    n_folds: int
    # (policy_id, cost) -> the sharpe IQM CI the current rl-10 code computed before it was nulled
    computed_cis: dict[tuple[str, float], tuple[float, float]]

    def argv(self, out: Path, *extra: str) -> list[str]:
        return [
            "--dataset",
            str(self.dataset),
            "--eval",
            str(self.report),
            "--walk-forward",
            str(self.walk_forward),
            "--out",
            str(out),
            *extra,
        ]


def dataset_bytes(n: int = N_BARS, *, seed: int = 11) -> bytes:
    rng = random.Random(seed)
    day, sessions = date(2019, 1, 2), list[datetime]()
    while len(sessions) < n:
        if day.weekday() < 5:
            sessions.append(datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC))
        day += timedelta(days=1)
    closes = [250.0]
    for _ in range(n - 1):
        closes.append(round(closes[-1] * (1.0 + rng.gauss(0.0004, 0.012)), 2))
    bars = tuple(
        RlBar(session_ts=s, known_at=session_known_at(s), close=c)
        for s, c in zip(sessions, closes, strict=True)
    )
    return rl_dataset_to_json(
        RlDataset(
            symbol="SPY",
            bars=bars,
            adjustment="all",
            source_adapter="AlpacaHistoricalData",
            start=sessions[0] - timedelta(days=1),
            end=GENERATED_AT,
            generated_at=GENERATED_AT,
        )
    )


def float32_policy(seed: int, policy_id: str) -> MlpPolicy:
    """A small 9 -> 8 -> 8 -> 3 tanh net with float32 weights (a real export's dtype)."""
    from trading.rl.env import DEFAULT_FEATURE_SPEC

    rng = random.Random(seed)

    def matrix(rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(fround(rng.uniform(-1.0, 1.0)) for _ in range(cols)) for _ in range(rows)
        )

    def bias(n: int) -> tuple[float, ...]:
        return tuple(fround(rng.uniform(-0.5, 0.5)) for _ in range(n))

    return MlpPolicy(
        policy_id=policy_id,
        schema_version=1,
        features=DEFAULT_FEATURE_SPEC,
        hidden=(
            DenseLayer(weight=matrix(8, 9), bias=bias(8), activation="tanh"),
            DenseLayer(weight=matrix(8, 8), bias=bias(8), activation="tanh"),
        ),
        action_head=DenseLayer(weight=matrix(3, 8), bias=bias(3), activation="identity"),
        actions=ENV_ACTION_ORDER,
    )


def build_layout(root: Path) -> Layout:
    root.mkdir(parents=True, exist_ok=True)
    data = dataset_bytes()
    dataset = root / "spy_daily.json"
    dataset.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    from trading.application.rl_dataset import rl_dataset_from_json

    table = BarTable.from_dataset(rl_dataset_from_json(data))
    folds = make_folds(n_bars=len(table), **GEOMETRY)
    models = root / "models"
    policies: dict[tuple[int, int, bool], MlpPolicy] = {}
    shas: dict[str, str] = {}
    for fold in folds:
        lo, hi = fold.train
        for seed in SEEDS:
            for untrained in (False, True):
                kind = "ppo-untrained" if untrained else "ppo"
                policy = float32_policy(
                    1000 * fold.index + 10 * seed + untrained, f"spy-{kind}-f{fold.index}-s{seed}"
                )
                meta: dict[str, Any] = {
                    "dataset": {"sha256": sha},
                    "splits": {
                        "train": {
                            "bars": [lo, hi],
                            "first_session": table.bars[lo].session_ts.date().isoformat(),
                            "last_session": table.bars[hi - 1].session_ts.date().isoformat(),
                        }
                    },
                    "walk_forward": {
                        "fold_index": fold.index,
                        "seed": seed,
                        "untrained": untrained,
                    },
                    "parity": {"n_ties": 0, "n_observations": 100},
                    "training": {
                        "run_id": f"spy_{kind}_fixture_seed{seed}",
                        "library_versions": {"sb3": "x", "torch": "y"},
                    },
                }
                path = rl_eval.policy_path(models, "SPY", fold.index, seed, untrained=untrained)
                shas[str(path.relative_to(models))] = write_policy(path, policy, meta=meta)
                policies[(fold.index, seed, untrained)] = policy
    rows = rl_eval.replay_all(
        table, folds, policies, seeds=SEEDS, random_seeds=RANDOM_SEEDS, costs=COSTS
    )
    notes = rl_eval.NOTES
    holdout = rl_eval.holdout_note(table, folds, first_look=1)
    request = RlEvaluationRequest(
        symbol="SPY",
        adjustment="all",
        dataset_sha256=sha,
        folds=folds,
        results=tuple(rows),
        agent_ids=(rl_eval.AGENT_ID, rl_eval.UNTRAINED_ID),
        matched_random_ids={a: rl_eval.matched_id(a) for a in ("ppo", "ppo_untrained")},
        buy_and_hold_id=rl_eval.BUY_AND_HOLD_ID,
        model_ids=tuple(p.policy_id for p in policies.values()),
        artifact_sha256s=shas,
        cost_bps_grid=COSTS,
        n_configurations_evaluated=2,
        parity_ties=0,
        parity_observations=100 * len(policies),
        evaluation_mode=rl_eval.EVALUATION_MODE,
        obs_window=WARMUP_BARS,
        n_configurations_source="the file ledger models/rl_eval_looks.jsonl (1 prior look(s))",
        notes=notes + ((holdout,) if holdout is not None else ()),
    )
    report = json.loads(rl_eval.report_to_json(build_rl_report(request, as_of=AS_OF)))
    computed: dict[tuple[str, float], tuple[float, float]] = {}
    for summary in report["summaries"]:
        if summary["policy_id"] in SINGLE_RUN:
            computed[(summary["policy_id"], summary["cost_bps"])] = (
                summary["sharpe_iqm_ci_low"],
                summary["sharpe_iqm_ci_high"],
            )
            summary["sharpe_iqm_ci_low"] = summary["sharpe_iqm_ci_high"] = None
            summary["sharpe_iqm_ci_note"] = "n/a: 1 run per fold"
    report_path = root / "rl_eval.json"
    report_path.write_bytes(
        (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    )
    ledger = rl_eval.ledger_path(models)
    ledger.write_text(
        json.dumps({"look": 2, "report_sha256": "0" * 64, "test_span": {}}) + "\n",
        encoding="utf-8",
    )
    return Layout(
        root=root,
        dataset=dataset,
        report=report_path,
        models=models,
        walk_forward=models / rl_eval.WALK_FORWARD_DIR,
        ledger=ledger,
        n_folds=len(folds),
        computed_cis=computed,
    )


def tree_digest(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root`` (relative path -> hash): proves inputs are untouched."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
