"""RL walk-forward evaluation job end-to-end on a SYNTHETIC dataset (task-rl-10, acceptance 11).

The real composition root runs (DATABASE_URL cleared => in-memory event store); the env replays run
for real through ``trading.rl.scoring.rollout`` (gymnasium + numpy, both in ``[dev]``). The per-fold
policies are small hand-built torch-free MLPs written through the real ``policy_store``, so the
default run needs no ``[rl]`` extra; the ``--retrain`` path (SB3 training + parity-gated export) is
covered by an ``rl``-marked test.
"""

import hashlib
import json
import random
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from trading.application.rl_dataset import (
    RlBar,
    RlDataset,
    rl_dataset_from_json,
    rl_dataset_to_json,
    session_known_at,
)
from trading.application.rl_evaluation import GATE_NAMES, make_folds
from trading.config import Settings
from trading.domain.events import EventType
from trading.domain.rl_policy import ENV_ACTION_ORDER, DenseLayer, MlpPolicy
from trading.infrastructure.rl.policy_store import write_policy
from trading.interface.app import AppDeps
from trading.jobs import rl_eval as job
from trading.main import build_dependencies
from trading.rl.bars import BarTable

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "DATABASE_URL",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "REQUIRE_REAL_DATA",
    "ENABLE_LIVE_QUOTES",
    "ENABLE_HISTORICAL_DATA",
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
    "API_KEY",
)
_NY = ZoneInfo("America/New_York")
_NOW = datetime(2026, 10, 22, tzinfo=UTC)
# Small geometry over 400 synthetic bars: train 100 / val 20 / purge 50 / test 20 => K = 11.
_GEOMETRY = ("--train-bars", "100", "--val-bars", "20", "--test-bars", "20", "--step-bars", "20")
_SEEDS = (0, 1, 2, 3, 4)
_POLICY_ROWS = (
    "ppo (5 seeds, IQM)",
    "ppo_untrained (5 seeds, IQM)",
    "buy_and_hold",
    "always_flat",
    "random (5 seeds, IQM)",
    "sma_cross_20_50",
    "random_matched[ppo] (5 seeds, IQM)",
    "random_matched[ppo_untrained] (5 seeds, IQM)",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(job, "Settings", lambda: Settings(_env_file=None))


def _dataset(n: int, *, adapter: str = "SyntheticHistoricalPrices", seed: int = 7) -> bytes:
    rng = random.Random(seed)
    day, sessions = date(2019, 1, 2), list[datetime]()
    while len(sessions) < n:
        if day.weekday() < 5:
            sessions.append(datetime.combine(day, time(0), tzinfo=_NY).astimezone(UTC))
        day += timedelta(days=1)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1.0 + rng.gauss(0.0004, 0.012)))
    bars = tuple(
        RlBar(session_ts=s, known_at=session_known_at(s), close=c)
        for s, c in zip(sessions, closes, strict=True)
    )
    return rl_dataset_to_json(
        RlDataset(
            symbol="SPY",
            bars=bars,
            adjustment="all",
            source_adapter=adapter,
            start=sessions[0] - timedelta(days=1),
            end=_NOW,
            generated_at=_NOW,
        )
    )


def _policy(seed: int, policy_id: str) -> MlpPolicy:
    from trading.rl.env import DEFAULT_FEATURE_SPEC

    rng = random.Random(seed)

    def matrix(rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(rng.uniform(-1.0, 1.0) for _ in range(cols)) for _ in range(rows))

    return MlpPolicy(
        policy_id=policy_id,
        schema_version=1,
        features=DEFAULT_FEATURE_SPEC,
        hidden=(DenseLayer(weight=matrix(8, 9), bias=(0.0,) * 8, activation="tanh"),),
        action_head=DenseLayer(weight=matrix(3, 8), bias=(0.0, 0.0, 0.0), activation="identity"),
        actions=ENV_ACTION_ORDER,
    )


Tamper = Callable[[int, int, bool, dict[str, Any]], None]


def _layout(
    tmp_path: Path, *, n: int = 400, tamper: Tamper | None = None
) -> tuple[Path, Path, int]:
    """A synthetic dataset + one exported policy per (fold, seed, untrained), each carrying the
    provenance a real rl-10 export records (dataset sha, train window by index AND session date,
    walk-forward identity, parity). ``tamper(fold, seed, untrained, meta)`` may corrupt it."""
    data = _dataset(n)
    dataset = tmp_path / "spy.json"
    dataset.write_bytes(data)
    models = tmp_path / "models"
    table = BarTable.from_dataset(rl_dataset_from_json(data))
    folds = make_folds(n_bars=len(table), train_bars=100, val_bars=20, test_bars=20, step_bars=20)
    sha = hashlib.sha256(data).hexdigest()
    for fold in folds:
        for seed in _SEEDS:
            for untrained in (False, True):
                lo, hi = fold.train
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
                }
                if tamper is not None:
                    tamper(fold.index, seed, untrained, meta)
                write_policy(
                    job.policy_path(models, "SPY", fold.index, seed, untrained=untrained),
                    _policy(1000 * fold.index + 10 * seed + untrained, f"spy-test-{seed}"),
                    meta=meta,
                )
    return dataset, models, len(folds)


def _args(dataset: Path, models: Path, *extra: str) -> list[str]:
    return [
        "--dataset",
        str(dataset),
        "--models",
        str(models),
        "--seeds",
        "0,1,2,3,4",
        "--random-seeds",
        "5",
        "--cost-bps",
        "0,2,5",
        *_GEOMETRY,
        *extra,
    ]


def _shared_deps(monkeypatch: pytest.MonkeyPatch) -> list[AppDeps]:
    """Every run in a test gets the SAME deps (one in-memory event store), so the
    multiple-testing counter can be observed across runs."""
    built: list[AppDeps] = []

    def build(settings: Settings, **kwargs: Any) -> AppDeps:
        if not built:
            deps = build_dependencies(settings, **kwargs)
            deps.aclosers.clear()  # keep the shared in-memory store usable across runs
            built.append(deps)
        return built[0]

    monkeypatch.setattr(job, "build_dependencies", build)
    return built


async def test_full_job_prints_a_row_per_policy_the_four_gates_and_one_audit_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    built = _shared_deps(monkeypatch)
    dataset, models, k = _layout(tmp_path)
    out = tmp_path / "rl_eval.json"

    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--out", str(out), "--confirm")))
    printed = capsys.readouterr().out
    assert rc == 0, printed

    lines = printed.splitlines()
    for label in _POLICY_ROWS:  # a row per policy, at every cost
        assert sum(1 for line in lines if line.startswith(label + " ")) == 3, label
    for name in GATE_NAMES:  # the four gate lines, once per gated agent (ppo, ppo_untrained)
        assert sum(1 for line in lines if line.startswith(name)) == 2, name
    assert sum(1 for line in lines if line.startswith("beats buy-and-hold: ")) == 2
    assert "SYNTHETIC" in printed

    report = json.loads(out.read_bytes())
    assert report["n_folds"] == k == 11
    assert report["n_seeds"] == 5 and report["low_n"] is False
    assert report["n_configurations_evaluated"] == 1
    assert report["cost_bps_grid"] == [0.0, 2.0, 5.0]
    assert len(report["gates"]) == 8
    assert set(report["break_even_cost_bps"]) >= {"ppo", "ppo_untrained"}
    assert len(report["model_ids"]) == 11 * 5 * 2
    # Per fold x 3 costs: 2 PPO groups x 5 seeds + 3 deterministic + 3 random groups x 5 seeds.
    assert len(report["results"]) == k * 3 * (10 + 3 + 15)
    ppo_rows = [r for r in report["results"] if r["policy_id"] == "ppo"]
    assert all(r["train_sharpe"] is not None for r in ppo_rows)  # IS replay recorded
    assert all(r["n_decisions"] == 20 for r in ppo_rows)  # OOS decisions = the test window
    # Pin: the IS replay runs on the TRAIN window, not the test segment (it would equal sharpe).
    assert any(r["train_sharpe"] != r["sharpe"] for r in ppo_rows)
    assert all(r["risk_clipped_fraction"] is None for r in report["results"])  # n/a, not 0%
    # Pin: GATE-LEARN's null is the TURNOVER-MATCHED random (mean(d) = CI midpoint), not uniform.
    learn = next(
        g for g in report["gates"] if g["policy_id"] == "ppo" and g["name"] == "GATE-LEARN"
    )
    at2 = [r for r in report["results"] if r["cost_bps"] == 2.0]
    diffs = []
    for fold in range(k):
        null = [
            r["sharpe"]
            for r in at2
            if r["policy_id"] == "random_matched[ppo]" and r["fold_index"] == fold
        ]
        diffs += [
            r["sharpe"] - sum(null) / len(null)
            for r in at2
            if r["policy_id"] == "ppo" and r["fold_index"] == fold
        ]
    assert (learn["ci_low"] + learn["ci_high"]) / 2 == pytest.approx(sum(diffs) / len(diffs))
    bh = next(s for s in report["summaries"] if s["policy_id"] == "buy_and_hold")
    assert bh["sharpe_iqm_ci_low"] is not None and bh["sharpe_iqm_ci_note"] == ""
    assert any(
        "look-once holdout" in note and "SPENT by look #1" in note for note in report["notes"]
    )

    events = await built[0].event_store.read_all()
    audits = [e.event for e in events if e.event.event_type == EventType.USER_ACTION]
    assert len(audits) == 1
    assert audits[0].payload["action"] == "rl.eval"
    assert audits[0].payload["n_configurations_evaluated"] == 1

    # A second look at the same test span is counted.
    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--out", str(out), "--confirm")))
    assert rc == 0
    events = await built[0].event_store.read_all()
    audits = [e.event for e in events if e.event.event_type == EventType.USER_ACTION]
    assert [a.payload["n_configurations_evaluated"] for a in audits] == [1, 2]
    assert json.loads(out.read_bytes())["n_configurations_evaluated"] == 2
    looks = [json.loads(line) for line in job.ledger_path(models).read_text().splitlines()]
    assert [look["look"] for look in looks] == [1, 2]
    assert looks[1]["report_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert looks[0]["results_sha256"] == looks[1]["results_sha256"]  # same replays, twice


async def test_the_file_ledger_counts_looks_across_processes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A FRESH composition root (fresh in-memory store) per run, as two separate processes get:
    # only the file ledger remembers the first look.
    dataset, models, _ = _layout(tmp_path)
    out = tmp_path / "rl_eval.json"
    for expected in (1, 2):
        rc = await job.run_eval(
            job.parse_args(_args(dataset, models, "--out", str(out), "--confirm"))
        )
        assert rc == 0
        assert json.loads(out.read_bytes())["n_configurations_evaluated"] == expected
    assert "1 prior look(s) on this span" in capsys.readouterr().out


async def test_a_look_that_cannot_be_recorded_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    built = _shared_deps(monkeypatch)
    dataset, models, _ = _layout(tmp_path)
    out = tmp_path / "rl_eval.json"

    def broken(path: Path, record: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(job, "append_look", broken)
    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--out", str(out), "--confirm")))
    printed = capsys.readouterr().out
    assert rc == 1
    assert "cannot record this look" in printed
    assert not out.exists() and "GATE-LEARN" not in printed
    events = await built[0].event_store.read_all()
    assert [e for e in events if e.event.event_type == EventType.USER_ACTION] == []


async def test_dry_run_computes_no_test_numbers_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    built = _shared_deps(monkeypatch)
    dataset, models, _ = _layout(tmp_path)
    out = tmp_path / "rl_eval.json"

    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--out", str(out))))
    printed = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in printed and "fold 10:" in printed
    assert not out.exists()
    assert built == []  # never even built the composition root


async def test_a_policy_trained_on_another_window_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _shared_deps(monkeypatch)

    def fold_zero(fold: int, seed: int, untrained: bool, meta: dict[str, Any]) -> None:
        meta["splits"]["train"]["bars"] = [0, 100]  # every fold claims fold 0's window

    dataset, models, _ = _layout(tmp_path, tamper=fold_zero)

    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--confirm")))
    printed = capsys.readouterr().out
    assert rc == 1
    assert "not fold 1's train window" in printed


def _sha(fold: int, seed: int, untrained: bool, meta: dict[str, Any]) -> None:
    meta["dataset"]["sha256"] = "f" * 64  # trained on another freeze of the bars


def _sessions(fold: int, seed: int, untrained: bool, meta: dict[str, Any]) -> None:
    meta["splits"]["train"]["first_session"] = "2001-01-02"


def _identity(fold: int, seed: int, untrained: bool, meta: dict[str, Any]) -> None:
    if fold == 4 and seed == 2 and not untrained:
        meta["walk_forward"]["seed"] = 3  # the file says seed 2


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_sha, "bar indices are only comparable within one frozen file"),
        (_sessions, "records train sessions 2001-01-02"),
        (_identity, "but its file name says"),
    ],
)
async def test_policies_from_another_dataset_date_window_or_slot_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    tamper: Tamper,
    message: str,
) -> None:
    _shared_deps(monkeypatch)
    dataset, models, _ = _layout(tmp_path, tamper=tamper)

    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--confirm")))
    printed = capsys.readouterr().out
    assert rc == 1
    assert message in printed
    assert not job.ledger_path(models).exists()  # a refused run records no look


async def test_missing_policies_point_at_retrain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _shared_deps(monkeypatch)
    dataset, models, _ = _layout(tmp_path)
    job.policy_path(models, "SPY", 3, 2, untrained=False).unlink()

    rc = await job.run_eval(job.parse_args(_args(dataset, models, "--confirm")))
    assert rc == 1
    assert "1 per-fold policy export(s) are missing" in capsys.readouterr().out


def test_cli_refuses_fewer_than_five_seeds_and_a_grid_without_2_bps(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    base = ["--dataset", str(tmp_path / "d.json"), "--models", str(tmp_path)]
    assert job.main([*base, "--seeds", "0,1,2,3"]) == 1
    assert job.SEEDS_ERROR in capsys.readouterr().out
    assert job.main([*base, "--seeds", "0,1,2,3,4", "--cost-bps", "0,5"]) == 1
    assert "2 bps gate cost" in capsys.readouterr().out


@pytest.mark.rl
async def test_retrain_trains_and_exports_every_fold_policy_then_evaluates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pytest.importorskip("stable_baselines3")
    _shared_deps(monkeypatch)
    dataset = tmp_path / "spy.json"
    dataset.write_bytes(_dataset(195, adapter="AlpacaHistoricalData"))
    models = tmp_path / "models"
    argv = [
        "--dataset",
        str(dataset),
        "--models",
        str(models),
        "--seeds",
        "0,1,2,3,4",
        "--random-seeds",
        "3",
        "--train-bars",
        "120",
        "--val-bars",
        "10",
        "--test-bars",
        "10",
        "--step-bars",
        "10",
        "--retrain",
        "--timesteps",
        "128",
        "--episode-length",
        "60",
        "--confirm",
    ]
    rc = await job.run_eval(job.parse_args(argv))
    printed = capsys.readouterr().out
    assert rc == 0, printed
    for seed in _SEEDS:
        for untrained in (False, True):
            path = job.policy_path(models, "SPY", 0, seed, untrained=untrained)
            meta = json.loads(path.read_bytes())["meta"]
            assert meta["splits"]["train"]["bars"] == [0, 120]
            assert meta["parity"]["n_disagreements"] == 0
            assert meta["training"]["untrained"] is untrained
    assert "LOW_N" in printed  # one fold: descriptive only, no gate may pass
