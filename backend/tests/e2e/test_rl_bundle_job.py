"""RL demo bundle job end-to-end on a SYNTHETIC layout (task-rl-12).

``python -m trading.jobs.rl_bundle`` through its CLI entrypoint: the real dataset reader, the real
``load_fold_policy`` identity checks, real env replays (``trading.rl.scoring``, gymnasium + numpy)
cross-checked against the rl-10 report's rows. A pure read/transform: the inputs (dataset, report,
policies, look ledger) are byte-identical afterwards and the composition root is never built (no
event can be appended).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.unit.rl.bundle_fixture import Layout, build_layout, tree_digest
from trading.infrastructure.rl.policy_store import load_policy, write_policy
from trading.jobs import rl_bundle as job


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Layout:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the bundle job must never build the composition root")

    monkeypatch.setattr("trading.main.build_dependencies", forbidden)
    monkeypatch.setattr("trading.jobs.rl_eval.build_dependencies", forbidden)
    return build_layout(tmp_path / "layout")


def test_cli_writes_the_bundle_and_touches_no_input(
    layout: Layout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = tree_digest(layout.root)
    out = tmp_path / "site" / "demo.json"

    assert job.main(layout.argv(out)) == 0
    printed = capsys.readouterr().out

    data = out.read_bytes()
    assert f"sha256  {hashlib.sha256(data).hexdigest()}" in printed
    assert f"{len(data)} bytes raw" in printed
    assert "ppo: beats buy-and-hold: " in printed
    assert f"{layout.n_folds} per-fold policies shipped" in printed
    assert tree_digest(layout.root) == before  # dataset, report, policies, ledger: untouched
    assert not out.with_name("demo.json.tmp").exists()
    bundle = json.loads(data)
    assert (
        bundle["meta"]["eval_report_sha256"]
        == hashlib.sha256(layout.report.read_bytes()).hexdigest()
    )

    # Determinism through the CLI: a second run writes the same bytes.
    second = tmp_path / "second.json"
    assert job.main(layout.argv(second)) == 0
    assert second.read_bytes() == data


def test_generated_at_defaults_to_the_report_as_of_and_can_be_overridden(
    layout: Layout, tmp_path: Path
) -> None:
    out = tmp_path / "demo.json"
    assert job.main(layout.argv(out)) == 0
    meta = json.loads(out.read_bytes())["meta"]
    assert meta["generated_at"] == json.loads(layout.report.read_bytes())["as_of"]
    assert "as_of" in meta["generated_at_source"]
    assert job.main(layout.argv(out, "--generated-at", "2026-10-23T00:00:00Z")) == 0
    meta = json.loads(out.read_bytes())["meta"]
    assert meta["generated_at"] == "2026-10-23T00:00:00Z"
    assert "--generated-at" in meta["generated_at_source"]


def _rewrite_policy(layout: Layout, fold: int, mutate: Any) -> None:
    path = layout.walk_forward / f"fold{fold:02d}" / "spy_ppo_seed0.policy.json"
    meta = json.loads(path.read_bytes())["meta"]
    mutate(meta)
    write_policy(path, load_policy(path), meta=meta)


def _train_window(meta: dict[str, Any]) -> None:
    meta["splits"]["train"]["bars"] = [0, 100]  # claims fold 0's window


def _innocent(meta: dict[str, Any]) -> None:
    meta["note"] = "re-exported"  # same weights + identity, different bytes than evaluated


def _identity(meta: dict[str, Any]) -> None:
    meta["walk_forward"]["seed"] = 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_train_window, "not fold 3's train window"),
        (_identity, "but its file name says"),
        (_innocent, "not the evaluated bytes"),
    ],
)
def test_a_policy_that_is_not_the_evaluated_one_is_refused(
    layout: Layout,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutate: Any,
    message: str,
) -> None:
    _rewrite_policy(layout, 3, mutate)
    out = tmp_path / "demo.json"
    assert job.main(layout.argv(out)) == 1
    assert message in capsys.readouterr().out
    assert not out.exists()


def test_a_report_row_the_replay_does_not_reproduce_is_refused(
    layout: Layout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = json.loads(layout.report.read_bytes())
    row = next(
        r
        for r in report["results"]
        if r["policy_id"] == "sma_cross_20_50" and r["fold_index"] == 2 and r["cost_bps"] == 5.0
    )
    row["final_equity"] *= 1.001
    layout.report.write_text(json.dumps(report), encoding="utf-8")
    out = tmp_path / "demo.json"
    assert job.main(layout.argv(out)) == 1
    assert "the inputs are not the evaluated ones" in capsys.readouterr().out
    assert not out.exists()


def test_a_report_about_another_dataset_is_refused(
    layout: Layout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = json.loads(layout.report.read_bytes())
    report["dataset_sha256"] = "f" * 64
    layout.report.write_text(json.dumps(report), encoding="utf-8")
    assert job.main(layout.argv(tmp_path / "demo.json")) == 1
    assert "refusing" in capsys.readouterr().out


def test_folds_that_are_not_make_folds_of_the_dataset_are_refused(
    layout: Layout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = json.loads(layout.report.read_bytes())
    report["folds"][4]["test"] = [report["folds"][4]["test"][0] - 5, report["folds"][4]["test"][1]]
    layout.report.write_text(json.dumps(report), encoding="utf-8")
    assert job.main(layout.argv(tmp_path / "demo.json")) == 1
    assert "not make_folds of this dataset" in capsys.readouterr().out


def test_an_unevaluated_exhibit_seed_and_unsafe_out_paths_are_refused(
    layout: Layout, capsys: pytest.CaptureFixture[str]
) -> None:
    assert job.main(layout.argv(layout.root / "demo.json", "--exhibit-seed", "9")) == 1
    assert "is not an evaluated 'ppo' seed" in capsys.readouterr().out
    assert job.main(layout.argv(layout.dataset)) == 1
    assert job.main(layout.argv(layout.walk_forward / "demo.json")) == 1
    assert "--out may not overwrite an input" in capsys.readouterr().out


def test_cli_usage_errors_exit_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert job.main(["--dataset", str(tmp_path / "x.json")]) == 1
    assert (
        job.main(
            [
                "--dataset",
                str(tmp_path / "missing.json"),
                "--eval",
                str(tmp_path / "r.json"),
                "--walk-forward",
                str(tmp_path / "walk_forward"),
                "--out",
                str(tmp_path / "o.json"),
            ]
        )
        == 1
    )
    assert "cannot read --dataset" in capsys.readouterr().out
