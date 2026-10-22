"""The heavy ML stack never enters the app import graph (task-rl-01).

import-linter enforces this statically ("Heavy ML deps never enter the app import graph"); these
tests prove it at RUNTIME, including transitive third-party imports the linter cannot see. Each
check runs in a FRESH interpreter: ``sys.modules`` is process-global, so an in-process check would
pass or fail depending on which other tests (e.g. future gymnasium env tests) ran first.
"""

import json
import subprocess
import sys

_HEAVY_ML_MODULES = frozenset({"torch", "stable_baselines3", "gymnasium"})


def _loaded_modules_after(statement: str) -> tuple[set[str], set[str]]:
    """Run ``statement`` in a fresh interpreter; return (modules before, modules after)."""
    probe = (
        "import json, sys\n"
        "before = sorted(sys.modules)\n"
        f"{statement}\n"
        "print(json.dumps({'before': before, 'after': sorted(sys.modules)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return set(payload["before"]), set(payload["after"])


def test_app_import_graph_has_no_ml_modules() -> None:
    _, loaded = _loaded_modules_after("import trading.main")

    assert "trading.main" in loaded
    assert _HEAVY_ML_MODULES & loaded == set()
    assert {m for m in loaded if m == "trading.rl" or m.startswith("trading.rl.")} == set()


def test_trading_rl_package_is_inert() -> None:
    before, after = _loaded_modules_after("import trading.rl")

    assert "trading.rl" in after
    assert _HEAVY_ML_MODULES & after == set()
    # "imports nothing": the only modules the import adds are the package itself and its parent.
    assert after - before <= {"trading", "trading.rl"}


def test_rl_dataset_job_import_graph_has_no_ml_modules() -> None:
    # task-rl-02: the dataset freeze is listed as a forbidden-contract source despite its rl_
    # prefix (pure fetch/serialize); prove it at runtime too.
    _, loaded = _loaded_modules_after("import trading.jobs.rl_dataset")

    assert "trading.jobs.rl_dataset" in loaded
    assert _HEAVY_ML_MODULES & loaded == set()
    assert {m for m in loaded if m == "trading.rl" or m.startswith("trading.rl.")} == set()


def test_rl_pure_half_loads_no_ml_or_numpy() -> None:
    # task-rl-03: bars/costs/rewards are the PURE half of trading.rl (program doc §3.1) — they must
    # import with zero optional deps, so not even numpy may load.
    _, loaded = _loaded_modules_after(
        "import trading.rl.bars, trading.rl.costs, trading.rl.rewards"
    )

    assert {"trading.rl.bars", "trading.rl.costs", "trading.rl.rewards"} <= loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()


def test_rl_env_needs_only_gymnasium_and_numpy() -> None:
    # task-rl-05: the env needs only gymnasium + numpy (both in [dev]), so the env and leakage
    # tests run in the default pytest without the [rl] extra. It must never pull torch / SB3.
    _, loaded = _loaded_modules_after("import trading.rl.env")

    assert {"trading.rl.env", "gymnasium", "numpy"} <= loaded
    assert {"torch", "stable_baselines3"} & loaded == set()


def test_rl_baselines_and_metrics_are_pure() -> None:
    # task-rl-07: the baseline policies and the evaluation metrics are PURE half too (a policy
    # that ignores the observation needs no numpy; reports and the demo bundle need no gymnasium).
    _, loaded = _loaded_modules_after("import trading.rl.baselines, trading.rl.metrics")

    assert {"trading.rl.baselines", "trading.rl.metrics"} <= loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()


def test_rl_scoring_needs_only_gymnasium_and_numpy() -> None:
    # task-rl-07: scoring drives the env, so it loads gymnasium + numpy — never torch / SB3.
    _, loaded = _loaded_modules_after("import trading.rl.scoring")

    assert {"trading.rl.scoring", "trading.rl.env", "gymnasium", "numpy"} <= loaded
    assert {"torch", "stable_baselines3"} & loaded == set()


def test_rl_train_job_import_graph_has_no_ml_modules() -> None:
    # task-rl-08: the training job imports trading.rl.train only INSIDE run_training (after its
    # guards), so loading the module pulls in no ML at all and a missing [rl] extra is reported
    # with a clear message instead of an ImportError at import time.
    _, loaded = _loaded_modules_after("import trading.jobs.rl_train")

    assert "trading.jobs.rl_train" in loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()
    assert "trading.rl.train" not in loaded


def test_rl_train_imports_sb3_and_torch_lazily() -> None:
    # task-rl-08: train.py is the SB3 consumer, but imports stable-baselines3 / torch only inside
    # train_ppo / load_model, so its manifest + bridge seams are testable without the [rl] extra.
    _, loaded = _loaded_modules_after("import trading.rl.train")

    assert {"trading.rl.train", "gymnasium", "numpy"} <= loaded
    assert {"torch", "stable_baselines3"} & loaded == set()


def test_strategy_and_policy_store_import_closure_is_torch_free() -> None:
    # task-rl-09: the DEPLOYABLE inference path — the stdlib MLP and its JSON store — must load
    # with zero optional deps: no torch / SB3 / gymnasium, and not even numpy.
    _, loaded = _loaded_modules_after(
        "import trading.infrastructure.rl.policy_store, trading.domain.rl_policy"
    )

    assert {"trading.infrastructure.rl.policy_store", "trading.domain.rl_policy"} <= loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()
    assert {m for m in loaded if m == "trading.rl" or m.startswith("trading.rl.")} == set()


def test_rl_export_job_import_graph_has_no_ml_modules() -> None:
    # task-rl-09: like rl_train, the export job imports trading.rl.export only INSIDE run_export
    # (after its guards), so loading the module pulls in no ML and no numpy.
    _, loaded = _loaded_modules_after("import trading.jobs.rl_export")

    assert "trading.jobs.rl_export" in loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()
    assert "trading.rl.export" not in loaded


def test_rl_export_imports_sb3_and_torch_lazily() -> None:
    # task-rl-09: export.py is an SB3 consumer, but imports stable-baselines3 / torch only inside
    # the functions that need them (like train.py).
    _, loaded = _loaded_modules_after("import trading.rl.export")

    assert {"trading.rl.export", "gymnasium", "numpy"} <= loaded
    assert {"torch", "stable_baselines3"} & loaded == set()


def test_rl_eval_job_import_graph_has_no_ml_modules() -> None:
    # task-rl-10: the eval job imports the env replay (trading.rl.scoring: gymnasium + numpy) and
    # the SB3 retrain path only INSIDE its functions, so loading the module pulls in no ML.
    _, loaded = _loaded_modules_after("import trading.jobs.rl_eval")

    assert "trading.jobs.rl_eval" in loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()
    assert {"trading.rl.scoring", "trading.rl.export", "trading.rl.train"} & loaded == set()


def test_rl_evaluation_use_case_is_stdlib_only() -> None:
    # task-rl-10: the report/gates use case is pure application code (no numpy, no trading.rl).
    _, loaded = _loaded_modules_after("import trading.application.rl_evaluation")

    assert "trading.application.rl_evaluation" in loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()
    assert {m for m in loaded if m == "trading.rl" or m.startswith("trading.rl.")} == set()


def test_rl_bundle_is_stdlib_only() -> None:
    # task-rl-12: the bundle assembly (stdlib replay, equity accounting, stitching, formats) is the
    # PURE half of trading.rl: no numpy, gymnasium or torch.
    _, loaded = _loaded_modules_after("import trading.rl.bundle")

    assert "trading.rl.bundle" in loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()


def test_rl_bundle_job_import_graph_has_no_ml_modules() -> None:
    # task-rl-12: the bundle job imports the env replay (trading.rl.scoring: gymnasium + numpy)
    # only INSIDE build(), so loading the module pulls in no ML and no numpy.
    _, loaded = _loaded_modules_after("import trading.jobs.rl_bundle")

    assert {"trading.jobs.rl_bundle", "trading.rl.bundle"} <= loaded
    assert (_HEAVY_ML_MODULES | {"numpy"}) & loaded == set()
    assert {"trading.rl.scoring", "trading.rl.env", "trading.rl.export"} & loaded == set()
