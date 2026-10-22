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
