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
