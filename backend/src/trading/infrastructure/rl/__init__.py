"""RL artifact adapters (task-rl-09). Pure stdlib: no numpy / torch / SB3 / gymnasium may enter
this package's import closure (import-linter "Heavy ML deps never enter the app import graph" lists
``trading.infrastructure`` as a source; ``tests/unit/test_ml_deps_isolated.py`` proves it at
runtime)."""
