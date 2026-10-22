"""Offline reinforcement-learning research package -- the HEAVY half of the RL program.

This is the ONLY place in ``trading`` where gymnasium / stable-baselines3 / torch / numpy may be
imported (alongside the offline ``trading.jobs.rl_*`` entrypoints that drive it). Nothing the web
service or the cron cycle imports may reach this package: the import-linter contract "Heavy ML
deps never enter the app import graph" forbids ``trading.rl``, ``torch``, ``stable_baselines3``,
``gymnasium`` and ``numpy`` from ``trading.{domain,application,strategies,interface,
infrastructure,main}``. See docs/RL-SLEEVE-PROGRAM.md section 3.

The heavy dependencies live in the optional ``[rl]`` extra (``pip install -e ".[rl]"``), which CI
and Render never install. This ``__init__`` deliberately imports nothing, so ``import trading.rl``
is free and leaves the interpreter exactly as it was.
"""
