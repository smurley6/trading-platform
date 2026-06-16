"""Regression: the synchronous composition root must load the congress roster from ANY context.

The cron job builds dependencies from *inside* ``asyncio.run(run_cycle())``; the original code
loaded the roster with a bare ``asyncio.run(...)``, which raises ``RuntimeError: asyncio.run()
cannot be called from a running event loop`` when a loop is already running. CI never caught it
because no test built deps with the roster enabled while a loop was running. These tests pin both
contexts: a running loop (the cron) and no loop (the FastAPI entrypoint / sync tests).
"""

import asyncio

import pytest

from trading import main
from trading.config import Settings
from trading.infrastructure.politicians.congress_legislators import CongressLegislatorsRepository


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        max_trade_pct_of_sleeve=10,
        max_position_concentration_pct=25,
        max_drawdown_pct=20,
    )


@pytest.fixture
def _fake_roster(monkeypatch: pytest.MonkeyPatch) -> CongressLegislatorsRepository:
    """Replace the networked fetch with a canned in-memory roster (empty is fine structurally)."""
    repo = CongressLegislatorsRepository(())

    async def _fake(settings: Settings) -> CongressLegislatorsRepository:
        return repo

    monkeypatch.setattr(main, "_load_congress_roster", _fake)
    return repo


async def test_load_roster_blocking_works_inside_a_running_loop(
    _fake_roster: CongressLegislatorsRepository,
) -> None:
    # An async test runs inside a live event loop — exactly the cron's situation. This must NOT
    # raise "asyncio.run() cannot be called from a running event loop".
    asyncio.get_running_loop()  # assert the premise: a loop really is running here
    assert main._load_roster_blocking(_settings()) is _fake_roster


def test_load_roster_blocking_works_without_a_loop(
    _fake_roster: CongressLegislatorsRepository,
) -> None:
    # The FastAPI entrypoint / sync tests build deps with no loop running — the fast path.
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()  # premise: no loop here
    assert main._load_roster_blocking(_settings()) is _fake_roster
