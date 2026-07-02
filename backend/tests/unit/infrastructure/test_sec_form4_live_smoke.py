"""LIVE smoke test for the SEC Form 4 historical fetcher (Task 04a) -- hits REAL EDGAR.

Skipped by default in CI and locally: opt in with ``RUN_LIVE_SMOKE=1``. This is the ONLY test in
the suite permitted to make a real network call, and it is scoped to a narrow 1-day window
against a source that requires no auth/keys (just a descriptive User-Agent per SEC policy).

Per the implementer's orchestrator instructions, this test should be run AT MOST ONCE manually to
verify the adapter end-to-end against live EDGAR; it is fine to leave it unexecuted.
"""

import os
from datetime import UTC, datetime

import httpx
import pytest

from tests.conftest import FixedClock
from trading.infrastructure.data_sources.sec_form4 import SecForm4Source

pytestmark = [
    pytest.mark.live_smoke,
    pytest.mark.skipif(
        not os.environ.get("RUN_LIVE_SMOKE"),
        reason="RUN_LIVE_SMOKE not set (opt-in only; hits real EDGAR)",
    ),
]

_USER_AGENT = "trading-platform research steven.murley@gmail.com"


async def test_one_day_window_returns_signals(clock: FixedClock) -> None:
    # A single business day with known heavy Form 4 volume across S&P-500-sized issuers.
    start = datetime(2024, 2, 12, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)

    async with httpx.AsyncClient(timeout=30.0) as client:
        source = SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT)
        signals = await source.fetch_historical(start, end)

    assert len(signals) > 0
    assert all(s.ticker for s in signals)
    assert all(s.created_at.date() == start.date() for s in signals)
