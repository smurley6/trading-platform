"""LIVE smoke test for the Alpaca historical-bars adapter (Task 09) -- hits REAL Alpaca data API.

Skipped by default in CI and locally: opt in with ``RUN_LIVE_SMOKE=1``. Reproduces the EXACT
app-shaped request that originally 403'd (task-09 spec probe B): start well in the past,
REQUESTED end in the future, NO `feed` param -- with the adapter's end-clamp applied. Asserts
HTTP 200 with bars, proving the clamp resolves the "subscription does not permit querying
recent SIP data" 403 without defaulting to a degraded (IEX) feed.

Uses `Settings` to resolve the real `alpaca_data_key_id`/`alpaca_data_secret_key` from the
checked-in-but-gitignored `backend/.env`, exactly as the composition root does -- this is a
read-only market-data GET, sanctioned per the orchestrator's live-test carve-out.
"""

import os
from datetime import UTC, datetime

import httpx
import pytest

from trading.config import Settings
from trading.infrastructure.market_data.alpaca_history import AlpacaHistoricalData

pytestmark = [
    pytest.mark.live_smoke,
    pytest.mark.skipif(
        not os.environ.get("RUN_LIVE_SMOKE"),
        reason="RUN_LIVE_SMOKE not set (opt-in only; hits real Alpaca data API)",
    ),
]


async def test_app_shaped_request_with_clamp_returns_200() -> None:
    settings = Settings()
    key_id = settings.data_key_id()
    secret_key = settings.data_secret_key()
    assert key_id and secret_key, "ALPACA_DATA_KEY_ID/ALPACA_DATA_SECRET_KEY must resolve"

    async with httpx.AsyncClient(timeout=30.0) as client:
        adapter = AlpacaHistoricalData(
            client=client,
            key_id=key_id,
            secret_key=secret_key,
            base_url=settings.alpaca_data_base_url,
            feed=settings.alpaca_data_feed,  # unset by default -- no feed param sent
        )
        # Reproduces task-09 spec probe B exactly: start well in the past, requested end in the
        # FUTURE relative to real wall-clock "now" -- no clock injected, so the adapter falls
        # back to datetime.now(UTC) for the clamp (the documented no-clock decision).
        start = datetime(2024, 6, 24, 20, 0, 5, tzinfo=UTC)
        requested_end = datetime.now(UTC).replace(year=datetime.now(UTC).year + 1)
        series = await adapter.price_series(["AAPL"], start, requested_end)

    assert series.symbols() == ("AAPL",)
    assert len(series.series["AAPL"]) > 0
