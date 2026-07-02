"""Shared polite-throttle helper for historical (backfill) data-source adapters.

Extracted from ``sec_form4.py`` (Task 04a) so the House disclosures historical fetcher (Task
04b) reuses the exact same sleep-based rate limiter instead of duplicating it — both walk a
government archive sequentially and must stay courteous (no parallel fan-out).
"""

import asyncio
import time
from typing import Any


class RateLimiter:
    """Sleeps as needed to keep requests at or below a ceiling of ``1 / min_interval`` req/s.

    Uses ``time.monotonic`` (never affected by wall-clock changes) and an injectable sleep
    function so tests can run the throttle logic without actually waiting (patch ``sleep`` with
    a no-op / recording stub, or inject a fake ``monotonic``).
    """

    def __init__(
        self,
        *,
        min_interval: float,
        monotonic: Any = time.monotonic,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self._min_interval = min_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request: float | None = None

    async def wait(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            elapsed = now - self._last_request
            remaining = self._min_interval - elapsed
            if remaining > 0:
                await self._sleep(remaining)
                now = self._monotonic()
        self._last_request = now
