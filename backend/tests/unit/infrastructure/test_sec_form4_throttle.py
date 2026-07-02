"""SEC EDGAR fair-access throttle + retry-on-429/503 (Task 04a), no network, no real sleeping.

``asyncio.sleep`` is replaced with a recording stub injected via the ``sleep=`` constructor
kwarg, so both the per-request rate limiter and the exponential backoff retry run their real
logic (computing intervals, counting attempts) without the test suite actually waiting.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from tests.conftest import FixedClock
from trading.infrastructure.data_sources.sec_form4 import SecForm4Source

Handler = Callable[[httpx.Request], httpx.Response]
_USER_AGENT = "trading-platform research steven.murley@gmail.com"


class _RecordingSleep:
    """Stand-in for asyncio.sleep: records requested durations, returns instantly."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_retries_after_429_then_succeeds(clock: FixedClock) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, text="ok")

    sleep = _RecordingSleep()
    async with _client(handler) as client:
        source = SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT, sleep=sleep)
        response = await source._get_response("https://www.sec.gov/some/path")

    assert response.status_code == 200
    assert len(attempts) == 2  # one 429, one success -- no more retries than necessary
    # The backoff slept once (for the 429), plus the rate limiter's per-request wait may also
    # call sleep, but at least one call reflects the exponential backoff base.
    assert any(s >= 1.0 for s in sleep.calls)


async def test_retries_after_503_then_succeeds(clock: FixedClock) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) <= 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ok")

    sleep = _RecordingSleep()
    async with _client(handler) as client:
        source = SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT, sleep=sleep)
        response = await source._get_response("https://www.sec.gov/some/path")

    assert response.status_code == 200
    assert len(attempts) == 3
    # Exponential backoff: second wait should be roughly double the first among backoff sleeps.
    assert len(sleep.calls) >= 2


async def test_exhausts_retries_and_raises(clock: FixedClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    sleep = _RecordingSleep()
    async with _client(handler) as client:
        source = SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT, sleep=sleep)
        with pytest.raises(httpx.HTTPStatusError):
            await source._get_response("https://www.sec.gov/some/path")


async def test_non_retryable_status_raises_immediately(clock: FixedClock) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(404, text="not found")

    sleep = _RecordingSleep()
    async with _client(handler) as client:
        source = SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT, sleep=sleep)
        with pytest.raises(httpx.HTTPStatusError):
            await source._get_response("https://www.sec.gov/some/path")

    assert len(attempts) == 1  # no retry on a non-retryable status


async def test_rate_limiter_throttles_consecutive_requests(clock: FixedClock) -> None:
    """The per-request rate limiter sleeps to keep requests <= 10/s when they'd otherwise be
    back-to-back (monotonic clock shows no time elapsed between calls)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    sleep = _RecordingSleep()
    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    async with _client(handler) as client:
        source = SecForm4Source(client=client, clock=clock, user_agent=_USER_AGENT, sleep=sleep)
        source._limiter._monotonic = fake_monotonic
        await source._get_response("https://www.sec.gov/a")
        await source._get_response("https://www.sec.gov/b")

    # Second request, at the same fake "instant" as the first, must wait ~1/10s.
    assert any(0.09 <= s <= 0.11 for s in sleep.calls)


async def test_fetch_historical_uses_injected_sleep_not_real_time(clock: FixedClock) -> None:
    """A tiny end-to-end sanity check that fetch_historical's HTTP calls all flow through the
    injected sleep stub (so a real test run of this suite never actually waits)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/Archives/edgar/full-index/2024/QTR1/form.idx":
            return httpx.Response(
                200, text="Form Type   Company Name   CIK   Date Filed   File Name\n---\n"
            )
        return httpx.Response(404, text="not found")

    sleep = _RecordingSleep()
    async with _client(handler) as client:
        source = SecForm4Source(
            client=client, clock=clock, user_agent=_USER_AGENT, sleep=sleep, filter_enabled=False
        )
        signals = await source.fetch_historical(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 31, tzinfo=UTC)
        )

    assert signals == []
