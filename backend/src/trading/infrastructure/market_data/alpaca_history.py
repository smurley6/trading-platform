"""Historical daily-bars adapter over the Alpaca Market Data API v2.

Fetches daily OHLC bars and builds a domain `PriceSeries` (bar **close** as the price point
per timestamp), satisfying `HistoricalPricesPort` so a backtest can price off REAL history.
Built in `AlpacaMarketData`'s style: an INJECTED `httpx.AsyncClient`, the same
`APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` auth, and explicit status handling — fully unit-
testable with `httpx.MockTransport`, no network calls unless wired.

Endpoint:
  ``GET {base}/v2/stocks/bars?symbols=AAPL,MSFT&timeframe=1Day&start=<ISO>&end=<ISO>``

Response shape (multi-symbol form):
  ``{"bars": {"AAPL": [{"t","o","h","l","c","v",...}], ...}, "next_page_token": <str|null>}``

Pagination: Alpaca returns a `next_page_token`; we loop, passing it back as `page_token`,
until it is null, concatenating each page's bars per symbol. Missing symbols are simply
absent from the series.

Default OFF: wired only when `enable_historical_data` is set and data credentials resolve
(see `main.py`); otherwise the backtest uses `SyntheticHistoricalPrices` and makes no network
calls.

**End clamping (Task 09).** The subscription in use is entitled to delayed/IEX data, not
recent SIP: a request whose ``end`` reaches into "now" 403s with
``{"message":"subscription does not permit querying recent SIP data"}"`` — confirmed live (see
the 2026-07-06 MEMORY.md change-log entry for the exact probe). The labeler legitimately asks
for bars through a signal's horizon end, which can be in the future for a young signal, so the
adapter itself clamps the EFFECTIVE end before every request rather than pushing that decision
onto every caller. Callers still see the FULL requested window in the returned `PriceSeries`
(simply fewer/no bars past the clamp) — this is transparent shrinkage, not a rejected request.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.domain.prices import PriceSeries

logger = logging.getLogger(__name__)

# How far behind "now" the effective request end is clamped, absent a session-boundary clamp.
# Chosen conservatively above Alpaca's ~15-minute SIP delay so the developing (in-progress)
# daily bar for the current session — which itself aggregates the trailing window up to "now" —
# is never inside the requested range.
_RECENT_DATA_BUFFER = timedelta(minutes=16)

# Truncate an error response body to this many characters before it goes into the raised
# message/log — enough to see the actual Alpaca error payload without unbounded log spam.
_ERROR_BODY_TRUNCATE = 500


class AlpacaHistoricalDataError(Exception):
    """Raised on a non-2xx response. Carries the HTTP status and (truncated) response body so
    the real cause (e.g. a subscription/SIP-entitlement error) is never masked — see the
    2026-07-06 MEMORY.md change-log entry for the debugging cost of the previous behavior."""

    def __init__(self, status_code: int, body: str, *, message: str | None = None) -> None:
        self.status_code = status_code
        self.body = body[:_ERROR_BODY_TRUNCATE]
        super().__init__(message or f"historical data request failed ({status_code}): {self.body}")


class AlpacaHistoricalData:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        key_id: str,
        secret_key: str,
        base_url: str,
        timeframe: str = "1Day",
        clock: ClockPort | None = None,
        feed: str | None = None,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeframe = timeframe
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        # Escape hatch only (Task 09, `Settings.alpaca_data_feed`): unset by default, so NO
        # `feed` param is sent and Alpaca serves its default entitled feed. Do not default this
        # to "iex" — see the config.py comment on `alpaca_data_feed`.
        self._feed = feed
        # Optional injected clock so the end-clamp is deterministically testable (FixedClock in
        # unit tests). The composition root has a `SystemClock` in scope at construction time
        # (see `main.py`), so it is threaded through here rather than re-plumbed; if ever
        # constructed without one (e.g. an older call site), `datetime.now(UTC)` is used
        # directly at call time — see `_now()`.
        self._clock = clock

    def _now(self) -> datetime:
        if self._clock is not None:
            return self._clock.now()
        # No clock injected: fall back to wall-clock time directly. Documented deliberate
        # decision (Task 09) — not re-plumbing the composition root just for this adapter; every
        # current call site (`main.py`) DOES pass a clock, so this branch is a defensive default
        # only (e.g. a test or future call site that constructs the adapter bare).
        return datetime.now(UTC)

    def _clamp_end(self, requested_end: datetime) -> datetime:
        """Clamp the effective request end to ``min(requested_end, now - _RECENT_DATA_BUFFER)``.

        The labeler may legitimately ask for bars through a horizon end that is in the future
        for a young signal; Alpaca's recent-SIP entitlement check 403s once the window reaches
        into "now" (live-verified — see the 2026-07-06 MEMORY.md entry). Clamping here means
        every caller of this adapter gets a request Alpaca will actually serve, without needing
        its own clamping logic.
        """
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        boundary = now.astimezone(UTC) - _RECENT_DATA_BUFFER
        requested = requested_end if requested_end.tzinfo else requested_end.replace(tzinfo=UTC)
        return min(requested.astimezone(UTC), boundary)

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        wanted = [s.strip().upper() for s in symbols if s.strip()]
        if not wanted:
            return PriceSeries(series={})

        effective_end = self._clamp_end(end)

        # Accumulate bars per symbol across pages.
        bars_by_symbol: dict[str, list[tuple[datetime, Decimal]]] = {}
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "symbols": ",".join(wanted),
                "timeframe": self._timeframe,
                "start": _iso(start),
                "end": _iso(effective_end),
            }
            if self._feed:
                params["feed"] = self._feed
            if page_token:
                params["page_token"] = page_token
            payload = await self._fetch(params)
            raw_bars: dict[str, Any] = payload.get("bars") or {}
            for symbol, entries in raw_bars.items():
                points = bars_by_symbol.setdefault(symbol, [])
                for entry in entries or []:
                    points.append((_parse_time(entry.get("t")), _decimal(entry.get("c"))))
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        series = {
            symbol: tuple(sorted(points, key=lambda point: point[0]))
            for symbol, points in bars_by_symbol.items()
            if points
        }
        return PriceSeries(series=series)

    async def _fetch(self, params: dict[str, str]) -> dict[str, Any]:
        response = await self._client.get(
            f"{self._base_url}/v2/stocks/bars",
            headers=self._headers,
            params=params,
        )
        if response.status_code == 401:
            # 401 is a genuine credential problem (missing/invalid key pair) — "check
            # credentials" is accurate guidance here and is kept.
            body = response.text[:_ERROR_BODY_TRUNCATE]
            logger.error("historical data auth failed (401): check credentials; body=%s", body)
            raise AlpacaHistoricalDataError(
                response.status_code,
                body,
                message=f"historical data auth failed (401): check credentials; body={body}",
            )
        if response.status_code >= 400:
            # 403 (and any other 4xx/5xx) is NOT assumed to be a credentials issue — it may be a
            # subscription/entitlement error (e.g. recent-SIP), rate limit, or bad request. The
            # real response body is logged and propagated verbatim (truncated) so the actual
            # cause is never masked.
            body = response.text[:_ERROR_BODY_TRUNCATE]
            logger.error("historical data request failed (%s): %s", response.status_code, body)
            raise AlpacaHistoricalDataError(response.status_code, body)
        payload: dict[str, Any] = response.json()
        return payload


def _iso(value: datetime) -> str:
    when = value if value.tzinfo else value.replace(tzinfo=UTC)
    return when.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    # Alpaca timestamps are RFC-3339 with nanoseconds; trim to microseconds for fromisoformat.
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        frac = tail
        offset = ""
        for marker in ("+", "-"):
            if marker in tail:
                frac, _, rest = tail.partition(marker)
                offset = marker + rest
                break
        text = f"{head}.{frac[:6]}{offset}"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
