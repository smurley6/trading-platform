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
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from trading.domain.prices import PriceSeries


class AlpacaHistoricalDataError(Exception):
    pass


class AlpacaHistoricalData:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        key_id: str,
        secret_key: str,
        base_url: str,
        timeframe: str = "1Day",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeframe = timeframe
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        wanted = [s.strip().upper() for s in symbols if s.strip()]
        if not wanted:
            return PriceSeries(series={})

        # Accumulate bars per symbol across pages.
        bars_by_symbol: dict[str, list[tuple[datetime, Decimal]]] = {}
        page_token: str | None = None
        while True:
            params: dict[str, str] = {
                "symbols": ",".join(wanted),
                "timeframe": self._timeframe,
                "start": _iso(start),
                "end": _iso(end),
            }
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
        if response.status_code in (401, 403):
            raise AlpacaHistoricalDataError(
                f"historical data auth failed ({response.status_code}): check credentials"
            )
        if response.status_code >= 400:
            raise AlpacaHistoricalDataError(
                f"historical data request failed ({response.status_code}): {response.text}"
            )
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
