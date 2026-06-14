"""Live quote adapter over the Alpaca Market Data API v2 (base `https://data.alpaca.markets`).

Implements both `MarketDataPort.get_quote` (single symbol) and `QuoteFeedPort.get_quotes`
(batch) so the same adapter serves the paper broker seam and the live tracking feed. Built
in the broker adapter's style: an INJECTED `httpx.AsyncClient`, the same
`APCA-API-KEY-ID`/`APCA-API-SECRET-KEY` auth headers, and explicit status handling — so it
is fully unit-testable with `httpx.MockTransport` and makes no network calls unless wired.

Quote mapping (Alpaca `latest quotes` shape, fields `bp`/`ap`):
  - bid  <- `bp`
  - ask  <- `ap`
  - last <- the National Best Bid/Offer midpoint `(bp + ap) / 2`, quantized to 4dp. The
    `latest quotes` endpoint carries no trade price, so the mid is the most faithful single
    "last" without a second `trades/latest` round-trip. When only one side is present we fall
    back to whichever side we have.

Default OFF: this adapter is wired only when `enable_live_quotes` is set and credentials are
present (see `main.py`); otherwise the app injects `DisabledQuoteFeed`.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from trading.application.ports.market_data import Quote


class AlpacaMarketDataError(Exception):
    pass


class AlpacaMarketData:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        key_id: str,
        secret_key: str,
        base_url: str,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }

    @property
    def enabled(self) -> bool:
        return True

    async def get_quote(self, symbol: str) -> Quote:
        quotes = await self.get_quotes([symbol])
        if not quotes:
            raise AlpacaMarketDataError(f"no quote returned for symbol '{symbol}'")
        return quotes[0]

    async def get_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        wanted = [s.strip().upper() for s in symbols if s.strip()]
        if not wanted:
            return []
        response = await self._client.get(
            f"{self._base_url}/v2/stocks/quotes/latest",
            headers=self._headers,
            params={"symbols": ",".join(wanted)},
        )
        if response.status_code in (401, 403):
            raise AlpacaMarketDataError(
                f"market data auth failed ({response.status_code}): check credentials"
            )
        if response.status_code >= 400:
            raise AlpacaMarketDataError(
                f"market data request failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
        raw: dict[str, Any] = payload.get("quotes", {}) or {}
        quotes: list[Quote] = []
        for symbol in wanted:
            entry = raw.get(symbol)
            if not entry:
                continue  # symbol unknown / no quote available — skip gracefully
            quotes.append(_to_quote(symbol, entry))
        return quotes


def _to_quote(symbol: str, entry: dict[str, Any]) -> Quote:
    bid = _decimal(entry.get("bp"))
    ask = _decimal(entry.get("ap"))
    last = _mid(bid, ask)
    return Quote(symbol=symbol, bid=bid, ask=ask, last=last, as_of=_parse_time(entry.get("t")))


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


def _mid(bid: Decimal, ask: Decimal) -> Decimal:
    if bid > 0 and ask > 0:
        return ((bid + ask) / 2).quantize(Decimal("0.0001"))
    return ask if ask > 0 else bid


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
