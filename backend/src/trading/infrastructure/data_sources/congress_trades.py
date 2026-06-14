"""Congressional disclosed-trade adapter (FR-ING-2, FR-POL-6) — OPTIONAL, off by default.

The STOCK Act PTR disclosures are lagged up to 45 days, so this leg is corroboration only,
never a primary real-time trigger. Disabled unless DISCLOSED_TRADE_LEG_ENABLED is set; gated
behind a provider API key. Modeled on FMP's senate/house trading endpoints; the response
shape is normalized defensively so a provider swap is a parsing change, not a contract change.

No commercial use of the free tier (spec section 2.2 non-goals).
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.politicians import Politician
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_DISCLOSED_TRADE
from trading.infrastructure.data_sources.weighting import politician_weight_for

# Purchases read bullish, sales bearish; magnitude is modest since the leg is lagged.
_BUY_SENTIMENT = 0.5
_SELL_SENTIMENT = -0.5


class CongressTradesSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        endpoint: str,
        api_key: str,
        politicians: PoliticianRepositoryPort,
        sector_map: SectorMapPort,
        clock: ClockPort,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._api_key = api_key
        self._politicians = politicians
        self._sector_map = sector_map
        self._clock = clock

    @property
    def name(self) -> str:
        return "congress_trades"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        response = await self._client.get(self._endpoint, params={"apikey": self._api_key})
        response.raise_for_status()
        rows: list[dict[str, Any]] = response.json()

        signals: list[Signal] = []
        for row in rows:
            ticker = str(row.get("symbol", "")).upper()
            if not ticker:
                continue
            traded = _parse_date(str(row.get("transactionDate", ""))) or self._clock.now()
            if traded <= since:
                continue
            politician = self._resolve_politician(row)
            if politician is None:
                continue
            sentiment = _direction_sentiment(str(row.get("type", "")))
            if sentiment == 0.0:
                continue
            signals.append(
                Signal(
                    id=f"trade-{politician.id}-{ticker}-{int(traded.timestamp())}",
                    created_at=traded,
                    source_type=SignalSource.DISCLOSED_TRADE,
                    ticker=ticker,
                    politician_id=politician.id,
                    sentiment=sentiment,
                    politician_weight=politician_weight_for(politician, ticker, self._sector_map),
                    source_reliability=RELIABILITY_DISCLOSED_TRADE,
                    raw_payload=row,
                )
            )
        return signals

    def _resolve_politician(self, row: dict[str, Any]) -> Politician | None:
        name = " ".join(
            part for part in (row.get("firstName"), row.get("lastName")) if part
        ) or str(row.get("representative", ""))
        return self._politicians.find_in_text(name) if name else None


def _direction_sentiment(transaction_type: str) -> float:
    lowered = transaction_type.lower()
    if "purchase" in lowered or "buy" in lowered:
        return _BUY_SENTIMENT
    if "sale" in lowered or "sell" in lowered:
        return _SELL_SENTIMENT
    return 0.0


def _parse_date(value: str) -> datetime | None:
    try:
        parsed: date = date.fromisoformat(value[:10])
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
