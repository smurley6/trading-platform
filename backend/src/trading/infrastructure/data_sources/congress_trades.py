"""Congressional disclosed-trade adapter (FR-ING-2, FR-POL-6) — OPTIONAL, off by default.

The STOCK Act PTR disclosures are lagged up to 45 days, so this leg is corroboration only,
never a primary real-time trigger. Disabled unless DISCLOSED_TRADE_LEG_ENABLED is set; gated
behind a provider API key. Modeled on FMP's senate/house trading endpoints; the response
shape is normalized defensively so a provider swap is a parsing change, not a contract change.

No commercial use of the free tier (spec section 2.2 non-goals).
"""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.politicians import Politician
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_DISCLOSED_TRADE
from trading.infrastructure.data_sources.weighting import politician_weight_for

logger = logging.getLogger(__name__)

# Purchases read bullish, sales bearish; magnitude is modest since the leg is lagged.
_BUY_SENTIMENT = 0.5
_SELL_SENTIMENT = -0.5

# FMP disclosure/filing-date field names, most-preferred first. The endpoint stamps the day the
# trade became PUBLIC under one of these (``dateRecieved`` is FMP's own long-standing misspelling
# on some senate/house endpoints). This is the PIT-correct timestamp — never the transaction date.
_DISCLOSURE_DATE_FIELDS = ("disclosureDate", "dateRecieved", "dateReceived", "filingDate")


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
        dropped_stale = 0  # disclosed on/before the cursor (already seen) — the F13 concern
        dropped_other = 0  # no ticker / no directional read
        for row in rows:
            ticker = str(row.get("symbol", "")).upper()
            if not ticker:
                dropped_other += 1
                continue
            # created_at = the DISCLOSURE date (when the trade became public), NOT the transaction
            # date (review F13): a STOCK Act filing can lag the trade up to 45 days, so windowing/
            # ageing by the trade date silently dropped late-disclosed trades and mis-aged the rest.
            # This matches the House/Senate PIT rule. The transaction date stays in raw_payload.
            traded = _parse_date(str(row.get("transactionDate", "")))
            # created_at/windowing must NEVER be the transaction date (PIT lookahead — review F13),
            # so when the row carries no disclosure field we fall back to ingestion time
            # (clock.now()), never to `traded`. This mirrors the Senate adapter's
            # `filing.received or clock.now()` chain. `traded` is used only for the id key below.
            disclosed = _disclosure_date(row) or self._clock.now()
            if disclosed <= since:
                dropped_stale += 1
                continue
            politician = self._resolve_politician(row)
            if politician is None:
                dropped_other += 1
                continue
            sentiment = _direction_sentiment(str(row.get("type", "")))
            if sentiment == 0.0:
                dropped_other += 1
                continue
            # Keep the id keyed on the transaction instant so two trades a filer discloses together
            # (same disclosure date) stay distinct; created_at carries the PIT disclosure time.
            uid_ts = int((traded or disclosed).timestamp())
            signals.append(
                Signal(
                    id=f"trade-{politician.id}-{ticker}-{uid_ts}",
                    created_at=disclosed,
                    source_type=SignalSource.DISCLOSED_TRADE,
                    ticker=ticker,
                    politician_id=politician.id,
                    sentiment=sentiment,
                    politician_weight=politician_weight_for(politician, ticker, self._sector_map),
                    source_reliability=RELIABILITY_DISCLOSED_TRADE,
                    raw_payload=row,
                    # REAL only if the matched politician is real (fictional roster -> SYNTHETIC).
                    provenance=(
                        Provenance.REAL
                        if politician.provenance is Provenance.REAL
                        else Provenance.SYNTHETIC
                    ),
                )
            )
        if dropped_stale or dropped_other:
            # A dropped-row counter so late/stale disclosures never vanish silently (review F13).
            logger.info(
                "congress_trades: %d signal(s), %d row(s) dropped "
                "(%d disclosed on/before cursor, %d no ticker/politician/direction)",
                len(signals),
                dropped_stale + dropped_other,
                dropped_stale,
                dropped_other,
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


def _disclosure_date(row: dict[str, Any]) -> datetime | None:
    """The filing/disclosure date from whichever FMP field the endpoint used (review F13)."""
    for field in _DISCLOSURE_DATE_FIELDS:
        parsed = _parse_date(str(row.get(field, "")))
        if parsed is not None:
            return parsed
    return None


def _parse_date(value: str) -> datetime | None:
    try:
        parsed: date = date.fromisoformat(value[:10])
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
