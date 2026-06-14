"""USAspending.gov government-contract adapter (FR-ING-2).

Queries the public `spending_by_award` API for recent contract awards, resolves the
recipient company to a ticker, and emits a contract Signal. A contract award is treated as
a bullish, high-reliability fact about the recipient; the weight scales with award size.
There is no politician attached to a contract, so `politician_id` is None.

API reference: https://api.usaspending.gov/api/v2/search/spending_by_award/
"""

import math
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_CONTRACT

_ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
# A contract award is a positive but not euphoric signal about the recipient.
_CONTRACT_SENTIMENT = 0.6


class UsaSpendingSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        lookback_days: int = 30,
        page_limit: int = 100,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._lookback_days = lookback_days
        self._page_limit = page_limit

    @property
    def name(self) -> str:
        return "usaspending"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        response = await self._client.post(_ENDPOINT, json=self._request_body())
        response.raise_for_status()
        results: list[dict[str, Any]] = response.json().get("results", [])

        signals: list[Signal] = []
        for award in results:
            recipient = str(award.get("Recipient Name", ""))
            tickers = self._entity_resolver.resolve_tickers(recipient)
            if not tickers:
                continue
            awarded = _parse_date(str(award.get("Action Date", ""))) or self._clock.now()
            if awarded <= since:
                continue
            amount = _to_float(award.get("Award Amount"))
            for ticker in tickers:
                signals.append(
                    Signal(
                        id=f"contract-{award.get('Award ID', ticker)}-{ticker}",
                        created_at=awarded,
                        source_type=SignalSource.CONTRACT,
                        ticker=ticker,
                        politician_id=None,
                        sentiment=_CONTRACT_SENTIMENT,
                        politician_weight=_amount_weight(amount),
                        source_reliability=RELIABILITY_CONTRACT,
                        raw_payload=award,
                    )
                )
        return signals

    def _request_body(self) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).isoformat()
        end = self._clock.now().date().isoformat()
        return {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],  # contract award types
                "time_period": [{"start_date": start, "end_date": end}],
            },
            "fields": ["Award ID", "Recipient Name", "Award Amount", "Action Date"],
            "limit": self._page_limit,
            "sort": "Award Amount",
            "order": "desc",
        }


def _amount_weight(amount: float) -> float:
    """Map award size to an influence weight in roughly [1.0, 2.0] on a log scale."""
    if amount <= 0:
        return 1.0
    return min(2.0, 1.0 + max(0.0, math.log10(amount) - 6.0) / 3.0)


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _parse_date(value: str) -> datetime | None:
    try:
        parsed: date = date.fromisoformat(value[:10])
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
