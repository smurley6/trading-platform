"""USAspending.gov government-contract adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Queries the public `spending_by_award` API for recent contract awards, resolves the
recipient company to a ticker, and emits a contract Signal. A contract award is treated as
a bullish, high-reliability fact about the recipient; the weight scales with award size.
There is no politician attached to a contract, so `politician_id` is None.

The endpoint paginates: the request body carries a 1-based ``page`` field and the response's
``page_metadata.hasNext`` tells us whether to fetch the next page. We walk pages up to a sane
cap (``max_pages``) so a single poll can ingest more than the per-page ``limit`` of awards
without unbounded fetching (FR-ING-4 rate-limit courtesy). Rows are sorted by award amount
descending, so the largest, most market-relevant awards arrive first and the cap drops only
the long tail of small contracts. The ``since`` cursor filters out rows at or before the
cursor (already-ingested awards) on every page.

A contract award is real federal data with no politician dependency, so every emitted Signal
is stamped ``Provenance.REAL`` (the anti-placeholder guard requires REAL to drive decisions).

API reference: https://api.usaspending.gov/api/v2/search/spending_by_award/
"""

import math
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_CONTRACT

_ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
# Contract ("D"=definitive, "C"=delivery order, "B"=purchase order, "A"=BPA call) award types.
_CONTRACT_AWARD_TYPE_CODES = ("A", "B", "C", "D")
# Display fields requested; these names become the keys on each result row.
_AWARD_ID_FIELD = "Award ID"
_RECIPIENT_FIELD = "Recipient Name"
_AMOUNT_FIELD = "Award Amount"
_ACTION_DATE_FIELD = "Action Date"
_FIELDS = (_AWARD_ID_FIELD, _RECIPIENT_FIELD, _AMOUNT_FIELD, _ACTION_DATE_FIELD)
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
        max_pages: int = 10,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._lookback_days = lookback_days
        self._page_limit = page_limit
        self._max_pages = max_pages

    @property
    def name(self) -> str:
        return "usaspending"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        signals: list[Signal] = []
        for page in range(1, self._max_pages + 1):
            response = await self._client.post(_ENDPOINT, json=self._request_body(page))
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            results: list[dict[str, Any]] = body.get("results", [])

            for award in results:
                awarded = _parse_date(str(award.get(_ACTION_DATE_FIELD, ""))) or self._clock.now()
                if awarded <= since:
                    # Already ingested (rows are amount-sorted, not date-sorted, so a stale row
                    # does not imply later rows are stale): skip just this row.
                    continue
                recipient = str(award.get(_RECIPIENT_FIELD, ""))
                tickers = self._entity_resolver.resolve_tickers(recipient)
                if not tickers:
                    continue
                amount = _to_float(award.get(_AMOUNT_FIELD))
                award_id = str(award.get(_AWARD_ID_FIELD, ""))
                for ticker in tickers:
                    signals.append(
                        Signal(
                            id=f"contract-{award_id or ticker}-{ticker}",
                            created_at=awarded,
                            source_type=SignalSource.CONTRACT,
                            ticker=ticker,
                            politician_id=None,
                            sentiment=_CONTRACT_SENTIMENT,
                            politician_weight=_amount_weight(amount),
                            source_reliability=RELIABILITY_CONTRACT,
                            raw_payload=award,
                            # Real federal contract data, no politician dependency -> REAL.
                            provenance=Provenance.REAL,
                        )
                    )

            # An empty page or a final page (per the API's own metadata) ends the walk; otherwise
            # advance, bounded by ``max_pages`` so a poll never fans out unboundedly.
            if not results or not _has_next_page(body):
                break
        return signals

    def _request_body(self, page: int) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).isoformat()
        end = self._clock.now().date().isoformat()
        return {
            "filters": {
                "award_type_codes": list(_CONTRACT_AWARD_TYPE_CODES),
                "time_period": [{"start_date": start, "end_date": end}],
            },
            "fields": list(_FIELDS),
            "limit": self._page_limit,
            "page": page,
            "sort": _AMOUNT_FIELD,
            "order": "desc",
        }


def _has_next_page(body: dict[str, Any]) -> bool:
    metadata = body.get("page_metadata")
    if isinstance(metadata, dict):
        return bool(metadata.get("hasNext", False))
    return False


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
