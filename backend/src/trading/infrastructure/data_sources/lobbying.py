"""U.S. Senate Lobbying Disclosure Act (LDA) adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Queries the public Senate LDA REST API for recently-posted lobbying *filings* (quarterly
registrations / reports), resolves the **client company being lobbied for** to a ticker, and
emits one Signal per (filing, ticker). A lobbying filing is treated as evidence of active
corporate / regulatory engagement by the client firm — an ambiguous-but-active fact, so the
sentiment is a *mild* positive rather than a euphoric one (a firm spending on lobbying is
defending or advancing its commercial interests; this correlates with engagement, not with a
guaranteed favourable outcome). There is no politician attached to a filing, so
``politician_id`` is None.

The endpoint paginates with the classic DRF envelope: each response carries ``count``,
``results`` and a ``next`` absolute URL (or null on the last page). We walk ``next`` up to a
sane cap (``max_pages``) so a single poll ingests more than one page without unbounded fetching
(FR-ING-4 rate-limit courtesy). Results are ordered ``-dt_posted`` (newest first), so the
freshest, most market-relevant filings arrive first. The ``since`` cursor filters out filings
at or before the cursor (already-ingested) on every page.

Lobbying filings are real federal disclosure data with no politician dependency, so every
emitted Signal is stamped ``Provenance.REAL`` (the anti-placeholder guard requires REAL to
drive decisions).

Anonymous access works at a lower rate limit; an optional API key (passed as
``Authorization: Token <key>``) raises it.

API reference: https://lda.senate.gov/api/  (v1; GET /api/v1/filings/)
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

# Per-source reliability prior for lobbying filings (the source_reliability term in conviction
# scoring). Disclosed lobbying is a real, audited fact but only an indirect, lagged read on a
# firm's prospects (it signals engagement, not outcome), so it sits below contracts/statements
# at a deliberately middling 0.5. Defined here, not in data_sources/__init__.py, per the brief.
RELIABILITY_LOBBYING = 0.5

_BASE_URL = "https://lda.senate.gov/api/v1/filings/"
# A lobbying filing reflects active corporate/regulatory engagement by the client firm: an
# ambiguous-but-active read that is mildly bullish, NOT euphoric. Kept well inside [-1, 1].
_LOBBYING_SENTIMENT = 0.25


class LobbyingSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        lookback_days: int = 30,
        api_key: str | None = None,
        max_pages: int = 5,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._lookback_days = lookback_days
        self._api_key = api_key
        self._max_pages = max_pages

    @property
    def name(self) -> str:
        return "lobbying"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        signals: list[Signal] = []
        url: str | None = _BASE_URL
        params: dict[str, Any] | None = self._initial_params()
        for _page in range(self._max_pages):
            if url is None:
                break
            # The `next` URL already carries its own query string, so params are only sent on
            # the first request; afterwards we follow `next` verbatim.
            response = await self._client.get(url, params=params, headers=self._headers())
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            results: list[dict[str, Any]] = body.get("results", [])

            for filing in results:
                posted = _parse_datetime(str(filing.get("dt_posted", "")))
                if posted is None or posted <= since:
                    # Already ingested (or undated). Results are newest-first, but we skip just
                    # this row rather than the rest, to stay robust to ordering surprises.
                    continue
                client = filing.get("client")
                client_name = str(client.get("name", "")) if isinstance(client, dict) else ""
                if not client_name:
                    continue
                tickers = self._entity_resolver.resolve_tickers(client_name)
                if not tickers:
                    continue
                spend = _filing_spend(filing)
                filing_id = str(filing.get("filing_uuid", "") or filing.get("url", ""))
                for ticker in tickers:
                    signals.append(
                        Signal(
                            id=f"lobbying-{filing_id or ticker}-{ticker}",
                            created_at=posted,
                            source_type=SignalSource.STATEMENT,
                            ticker=ticker,
                            politician_id=None,
                            sentiment=_LOBBYING_SENTIMENT,
                            politician_weight=_amount_weight(spend),
                            source_reliability=RELIABILITY_LOBBYING,
                            raw_payload=filing,
                            # Real federal disclosure data, no politician dependency -> REAL.
                            provenance=Provenance.REAL,
                        )
                    )

            url = body.get("next")
            params = None  # the `next` URL is self-contained.
            if not url:
                break
        return signals

    def _initial_params(self) -> dict[str, Any]:
        # Newest-first within the lookback window; the API filters by `filing_year`, so we span
        # the current and previous calendar year to safely cover a lookback that crosses Jan 1.
        now = self._clock.now()
        start_year = (now - timedelta(days=self._lookback_days)).year
        years = sorted({start_year, now.year})
        return {
            "filing_year": ",".join(str(year) for year in years),
            "ordering": "-dt_posted",
        }

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Token {self._api_key}"}
        return {}


def _amount_weight(amount: float) -> float:
    """Map reported lobbying spend to an influence weight in roughly [1.0, 2.0] on a log scale.

    Modelled on the USAspending award-weight helper but recentred for lobbying: reported spend
    is typically tens of thousands to low millions of dollars (vs. contracts in the billions),
    so the curve lifts off around $10k rather than $1M.
    """
    if amount <= 0:
        return 1.0
    return min(2.0, 1.0 + max(0.0, math.log10(amount) - 4.0) / 3.0)


def _filing_spend(filing: dict[str, Any]) -> float:
    """Reported spend for a filing: `income` (registrant fees) or `expenses` (in-house spend).

    A given filing reports one or the other depending on whether the filer is a lobbying firm
    (income) or an in-house lobbying organization (expenses); the other is null. Take whichever
    is present, preferring the larger when both happen to be set.
    """
    income = _to_float(filing.get("income"))
    expenses = _to_float(filing.get("expenses"))
    return max(income, expenses)


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _parse_datetime(value: str) -> datetime | None:
    """Parse an LDA `dt_posted` ISO-8601 timestamp into a UTC-aware datetime.

    The API returns values like ``2024-04-22T13:45:31.123456-04:00`` (offset-aware) or, on some
    fields, a trailing ``Z``. Normalize ``Z`` -> ``+00:00`` and coerce naive values to UTC.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
