"""Regulations.gov API v4 federal-docket adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Queries the public Regulations.gov ``/v4/documents`` endpoint for regulatory documents posted in
a lookback window, resolves any company named in the document title to a ticker, and emits a
regulatory-risk :class:`~trading.domain.signals.Signal`. A proposed rule open for public comment
is a fresh regulatory-risk catalyst for the affected issuer (bearish); a withdrawal/deregulation
of a rule reads bullish. There is no politician attached to a regulatory action, so
``politician_id`` is None, and the signal is classified :data:`SignalSource.STATEMENT` (a
regulatory statement), exactly like the sibling FDA approval leg.

Why the ``/v4/documents`` endpoint: it is the endpoint that ties together, in one record, the
three things this catalyst needs — a *document type* (Proposed Rule / Rule / Notice…) with a
*posted date*, the *agency*, and a *title* we can resolve to an issuer. The endpoint filters
server-side on ``filter[postedDate][ge|le]`` (a ``YYYY-MM-DD`` window in Eastern time) and pages
with ``page[size]`` (max 250) and ``page[number]`` (max 20); ``meta`` reports the page shape. We
walk pages up to a sane cap (``max_pages``) so a single poll ingests more than one page without
unbounded fetching (FR-ING-4 rate-limit courtesy — the free key allows 1,000 requests/hour). The
``since`` cursor then drops any document posted at or before the cursor (already-ingested events)
client-side so a poll never re-emits.

Direction: the document type alone gives the stance — a proposed rule/rule that is open for
comment is regulatory risk (bearish); a withdrawn action is deregulation (bullish). A document
with no directional read (a notice, supporting material, an unclassified type) scores 0.0 and is
skipped rather than emitted as noise, mirroring the news/SEC-Atom legs. Titles that resolve to no
tracked ticker are dropped (logged), exactly as the FDA and SEC-Atom legs drop unresolved
mentions.

The influence weight scales up for a *short* comment deadline (an imminent deadline is a nearer,
sharper catalyst) between the baseline 1.0 and a 2.0 ceiling.

Regulations.gov is real federal regulatory data with no politician dependency, so every emitted
Signal is stamped ``Provenance.REAL`` (the anti-placeholder guard requires REAL to drive
decisions). Default OFF: with no ``api_key`` configured the adapter yields no signals and makes
no network call (the endpoint rejects unauthenticated requests), so an unconfigured adapter is
inert at construction and at poll time.

API reference: https://open.gsa.gov/api/regulationsgov/ and https://api.regulations.gov/v4/documents
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

logger = logging.getLogger(__name__)

# Source-type reliability prior (the source_reliability term in conviction scoring). Regulations.gov
# is an authoritative federal fact, but the *market reaction* to a docket is indirect (priced-in
# expectations, scope), so it sits just below the FDA approval prior (0.7). Lives here (not in
# data_sources/__init__.py) per the build constraints for this adapter.
RELIABILITY_REGULATORY = 0.65

_ENDPOINT = "https://api.regulations.gov/v4/documents"

# A proposed rule / rule that is open for public comment is a regulatory-risk (bearish) catalyst
# for the affected issuer; a withdrawn action is deregulation (bullish). Magnitudes are modest:
# the market impact of a docket is real but indirect.
_REGULATORY_RISK_SENTIMENT = -0.4
_DEREGULATION_SENTIMENT = 0.3

# Baseline influence weight and the ceiling a short comment deadline can scale it to.
_BASELINE_WEIGHT = 1.0
_MAX_WEIGHT = 2.0
# A comment window at or below this many days counts as "imminent" and earns the full multiplier;
# longer windows interpolate down toward the baseline.
_URGENT_WINDOW_DAYS = 7
# Document types that carry a rulemaking stance (everything else is a notice / support / other).
_RULE_TYPES = frozenset({"Proposed Rule", "Rule"})


class RegulationsGovSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        api_key: str | None = None,
        lookback_days: int = 7,
        page_size: int = 250,
        max_pages: int = 5,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._api_key = api_key
        self._lookback_days = lookback_days
        self._page_size = page_size
        self._max_pages = max_pages

    @property
    def name(self) -> str:
        return "regulations_gov"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        if not self._api_key:
            # Default OFF: unconfigured -> no key -> no network call, no signals. The endpoint
            # rejects unauthenticated requests, so there is nothing to fetch.
            return []

        signals: list[Signal] = []
        for page in range(1, self._max_pages + 1):
            response = await self._client.get(
                _ENDPOINT,
                params=self._params(page),
                headers={"X-Api-Key": self._api_key},
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            data: list[dict[str, Any]] = body.get("data", [])

            for document in data:
                signal = self._to_signal(document, since)
                if signal is not None:
                    signals.append(signal)

            # A short page (fewer rows than the page size) is the last page; stop the walk.
            if len(data) < self._page_size:
                break
        return signals

    def _to_signal(self, document: dict[str, Any], since: datetime) -> Signal | None:
        attributes = document.get("attributes")
        if not isinstance(attributes, dict):
            return None

        posted = _parse_iso_datetime(str(attributes.get("postedDate", "")))
        if posted is None or posted <= since:
            # No posted date, or already ingested (the cursor filters at or before itself): skip.
            return None

        sentiment = _direction_sentiment(attributes)
        if sentiment == 0.0:
            # No directional read (a notice, supporting material, unclassified type) -> skip.
            return None

        title = str(attributes.get("title", ""))
        tickers = self._entity_resolver.resolve_tickers(title)
        if not tickers:
            logger.debug("regulations_gov: unresolved title %r; skipping", title)
            return None

        weight = _BASELINE_WEIGHT * _deadline_multiplier(attributes, posted)
        document_id = str(document.get("id", "")) or str(attributes.get("docketId", ""))
        ticker = tickers[0]
        return Signal(
            id=f"regulations-{document_id}-{ticker}",
            created_at=posted,
            source_type=SignalSource.STATEMENT,
            ticker=ticker,
            politician_id=None,
            sentiment=sentiment,
            politician_weight=weight,
            source_reliability=RELIABILITY_REGULATORY,
            raw_payload=attributes,
            # Real federal regulatory data, no politician dependency -> REAL.
            provenance=Provenance.REAL,
        )

    def _params(self, page: int) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).strftime(
            "%Y-%m-%d"
        )
        end = self._clock.now().date().strftime("%Y-%m-%d")
        return {
            # Server-side window: documents posted on-or-after start and on-or-before end.
            "filter[postedDate][ge]": start,
            "filter[postedDate][le]": end,
            "page[size]": self._page_size,
            "page[number]": page,
            "sort": "-postedDate",
        }


def _direction_sentiment(attributes: dict[str, Any]) -> float:
    """Stance from the document type: withdrawal reads bullish, an open rulemaking bearish.

    A withdrawn/deregulatory action is a positive catalyst for the affected issuer. A proposed
    rule or rule that is open for public comment is a regulatory-risk catalyst (bearish). Anything
    else (a notice, supporting material, an unclassified type, or a rule that is not open for
    comment) carries no clean directional read and scores 0.0 (skipped by the caller).
    """
    if bool(attributes.get("withdrawn")):
        return _DEREGULATION_SENTIMENT
    document_type = str(attributes.get("documentType", ""))
    if document_type in _RULE_TYPES and bool(attributes.get("openForComment")):
        return _REGULATORY_RISK_SENTIMENT
    return 0.0


def _deadline_multiplier(attributes: dict[str, Any], posted: datetime) -> float:
    """Scale weight up for a short comment window, in [1.0, 2.0].

    A nearer comment deadline is a sharper, more imminent catalyst. We measure the window from the
    posted date to ``commentEndDate``: a window at or below ``_URGENT_WINDOW_DAYS`` earns the full
    multiplier; longer windows interpolate linearly down toward the baseline over a 90-day span; a
    missing/unparseable deadline falls back to the neutral baseline.
    """
    end = _parse_iso_datetime(str(attributes.get("commentEndDate", "")))
    if end is None:
        return 1.0
    window_days = (end - posted).days
    if window_days <= _URGENT_WINDOW_DAYS:
        return _MAX_WEIGHT / _BASELINE_WEIGHT
    # Interpolate: shrink the extra weight linearly to zero over a 90-day comment window.
    span = 90 - _URGENT_WINDOW_DAYS
    over = min(window_days - _URGENT_WINDOW_DAYS, span)
    extra = (_MAX_WEIGHT - _BASELINE_WEIGHT) * (1 - over / span)
    return (_BASELINE_WEIGHT + extra) / _BASELINE_WEIGHT


def _parse_iso_datetime(value: str) -> datetime | None:
    """Parse a Regulations.gov ISO-8601 timestamp (``2026-06-22T04:00:00Z``) to UTC, or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
