"""Federal Register (Public Inspection) adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Queries the public Federal Register API's ``documents.json`` endpoint for recently published
agency actions (Rules, Proposed Rules, Presidential Documents), resolves any regulated entity
named in the document title/abstract to a ticker, and emits a regulatory-catalyst
:class:`~trading.domain.signals.Signal`. A finalized agency Rule that names a tradeable sponsor
(a drug approval standard for "Pfizer", an emissions rule touching "Moderna") is a regulatory
catalyst for that issuer; the direction depends on the document *type* and the *agency*.

Why the Federal Register ``documents.json`` endpoint: it is the single free, keyless, authoritative
feed that ties an agency *action with a publication date* to the *regulated entity named in the
text*. Each result carries ``title``, ``abstract``, ``type``, ``publication_date``, an ``agencies``
array, and a stable ``document_number`` for traceability. We concatenate title + abstract and hand
it to the injected :class:`~trading.application.ports.nlp.EntityResolverPort`, exactly as the FDA
and news legs do; documents that resolve to no tracked ticker carry no market relevance and are
dropped (logged).

Direction (sentiment) is derived from the document type and agency, not from prose:
  * ``Rule`` published by the FDA -> ``+0.6`` (a regulatory-approval catalyst, bullish).
  * ``Rule`` from any other agency -> ``+0.5`` (a regulatory action, neutral-to-positive).
  * ``Proposed Rule`` -> ``+0.1`` (uncertainty; a proposal may burden or benefit).
  * ``Presidential Document`` -> scored toward the ticker via the injected
    :class:`~trading.application.ports.nlp.SentimentAnalyzerPort` when one is supplied, else a
    modest ``+0.3`` fallback (an executive action is a catalyst of ambiguous polarity).

The endpoint paginates with a cursor exposed as ``next_page_url``; we follow it up to a sane cap
(``max_pages``) so a single poll ingests more than one page without unbounded fetching (FR-ING-4
rate-limit courtesy). Server-side we filter to documents published in the lookback window via
``publication_date.gte``; the ``since`` cursor then drops any document at or before it client-side
so a poll never re-emits a document it has already seen.

Federal Register data is live, authoritative federal data with no politician dependency, so every
emitted Signal is stamped ``Provenance.REAL`` (the anti-placeholder guard requires REAL to drive
real decisions). The adapter is default OFF: an adapter constructed with ``enabled=False`` (the
default) yields no signals and makes no network call.

API reference: https://www.federalregister.gov/developers/documentation/api/v1 and
https://www.federalregister.gov/api/v1/documents.json
"""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort, SentimentAnalyzerPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

logger = logging.getLogger(__name__)

# Source-type reliability priors (the source_reliability term in conviction scoring). A Federal
# Register document is an authoritative federal fact, but the *market reaction* is implementation-
# and industry-dependent, so it sits at or below the FDA regulatory prior. FDA rules get the
# regulatory prior; other agencies a touch lower. Defined HERE per this adapter's brief.
RELIABILITY_FDA_REGULATORY = 0.7
RELIABILITY_REGULATORY = 0.65

_ENDPOINT = "https://www.federalregister.gov/api/v1/documents.json"

# Type/agency-conditioned sentiment priors (see the module docstring). A finalized rule is a
# clearer catalyst than a proposal; the FDA leg reads as an approval catalyst.
_RULE_FDA_SENTIMENT = 0.6
_RULE_OTHER_SENTIMENT = 0.5
_PROPOSED_RULE_SENTIMENT = 0.1
# Fallback polarity for an executive action when no sentiment analyzer is injected.
_PRESIDENTIAL_FALLBACK_SENTIMENT = 0.3

# A regulatory action carries no dollar amount to scale by, so every resolved document gets the
# same baseline influence weight (>= 0), like the FDA approval leg's approval_weight.
_DOCUMENT_WEIGHT = 1.0

# Document-type labels as the Federal Register API spells them.
_TYPE_RULE = "Rule"
_TYPE_PROPOSED_RULE = "Proposed Rule"
_TYPE_PRESIDENTIAL = "Presidential Document"

# Agency slug fragment identifying the FDA (its agency record slug is
# ``food-and-drug-administration``); an FDA rule reads as an approval catalyst.
_FDA_SLUG_FRAGMENT = "food-and-drug-administration"


class FederalRegisterSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        sentiment: SentimentAnalyzerPort | None = None,
        enabled: bool = False,
        lookback_days: int = 30,
        api_key: str | None = None,
        per_page: int = 100,
        max_pages: int = 5,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._sentiment = sentiment
        self._enabled = enabled
        self._lookback_days = lookback_days
        self._api_key = api_key
        self._per_page = per_page
        self._max_pages = max_pages

    @property
    def name(self) -> str:
        return "federal_register"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        # Default OFF: a disabled adapter yields nothing and never opens a socket.
        if not self._enabled:
            return []

        signals: list[Signal] = []
        url: str | None = _ENDPOINT
        params: dict[str, Any] | None = self._params()
        for _page in range(self._max_pages):
            if url is None:
                break
            # The cursor 'next_page_url' already carries every query param, so only the first
            # request needs our params; following ones pass the URL as-is.
            response = await self._client.get(url, params=params)
            if response.status_code == httpx.codes.NOT_FOUND:
                # A zero-match search is a normal empty result, not an error: stop the walk.
                break
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            results = body.get("results")
            if not isinstance(results, list):
                break

            for record in results:
                if not isinstance(record, dict):
                    continue
                signals.extend(self._records_to_signals(record, since))

            next_url = body.get("next_page_url")
            url = next_url if isinstance(next_url, str) and next_url else None
            params = None

        return signals

    def _records_to_signals(self, record: dict[str, Any], since: datetime) -> list[Signal]:
        published = _parse_fr_date(str(record.get("publication_date", "")))
        if published is None or published <= since:
            # No usable date, or already ingested (the cursor filters at or before itself): skip.
            return []

        text = f"{record.get('title') or ''} {record.get('abstract') or ''}".strip()
        tickers = self._entity_resolver.resolve_tickers(text)
        if not tickers:
            # No tradeable entity named -> no market relevance, drop (like the FDA/news legs).
            logger.debug("federal_register: no ticker resolved for %r; skipping", text[:80])
            return []

        doc_type = str(record.get("type", ""))
        is_fda = _mentions_fda(record)
        document_number = str(record.get("document_number", ""))

        signals: list[Signal] = []
        for ticker in tickers:
            sentiment = self._sentiment_for(doc_type, is_fda, text, ticker)
            signals.append(
                Signal(
                    id=f"fr-document-{document_number or ticker}-{ticker}",
                    created_at=published,
                    source_type=SignalSource.STATEMENT,
                    ticker=ticker,
                    politician_id=None,
                    sentiment=sentiment,
                    politician_weight=_DOCUMENT_WEIGHT,
                    source_reliability=(
                        RELIABILITY_FDA_REGULATORY if is_fda else RELIABILITY_REGULATORY
                    ),
                    raw_payload=record,
                    # Real federal regulatory data, no politician dependency -> REAL.
                    provenance=Provenance.REAL,
                )
            )
        return signals

    def _sentiment_for(self, doc_type: str, is_fda: bool, text: str, ticker: str) -> float:
        if doc_type == _TYPE_PROPOSED_RULE:
            return _PROPOSED_RULE_SENTIMENT
        if doc_type == _TYPE_PRESIDENTIAL:
            if self._sentiment is not None:
                # An executive action's polarity is genuinely ambiguous; let the analyzer read it.
                return self._sentiment.score(text, ticker)
            return _PRESIDENTIAL_FALLBACK_SENTIMENT
        # Any Rule (or unknown type) reads as a regulatory action; the FDA leg is an approval read.
        return _RULE_FDA_SENTIMENT if is_fda else _RULE_OTHER_SENTIMENT

    def _params(self) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).strftime(
            "%Y-%m-%d"
        )
        params: dict[str, Any] = {
            "per_page": self._per_page,
            "order": "newest",
            "conditions[publication_date][gte]": start,
        }
        if self._api_key is not None:
            # An optional (free) api.data.gov key is passed as the 'api_key' query param; the
            # endpoint works keyless, so this is never hardcoded here.
            params["api_key"] = self._api_key
        return params


def _mentions_fda(record: dict[str, Any]) -> bool:
    """True if any agency on the document is the Food and Drug Administration."""
    agencies = record.get("agencies")
    if not isinstance(agencies, list):
        return False
    for agency in agencies:
        if not isinstance(agency, dict):
            continue
        slug = str(agency.get("slug", "")).lower()
        name = str(agency.get("name", "")).lower()
        if _FDA_SLUG_FRAGMENT in slug or "food and drug administration" in name:
            return True
    return False


def _parse_fr_date(value: str) -> datetime | None:
    """Parse a Federal Register ``YYYY-MM-DD`` date into a UTC datetime at midnight, or None."""
    try:
        parsed: date = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
