"""SEC EDGAR "latest filings" Form 4 Atom adapter (FR-ING-2, FR-ING-3, FR-ING-4).

The political legs we already run are slow: House PTRs are lagged up to 45 days and Senate EFD
similarly. By contrast SEC EDGAR's *latest filings* Atom feed publishes Form 4 corporate-insider
transactions within minutes-to-hours of acceptance — it is the closest thing to a real-time
disclosure signal we can poll for free with no key. This adapter walks that feed and emits one
Signal per *resolvable* issuer.

Why parse the Atom feed and not the ownership XML (as :mod:`sec_form4` does): this is the
near-real-time WATCH leg. The feed entry already carries everything we need to fire fast — the
issuer name, the accession id, and the acceptance timestamp — without the extra per-filing
``index.json`` -> ownership-XML round trips that add latency. The feed entry does NOT carry the
issuer's ticker (only the company name and CIK), so unlike ``sec_form4`` we MUST resolve the
company to a ticker via the injected :class:`EntityResolverPort`; entries whose issuer does not
resolve are dropped (logged), exactly as the news/FDA legs drop unresolved mentions.

Direction: the feed entry alone does not say buy vs sell (that lives in the ownership XML), so we
score the entry text toward the resolved ticker with the injected :class:`SentimentAnalyzerPort`,
mirroring :mod:`news`; a zero score (no stance) is skipped rather than emitted as noise. Every
emitted Signal is stamped :class:`Provenance.REAL` — it is genuine SEC disclosure data with no
politician dependency — and classified :data:`SignalSource.DISCLOSED_TRADE`, exactly as the
sibling Form 4 adapter classifies insider trades.

Only entries whose acceptance timestamp is strictly after the ``since`` cursor are emitted, so a
poll never re-emits a filing it has already seen.

CRITICAL: SEC rejects requests without a descriptive ``User-Agent`` (HTTP 403). The injected
``httpx.AsyncClient`` is SHARED across adapters, so we MUST NOT mutate its headers; instead we
pass the UA on every request via the per-request ``headers=`` argument, exactly as
:mod:`sec_form4` does (the UA is sourced from ``SEC_FORM4_USER_AGENT``).

Feed: https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&count=100&output=atom
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from trading.application.ports.nlp import EntityResolverPort, SentimentAnalyzerPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

logger = logging.getLogger(__name__)

# Per-source-type reliability prior for corporate-insider trades. Kept identical to the sibling
# Form 4 leg (it is the same underlying disclosure), defined HERE per this adapter's brief.
RELIABILITY_INSIDER_TRADE = 0.75

_RECENT_FEED = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4"
    "&company=&dateb=&owner=include&count=100&output=atom"
)
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

# A Form 4 filing in the feed does not, on its own, encode a dollar amount to scale by, so every
# resolved filing gets the same baseline influence weight (>= 0), like the FDA approval leg.
_FILING_WEIGHT = 1.0


class SecEdgarAtomSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        sentiment: SentimentAnalyzerPort,
        user_agent: str,
        max_filings: int = 100,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._sentiment = sentiment
        self._user_agent = user_agent
        self._max_filings = max_filings

    @property
    def name(self) -> str:
        return "sec_edgar_atom"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        feed = await self._get_text(_RECENT_FEED)
        # Newest-first in the feed; cap the walk so a single poll never fans out unboundedly.
        entries = _parse_recent_feed(feed)[: self._max_filings]

        signals: list[Signal] = []
        for entry in entries:
            if entry.updated <= since:
                # Already ingested (the cursor filters at or before itself) -> skip.
                continue
            tickers = self._entity_resolver.resolve_tickers(entry.company)
            if not tickers:
                # Issuer name does not resolve to a tracked ticker -> drop, like the news/FDA legs.
                logger.debug("sec_edgar_atom: unresolved issuer %r; skipping", entry.company)
                continue
            for ticker in tickers:
                sentiment = self._sentiment.score(entry.title, ticker)
                if sentiment == 0.0:
                    continue
                signals.append(self._to_signal(entry, ticker, sentiment))
        return signals

    def _to_signal(self, entry: "_FeedEntry", ticker: str, sentiment: float) -> Signal:
        raw_payload: dict[str, Any] = {
            "company": entry.company,
            "issuer_cik": entry.cik,
            "accession": entry.accession,
            "ticker": ticker,
            "title": entry.title,
            "link": entry.link,
            "updated": entry.updated.isoformat(),
        }
        return Signal(
            id=f"sec-edgar-atom-{entry.accession or entry.cik}-{ticker}",
            created_at=entry.updated,
            source_type=SignalSource.DISCLOSED_TRADE,
            ticker=ticker,
            politician_id=None,
            sentiment=sentiment,
            politician_weight=_FILING_WEIGHT,
            source_reliability=RELIABILITY_INSIDER_TRADE,
            raw_payload=raw_payload,
            # Real SEC disclosure, no politician dependency -> REAL.
            provenance=Provenance.REAL,
        )

    async def _get_response(self, url: str) -> httpx.Response:
        # Per-request UA only: the client is shared, so we never mutate its headers (a sibling
        # adapter could be mid-flight). SEC 403s any request lacking a descriptive User-Agent.
        response = await self._client.get(url, headers={"User-Agent": self._user_agent})
        response.raise_for_status()
        return response

    async def _get_text(self, url: str) -> str:
        return (await self._get_response(url)).text


class _FeedEntry:
    """One parsed Form 4 ``<entry>`` from the recent-filings Atom feed."""

    __slots__ = ("accession", "cik", "company", "link", "title", "updated")

    def __init__(
        self,
        *,
        company: str,
        cik: str,
        accession: str,
        title: str,
        link: str,
        updated: datetime,
    ) -> None:
        self.company = company
        self.cik = cik
        self.accession = accession
        self.title = title
        self.link = link
        self.updated = updated


def _parse_recent_feed(feed: str) -> list[_FeedEntry]:
    """Parse the recent-Form-4 Atom feed into entries, in feed order (newest first)."""
    try:
        root = ET.fromstring(feed)
    except ET.ParseError:
        logger.warning("sec_edgar_atom: could not parse Atom feed body; skipping")
        return []

    entries: list[_FeedEntry] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        # Defensive: only accept entries the feed labels as form type 4.
        category = entry.find(f"{_ATOM_NS}category")
        if category is not None and category.get("term") not in (None, "4"):
            continue

        title = _text(entry, f"{_ATOM_NS}title")
        updated = _parse_atom_datetime(_text(entry, f"{_ATOM_NS}updated"))
        if not title or updated is None:
            continue

        company = _issuer_name(title)
        link = entry.find(f"{_ATOM_NS}link")
        href = (link.get("href") or "") if link is not None else ""
        entries.append(
            _FeedEntry(
                company=company,
                cik=_issuer_cik(title),
                accession=_accession(_text(entry, f"{_ATOM_NS}id"), href),
                title=title,
                link=href,
                updated=updated,
            )
        )
    return entries


def _issuer_name(title: str) -> str:
    """Pull the issuer/company name out of a feed title like
    ``4 - APPLE INC (0000320193) (Issuer)`` -> ``APPLE INC``."""
    body = title.split(" - ", 1)[1] if " - " in title else title
    paren = body.find(" (")
    return (body[:paren] if paren != -1 else body).strip()


def _issuer_cik(title: str) -> str:
    """Pull the 10-digit CIK out of a feed title, or "" if absent."""
    start = title.find("(")
    if start == -1:
        return ""
    end = title.find(")", start)
    if end == -1:
        return ""
    cik = title[start + 1 : end].strip()
    return cik if cik.isdigit() else ""


def _accession(entry_id: str, href: str) -> str:
    """Resolve the accession number from the entry ``id`` (preferred) or its filing link.

    The id looks like ``urn:tag:sec.gov,2008:accession-number=0000320193-26-000077``; the link's
    last path component is ``0000320193-26-000077-index.htm``. Either yields the accession.
    """
    marker = "accession-number="
    if marker in entry_id:
        return entry_id.split(marker, 1)[1].strip()
    tail = href.rsplit("/", 1)[-1]
    return tail.removesuffix("-index.htm").removesuffix("-index.html").strip()


def _parse_atom_datetime(value: str) -> datetime | None:
    """Parse an Atom RFC-3339 timestamp (e.g. ``2026-06-12T16:30:11-04:00``) to a UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path)
    if found is None or found.text is None:
        return ""
    return found.text.strip()
