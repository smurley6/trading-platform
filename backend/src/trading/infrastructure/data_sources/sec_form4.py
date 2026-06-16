"""SEC EDGAR Form 4 corporate-insider adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Form 4 is the SEC filing an officer, director, or 10%-owner must file when they trade their
OWN company's stock. It is FREE, REAL, key-less data, and — crucially for us — the ownership
document carries the issuer's ticker explicitly (``issuerTradingSymbol``), so there is NO entity
resolution to do: the ticker is a fact in the filing. This makes the source closest in shape to
the USAspending contract adapter (ticker-based, no politician, log-scaled weight, always REAL).

We walk the public "recent filings" Atom feed for form type 4 (newest first), cap the walk at
``max_filings``, and for each filing fetch the accession directory's ``index.json`` to locate the
ownership XML, then parse the ``nonDerivativeTransaction`` rows. An OPEN-MARKET PURCHASE
(transactionCode ``P``) is treated as a bullish, self-funded vote of confidence; an OPEN-MARKET
SALE (``S``) as bearish. Award/grant (``A``), option exercise (``M``), gift (``G``), and any
other coding are NOT discretionary open-market trades, so they are skipped (they carry little
directional signal and would otherwise flood us with comp-driven noise). The weight scales with
the transaction value (shares x price) on a log scale, mirroring ``usaspending._amount_weight``.

Every emitted Signal is stamped ``Provenance.REAL`` — it is genuine SEC disclosure data with no
politician dependency, so it can drive a real decision past the anti-placeholder guard.

CRITICAL: SEC rejects requests without a descriptive ``User-Agent`` (HTTP 403). The injected
``httpx.AsyncClient`` is SHARED across adapters, so we MUST NOT mutate its headers; instead we
pass the UA on every request via the per-request ``headers=`` argument.

Endpoints (verified against live EDGAR, June 2026):
  - Recent Form 4 feed:
      https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom
  - Per-filing directory listing (JSON):
      https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/index.json
  - Ownership document: the ``.xml`` item in that directory (not the ``xsl*`` rendering copy).
"""

import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from trading.application.ports.clock import ClockPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

# Per-source-type reliability prior for corporate-insider trades. Defined HERE (not in the
# data_sources __init__) per the adapter's brief. Form 4 is hard, legally-mandated disclosure
# but is lagged (filed up to two business days after the trade) and indirect, so it sits a touch
# below contract awards yet above noisy statements.
RELIABILITY_INSIDER_TRADE = 0.75

_RECENT_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

# Open-market transaction codes we map to a directional sentiment. Everything else (A=grant/award,
# M=option exercise, G=gift, F=tax withholding, ...) is non-discretionary noise -> skip.
_PURCHASE_CODE = "P"
_SALE_CODE = "S"
_PURCHASE_SENTIMENT = 0.5
_SALE_SENTIMENT = -0.5
_SENTIMENT_BY_CODE = {_PURCHASE_CODE: _PURCHASE_SENTIMENT, _SALE_CODE: _SALE_SENTIMENT}


class SecForm4Source:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        clock: ClockPort,
        user_agent: str,
        lookback_days: int = 3,
        max_filings: int = 100,
    ) -> None:
        self._client = client
        self._clock = clock
        self._user_agent = user_agent
        self._lookback_days = lookback_days
        self._max_filings = max_filings

    @property
    def name(self) -> str:
        return "sec_form4"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        feed = await self._get_text(_RECENT_FEED)
        # Newest-first in the feed; cap the walk so a single poll never fans out unboundedly.
        filing_urls = _parse_recent_feed(feed)[: self._max_filings]

        signals: list[Signal] = []
        for index_url in filing_urls:
            xml_url = await self._resolve_ownership_xml_url(index_url)
            if xml_url is None:
                continue
            document = await self._get_text(xml_url)
            signals.extend(self._parse_ownership_document(document, since, source_url=xml_url))
        return signals

    async def _resolve_ownership_xml_url(self, index_url: str) -> str | None:
        """From a filing's ``...-index.htm`` link, fetch the directory ``index.json`` and pick the
        ownership XML (the bare ``.xml`` item, not the ``xsl*`` HTML-rendering copy)."""
        base = index_url.rsplit("/", 1)[0]
        try:
            listing: Any = (await self._get_response(f"{base}/index.json")).json()
        except (httpx.HTTPError, ValueError):
            return None
        items = listing.get("directory", {}).get("item", []) if isinstance(listing, dict) else []
        for item in items:
            name = item.get("name", "") if isinstance(item, dict) else ""
            if name.endswith(".xml") and "/" not in name and not name.startswith("xsl"):
                return f"{base}/{name}"
        return None

    def _parse_ownership_document(
        self, document: str, since: datetime, *, source_url: str
    ) -> list[Signal]:
        try:
            root = ET.fromstring(document)
        except ET.ParseError:
            return []

        ticker = _text(root, "issuer/issuerTradingSymbol").upper()
        if not ticker:
            return []
        owner = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
        issuer_cik = _text(root, "issuer/issuerCik")

        signals: list[Signal] = []
        for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
            code = _text(txn, "transactionCoding/transactionCode")
            sentiment = _SENTIMENT_BY_CODE.get(code)
            if sentiment is None:
                continue  # A / M / G / F / ... -> not a discretionary open-market trade.

            txn_date = _parse_date(_text(txn, "transactionDate/value"))
            if txn_date is None or txn_date <= since:
                continue

            shares = _to_float(_text(txn, "transactionAmounts/transactionShares/value"))
            price = _to_float(_text(txn, "transactionAmounts/transactionPricePerShare/value"))
            value = shares * price

            payload: dict[str, Any] = {
                "ticker": ticker,
                "issuer_cik": issuer_cik,
                "owner": owner,
                "transaction_code": code,
                "transaction_shares": shares,
                "transaction_price_per_share": price,
                "transaction_value": value,
                "transaction_date": txn_date.date().isoformat(),
                "source_url": source_url,
            }
            signals.append(
                Signal(
                    id=f"sec-form4-{issuer_cik or ticker}-{txn_date.date().isoformat()}-{code}",
                    created_at=txn_date,
                    source_type=SignalSource.DISCLOSED_TRADE,
                    ticker=ticker,
                    politician_id=None,
                    sentiment=sentiment,
                    politician_weight=_value_weight(value),
                    source_reliability=RELIABILITY_INSIDER_TRADE,
                    raw_payload=payload,
                    # Real SEC disclosure, no politician dependency -> REAL.
                    provenance=Provenance.REAL,
                )
            )
        return signals

    async def _get_response(self, url: str) -> httpx.Response:
        # Per-request UA only: the client is shared, so we never mutate its headers (a sibling
        # adapter could be mid-flight). SEC 403s any request lacking a descriptive User-Agent.
        response = await self._client.get(url, headers={"User-Agent": self._user_agent})
        response.raise_for_status()
        return response

    async def _get_text(self, url: str) -> str:
        return (await self._get_response(url)).text


def _parse_recent_feed(feed: str) -> list[str]:
    """Return the filing index-page URLs from the recent-Form-4 Atom feed, in feed order."""
    try:
        root = ET.fromstring(feed)
    except ET.ParseError:
        return []
    urls: list[str] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        # Defensive: only accept entries the feed labels as form type 4.
        category = entry.find(f"{_ATOM_NS}category")
        if category is not None and category.get("term") not in (None, "4"):
            continue
        link = entry.find(f"{_ATOM_NS}link")
        href = link.get("href") if link is not None else None
        if href:
            urls.append(href)
    return urls


def _value_weight(value: float) -> float:
    """Map transaction value (shares x price) to an influence weight in roughly [1.0, 2.0] on a
    log scale. Modelled on ``usaspending._amount_weight``: a $1M trade ~= 1.0, scaling up toward
    2.0 for nine-figure trades. Small/zero-value rows floor at 1.0."""
    if value <= 0:
        return 1.0
    return min(2.0, 1.0 + max(0.0, math.log10(value) - 6.0) / 3.0)


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(value: str) -> datetime | None:
    try:
        parsed: date = date.fromisoformat(value[:10])
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path)
    if found is None or found.text is None:
        return ""
    return found.text.strip()
