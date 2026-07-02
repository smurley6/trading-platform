"""SEC EDGAR Form 4 corporate-insider adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Form 4 is the SEC filing an officer, director, or 10%-owner must file when they trade their
OWN company's stock. It is FREE, REAL, key-less data, and — crucially for us — the ownership
document carries the issuer's ticker explicitly (``issuerTradingSymbol``), so there is NO entity
resolution to do: the ticker is a fact in the filing. This makes the source closest in shape to
the USAspending contract adapter (ticker-based, no politician, log-scaled weight, always REAL).

Two independent fetch paths share the same transaction-classification core
(:func:`_classify_transactions`: P/S open-market only, A/M/G/F/... skipped, value-scaled weight):

  * ``fetch_signals`` (LIVE, unchanged behavior) — walks the public "recent filings" Atom feed for
    form type 4 (newest first), fetches each accession directory's ``index.json`` to locate the
    ownership XML, and stamps ``created_at`` = the transaction date (the live feed has no cheap
    access to the filing's acceptance datetime, and the feed itself is near-real-time, so the
    transaction date has always been an acceptable proxy here).
  * ``fetch_historical`` (BACKFILL, Task 04a of docs/RESEARCH-PLAN.md) — walks EDGAR's quarterly
    "form.idx" indexes for form type 4 filings in a date window, fetches the single full-submission
    ``.txt`` per filing (SGML header + embedded ownership XML in ONE request), and stamps
    ``created_at`` = the filing's ``ACCEPTANCE-DATETIME`` (when it became PUBLIC) — the PIT-correct
    timestamp for research purposes. The underlying transaction date is retained only inside
    ``raw_payload``. This is the key PIT distinction from the live path: a backfilled Form 4 signal
    must never be timestamped by the trade itself, only by its public disclosure.

Every emitted Signal is stamped ``Provenance.REAL`` by this module — it is genuine SEC disclosure
data with no politician dependency. (The backfill USE CASE re-stamps to ``BACKFILLED`` +
``pit_confidence`` regardless; see ``application/backfill_signals.py``.)

CRITICAL: SEC rejects requests without a descriptive ``User-Agent`` (HTTP 403). The injected
``httpx.AsyncClient`` is SHARED across adapters, so we MUST NOT mutate its headers; instead we
pass the UA on every request via the per-request ``headers=`` argument. The historical path
additionally respects SEC's fair-access policy: throttled to <= 10 req/s and retries 429/503 with
bounded exponential backoff.

Endpoints (verified against live EDGAR, June 2026):
  - Recent Form 4 feed (live path):
      https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom
  - Per-filing directory listing (JSON, live path):
      https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/index.json
  - Ownership document (live path): the ``.xml`` item in that directory (not the ``xsl*``
    rendering copy).
  - Quarterly form index (historical path):
      https://www.sec.gov/Archives/edgar/full-index/<year>/QTR<q>/form.idx
  - Full submission text file (historical path, ONE fetch per filing):
      https://www.sec.gov/Archives/<File Name column from form.idx>
  - Ticker -> CIK mapping (historical path, CIK filter):
      https://www.sec.gov/files/company_tickers.json
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import httpx

from trading.application.ports.clock import ClockPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.reference import COMPANY_DIRECTORY

logger = logging.getLogger(__name__)

# Per-source-type reliability prior for corporate-insider trades. Defined HERE (not in the
# data_sources __init__) per the adapter's brief. Form 4 is hard, legally-mandated disclosure
# but is lagged (filed up to two business days after the trade) and indirect, so it sits a touch
# below contract awards yet above noisy statements.
RELIABILITY_INSIDER_TRADE = 0.75

_RECENT_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

_FORM_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
_SUBMISSION_URL = "https://www.sec.gov/Archives/{file_name}"
_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

# EDGAR full-submission SGML header carries the acceptance datetime as a naive
# "YYYYMMDDHHMMSS" local wall-clock value in US/Eastern (EDGAR's filing-agent timezone),
# NOT UTC. We parse it as naive-Eastern and convert via zoneinfo (handles EST/EDT correctly,
# including the DST transition) rather than assuming a fixed UTC offset — a fixed offset would
# be off by an hour for roughly half the year. This is the PIT-correct choice: converting
# properly can only ever make the acceptance instant LATER in UTC on days it under-converts
# (never earlier), so the labeler's strictly-after-date "next session" rule is never violated
# by getting this right.
_EASTERN = ZoneInfo("America/New_York")

# Open-market transaction codes we map to a directional sentiment. Everything else (A=grant/award,
# M=option exercise, G=gift, F=tax withholding, ...) is non-discretionary noise -> skip.
_PURCHASE_CODE = "P"
_SALE_CODE = "S"
_PURCHASE_SENTIMENT = 0.5
_SALE_SENTIMENT = -0.5
_SENTIMENT_BY_CODE = {_PURCHASE_CODE: _PURCHASE_SENTIMENT, _SALE_CODE: _SALE_SENTIMENT}

# SEC fair-access policy: <= 10 requests/second. A simple monotonic sleep-based limiter is
# sufficient for our sequential (non-parallel) walk.
_MAX_REQUESTS_PER_SECOND = 10.0
_MIN_REQUEST_INTERVAL = 1.0 / _MAX_REQUESTS_PER_SECOND
_RETRY_STATUS_CODES = {429, 503}
_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class _ParsedTransaction:
    """One classified open-market transaction from an ownership document, timestamp-agnostic.

    Shared by the live and historical paths: everything that does NOT depend on which timestamp
    the caller wants (transaction date vs. acceptance datetime) lives here, so the classification
    logic (P/S only, A/M/G/F skipped, value-scaled weight) exists in exactly one place.
    """

    ticker: str
    issuer_cik: str
    owner: str
    transaction_index: int  # position within the filing's kept (P/S) rows, for source_uid
    code: str
    sentiment: float
    shares: float
    price: float
    value: float
    txn_date: datetime  # the underlying transaction date (payload only on the historical path)


def _classify_transactions(document: str) -> list[_ParsedTransaction]:
    """Parse an ownership XML document into its classified open-market (P/S) transactions.

    Returns an empty list for malformed XML, a document with no issuer ticker, or a document
    with no qualifying (P/S) rows. Non-discretionary codes (A/M/G/F/...) are skipped entirely —
    they never appear in the result, so callers cannot accidentally emit them.
    """
    try:
        root = ET.fromstring(document)
    except ET.ParseError:
        return []

    ticker = _text(root, "issuer/issuerTradingSymbol").upper()
    if not ticker:
        return []
    owner = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    issuer_cik = _text(root, "issuer/issuerCik")

    parsed: list[_ParsedTransaction] = []
    for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(txn, "transactionCoding/transactionCode")
        sentiment = _SENTIMENT_BY_CODE.get(code)
        if sentiment is None:
            continue  # A / M / G / F / ... -> not a discretionary open-market trade.

        txn_date = _parse_date(_text(txn, "transactionDate/value"))
        if txn_date is None:
            continue

        shares = _to_float(_text(txn, "transactionAmounts/transactionShares/value"))
        price = _to_float(_text(txn, "transactionAmounts/transactionPricePerShare/value"))
        value = shares * price

        parsed.append(
            _ParsedTransaction(
                ticker=ticker,
                issuer_cik=issuer_cik,
                owner=owner,
                transaction_index=len(parsed),
                code=code,
                sentiment=sentiment,
                shares=shares,
                price=price,
                value=value,
                txn_date=txn_date,
            )
        )
    return parsed


class _RateLimiter:
    """SEC fair-access throttle: sleeps as needed to keep requests at or below the ceiling.

    Uses ``time.monotonic`` (never affected by wall-clock changes) and an injectable sleep
    function so tests can run the throttle logic without actually waiting (patch ``sleep`` with a
    no-op / recording stub, or inject a fake ``monotonic``).
    """

    def __init__(
        self,
        *,
        min_interval: float = _MIN_REQUEST_INTERVAL,
        monotonic: Any = time.monotonic,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self._min_interval = min_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request: float | None = None

    async def wait(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            elapsed = now - self._last_request
            remaining = self._min_interval - elapsed
            if remaining > 0:
                await self._sleep(remaining)
                now = self._monotonic()
        self._last_request = now


class SecForm4Source:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        clock: ClockPort,
        user_agent: str,
        lookback_days: int = 3,
        max_filings: int = 100,
        filter_enabled: bool = True,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self._client = client
        self._clock = clock
        self._user_agent = user_agent
        self._lookback_days = lookback_days
        self._max_filings = max_filings
        self._filter_enabled = filter_enabled
        self._limiter = _RateLimiter(sleep=sleep)
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "sec_form4"

    # --- FilterableHistoricalSource capability (application/ports/data_source.py) ---------------

    def set_filter_enabled(self, enabled: bool) -> None:
        self._filter_enabled = enabled

    def filter_description(self) -> str | None:
        if not self._filter_enabled:
            return None
        return "CIK filter: issuers whose tickers are in COMPANY_DIRECTORY"

    # --- Live path (unchanged behavior) ----------------------------------------------------------

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
            for txn in _classify_transactions(document):
                if txn.txn_date <= since:
                    continue
                signals.append(_live_signal(txn, source_url=xml_url))
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

    # --- Historical path (Task 04a, docs/RESEARCH-PLAN.md) ---------------------------------------

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        """Reconstruct historical Form 4 signals disclosed within ``[start, end)``.

        Walks EDGAR's quarterly form.idx indexes covering the window, keeps form-type-4 rows
        (form 4/A amendments are counted and skipped) whose Date Filed falls in the window and
        whose issuer CIK passes the (default-on) filter, fetches the full-submission text for
        each kept row (ONE request per filing — header + ownership XML together), and emits one
        Signal per classified open-market transaction. ``created_at`` = the filing's
        ACCEPTANCE-DATETIME (converted from US/Eastern to UTC) — never the transaction date.
        """
        cik_filter = await self._load_cik_filter() if self._filter_enabled else None

        amendments_skipped = 0
        no_ticker_skipped = 0
        cik_filtered_out = 0
        fetch_failed = 0
        signals: list[Signal] = []

        for year, quarter in _quarters_covering(start, end):
            index_text = await self._get_text(_FORM_INDEX_URL.format(year=year, quarter=quarter))
            for row in _parse_form_index(index_text):
                if row.form_type == "4/A":
                    amendments_skipped += 1
                    continue
                if row.form_type != "4":
                    continue
                if not (start.date() <= row.date_filed <= end.date()):
                    continue
                if cik_filter is not None and row.cik not in cik_filter:
                    cik_filtered_out += 1
                    continue

                submission_url = _SUBMISSION_URL.format(file_name=row.file_name)
                try:
                    submission = await self._get_text(submission_url)
                except httpx.HTTPError:
                    # A single filing's fetch failing (transient 5xx after retries exhausted,
                    # malformed URL, ...) must not abort the whole window's walk — log, count,
                    # continue. Never silently dropped: the counter surfaces in the run log.
                    fetch_failed += 1
                    logger.warning(
                        "sec_form4 historical: failed to fetch filing %s", submission_url
                    )
                    continue

                accepted_at = _parse_acceptance_datetime(submission)
                if accepted_at is None:
                    continue
                accession = _extract_accession(submission) or row.file_name

                document = _extract_ownership_xml(submission)
                if document is None:
                    continue
                transactions = _classify_transactions(document)
                if not transactions:
                    continue
                if not transactions[0].ticker:
                    no_ticker_skipped += 1
                    continue

                for txn in transactions:
                    signals.append(
                        _historical_signal(
                            txn,
                            accession=accession,
                            accepted_at=accepted_at,
                            source_url=submission_url,
                        )
                    )

        logger.info(
            "sec_form4 historical walk: %d signal(s), %d 4/A amendment(s) skipped, "
            "%d row(s) CIK-filtered out, %d filing(s) skipped (no issuer ticker), "
            "%d filing(s) failed to fetch",
            len(signals),
            amendments_skipped,
            cik_filtered_out,
            no_ticker_skipped,
            fetch_failed,
        )
        return signals

    async def _load_cik_filter(self) -> set[str]:
        """The set of zero-padded 10-digit CIKs for tickers in COMPANY_DIRECTORY.

        Fetched once per run from EDGAR's ticker->CIK mapping file (keyed by an opaque numeric
        index in the source JSON, not by CIK or ticker).
        """
        payload: Any = (await self._get_response(_TICKER_CIK_URL)).json()
        wanted = set(COMPANY_DIRECTORY)
        ciks: set[str] = set()
        rows = payload.values() if isinstance(payload, dict) else payload
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).upper()
            if ticker in wanted:
                cik = row.get("cik_str", row.get("cik"))
                if cik is not None:
                    ciks.add(_pad_cik(str(cik)))
        return ciks

    # --- Shared HTTP plumbing (fair access: UA + throttle + retry) -------------------------------

    async def _get_response(self, url: str) -> httpx.Response:
        # Per-request UA only: the client is shared, so we never mutate its headers (a sibling
        # adapter could be mid-flight). SEC 403s any request lacking a descriptive User-Agent.
        for attempt in range(_MAX_RETRIES + 1):
            await self._limiter.wait()
            response = await self._client.get(url, headers={"User-Agent": self._user_agent})
            if response.status_code not in _RETRY_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt == _MAX_RETRIES:
                response.raise_for_status()
            await self._sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
        raise AssertionError("unreachable: loop always returns or raises")

    async def _get_text(self, url: str) -> str:
        return (await self._get_response(url)).text


def _live_signal(txn: _ParsedTransaction, *, source_url: str) -> Signal:
    payload: dict[str, Any] = {
        "ticker": txn.ticker,
        "issuer_cik": txn.issuer_cik,
        "owner": txn.owner,
        "transaction_code": txn.code,
        "transaction_shares": txn.shares,
        "transaction_price_per_share": txn.price,
        "transaction_value": txn.value,
        "transaction_date": txn.txn_date.date().isoformat(),
        "source_url": source_url,
    }
    return Signal(
        id=f"sec-form4-{txn.issuer_cik or txn.ticker}-{txn.txn_date.date().isoformat()}-{txn.code}",
        created_at=txn.txn_date,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=txn.ticker,
        politician_id=None,
        sentiment=txn.sentiment,
        politician_weight=_value_weight(txn.value),
        source_reliability=RELIABILITY_INSIDER_TRADE,
        raw_payload=payload,
        # Real SEC disclosure, no politician dependency -> REAL.
        provenance=Provenance.REAL,
    )


def _historical_signal(
    txn: _ParsedTransaction, *, accession: str, accepted_at: datetime, source_url: str
) -> Signal:
    # source_uid granularity: one filing can yield multiple signals (one per kept transaction),
    # so the dedupe key is accession + a stable per-signal suffix (the transaction's index among
    # the filing's kept P/S rows), not the bare accession. The bare accession is ALSO kept in
    # raw_payload as its own field for traceability / joining filings back together.
    source_uid = f"{accession}:{txn.transaction_index}"
    payload: dict[str, Any] = {
        "ticker": txn.ticker,
        "issuer_cik": txn.issuer_cik,
        "owner": txn.owner,
        "transaction_code": txn.code,
        "transaction_shares": txn.shares,
        "transaction_price_per_share": txn.price,
        "transaction_value": txn.value,
        # PIT trap avoidance: the transaction date is retained here for research/audit, but is
        # NEVER used as created_at on this path — that would leak the pre-disclosure event date.
        "transaction_date": txn.txn_date.date().isoformat(),
        "source_url": source_url,
        "accession": accession,
        "source_uid": source_uid,
    }
    return Signal(
        id=f"sec-form4-hist-{source_uid}",
        created_at=accepted_at,  # the filing's ACCEPTANCE-DATETIME, not the transaction date.
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=txn.ticker,
        politician_id=None,
        sentiment=txn.sentiment,
        politician_weight=_value_weight(txn.value),
        source_reliability=RELIABILITY_INSIDER_TRADE,
        raw_payload=payload,
        # Real SEC disclosure; the backfill use case re-stamps to BACKFILLED regardless (Task 02).
        provenance=Provenance.REAL,
    )


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


@dataclass(frozen=True, slots=True)
class _IndexRow:
    form_type: str
    cik: str  # zero-padded 10-digit, matching company_tickers.json's cik_str once padded
    date_filed: date
    file_name: str


def _quarters_covering(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """The (year, quarter) pairs whose form.idx spans any part of ``[start, end]`` inclusive."""
    quarters: list[tuple[int, int]] = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    end_year, end_quarter = end.year, (end.month - 1) // 3 + 1
    while (year, quarter) <= (end_year, end_quarter):
        quarters.append((year, quarter))
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return quarters


_INDEX_HEADER_PREFIX = "Form Type"
# EDGAR's form.idx columns are NOMINALLY fixed-width, but the header labels' character positions
# do not reliably align with the data columns' actual positions (verified against a live 2024
# QTR1 index: the header's "Date Filed" label starts two characters left of where the date value
# itself starts on every data row), so slicing by header-derived offsets silently truncates the
# date/filename columns. Anchoring on content shape is robust to that drift: form type is the
# first whitespace-delimited token, CIK is a run of digits, date is ISO ``YYYY-MM-DD``, file name
# is the last whitespace-delimited token (a path ending ``.txt``); the company name is a non-
# greedy ``.*?`` filling whatever is between the anchors (spaces and all, trimmed).
_INDEX_ROW_RE = re.compile(
    r"^(?P<form_type>\S+)\s+(?P<company>.*?)\s+(?P<cik>\d+)\s+"
    r"(?P<date_filed>\d{4}-\d{2}-\d{2})\s+(?P<file_name>\S+)\s*$"
)


def _parse_form_index(text: str) -> Iterator[_IndexRow]:
    """Parse an EDGAR ``form.idx`` file (preamble + header + dashed separator + data rows).

    We skip everything up to and including the dashed separator line that follows the
    ``Form Type ...`` header (both present in every EDGAR quarterly index), then parse each data
    row by content shape (see ``_INDEX_ROW_RE``) rather than fixed character offsets.
    """
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(_INDEX_HEADER_PREFIX)), None
    )
    if header_idx is None:
        return
    separator_idx = next(
        (i for i in range(header_idx + 1, len(lines)) if lines[i].startswith("---")), None
    )
    data_start = separator_idx + 1 if separator_idx is not None else header_idx + 1

    for line in lines[data_start:]:
        if not line.strip():
            continue
        match = _INDEX_ROW_RE.match(line)
        if match is None:
            continue
        try:
            date_filed = date.fromisoformat(match.group("date_filed"))
        except ValueError:
            continue
        yield _IndexRow(
            form_type=match.group("form_type"),
            cik=_pad_cik(match.group("cik")),
            date_filed=date_filed,
            file_name=match.group("file_name"),
        )


def _pad_cik(raw: str) -> str:
    digits = raw.strip().lstrip("0") or "0"
    return digits.zfill(10)


def _parse_acceptance_datetime(submission_text: str) -> datetime | None:
    """Extract ``<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS`` from a full-submission SGML header.

    EDGAR records this value as a NAIVE US/Eastern wall-clock timestamp (the filing agent's local
    time, not UTC — SEC's own documentation is silent on the exact zone label, but EDGAR filing
    infrastructure runs on US/Eastern). We attach that zone and convert to UTC via zoneinfo, which
    correctly handles the EST/EDT boundary (a fixed offset would be wrong for roughly half the
    year). This is the single point-in-time timestamp for a historical Form 4 signal.
    """
    marker = "<ACCEPTANCE-DATETIME>"
    idx = submission_text.find(marker)
    if idx == -1:
        return None
    raw = submission_text[idx + len(marker) : idx + len(marker) + 14]
    if len(raw) != 14 or not raw.isdigit():
        return None
    # Constructed WITH tzinfo directly (not via a naive datetime + .replace()) purely to satisfy
    # the DTZ001 lint rule; ZoneInfo's "local wall-clock" semantics are identical either way — the
    # value we hand it (YYYY, MM, DD, hh, mm, ss) is EDGAR's naive Eastern wall-clock reading.
    eastern = datetime(
        int(raw[0:4]),
        int(raw[4:6]),
        int(raw[6:8]),
        int(raw[8:10]),
        int(raw[10:12]),
        int(raw[12:14]),
        tzinfo=_EASTERN,
    )
    return eastern.astimezone(UTC)


_ACCESSION_RE = re.compile(r"^ACCESSION NUMBER:\s*(\S+)", re.MULTILINE)


def _extract_accession(submission_text: str) -> str | None:
    """The dashed accession number (e.g. ``0001437749-24-003683``) from the SGML header's
    ``ACCESSION NUMBER:`` line (a plain key-value line, not an XML-style tag). Falls back to
    ``None`` (the caller then uses the index row's File Name, which encodes the same accession)
    if the header ever omits it."""
    match = _ACCESSION_RE.search(submission_text)
    return match.group(1) if match else None


def _extract_ownership_xml(submission_text: str) -> str | None:
    """Pull the embedded ownership document out of a full-submission ``.txt``'s ``<XML>`` block.

    A full submission may contain multiple ``<DOCUMENT>`` sections (the primary form + exhibits);
    the ownership XML is the one whose content starts with ``<ownershipDocument``. We scan every
    ``<XML>...</XML>`` block and return the first that parses as an ownership document.
    """
    start_marker, end_marker = "<XML>", "</XML>"
    pos = 0
    while True:
        start = submission_text.find(start_marker, pos)
        if start == -1:
            return None
        end = submission_text.find(end_marker, start)
        if end == -1:
            return None
        candidate = submission_text[start + len(start_marker) : end].strip()
        if "<ownershipDocument" in candidate:
            return candidate
        pos = end + len(end_marker)


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
