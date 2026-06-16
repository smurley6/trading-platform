"""U.S. House Clerk financial-disclosure adapter (FR-ING-2, FR-POL-6) — STOCK Act PTRs.

Reads Periodic Transaction Reports (PTRs) straight from the official House Clerk financial
disclosure site — free, no API key, real, authoritative. Each poll:

1. Downloads the per-year index zip (``{YEAR}FD.ZIP`` containing ``{YEAR}FD.xml``) and parses
   the ``<Member>`` records. We keep only PTRs (``FilingType == 'P'``) that are e-filed
   (``DocID`` starts with ``'2'`` and is at least 8 chars) — those PDFs carry an extractable
   text layer. Older/paper filings are scanned images we cannot parse, so they are skipped.
2. Fetches the newest PTR PDFs first (sorted by ``FilingDate`` descending), capped at
   ``max_filings`` per poll to stay courteous to the Clerk's servers (FR-ING-4).
3. Opens each PDF with ``pdfplumber`` on an in-memory ``BytesIO`` and parses the transaction
   rows: ticker (token in parens), asset-type code (``[ST]`` stock / ``[OP]`` option — both
   kept), transaction type (P=purchase bullish, S=sale bearish, E=exchange skipped),
   transaction date and dollar-amount range. The amount-range midpoint scales the politician
   weight on a log scale (modeled on :func:`usaspending._amount_weight`).

Like the congressional disclosed-trade leg, PTRs are lagged up to 45 days, so this is a
corroborating signal, never a primary real-time trigger. The matched member determines
provenance: a Signal is ``Provenance.REAL`` only if the matched politician is REAL (a fictional
seed roster yields ``SYNTHETIC``), mirroring :mod:`congress_trades`. Filings whose member is not
in the roster are skipped entirely.

Only filings whose ``FilingDate`` is strictly after the ``since`` cursor are emitted.

Index:   https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{YEAR}FD.ZIP
PTR PDF: https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf
"""

import io
import math
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree

import httpx
import pdfplumber

from trading.application.ports.clock import ClockPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.politicians import Politician
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_DISCLOSED_TRADE
from trading.infrastructure.data_sources.weighting import politician_weight_for

_INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
_PTR_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"

# FilingType 'P' is a Periodic Transaction Report; e-filed PTRs (DocID starting '2', length >= 8)
# have an extractable text layer. Paper/older filings are scanned images we cannot parse.
_PTR_FILING_TYPE = "P"
_EFILED_DOC_ID_PREFIX = "2"
_EFILED_DOC_ID_MIN_LEN = 8

# Purchases read bullish, sales bearish; magnitude is modest since the leg is lagged.
_BUY_SENTIMENT = 0.5
_SELL_SENTIMENT = -0.5

# Asset-type codes we keep: stock and options (both are tradeable equity exposure). Anything
# else (e.g. bonds, mutual funds) is ignored.
_KEPT_ASSET_TYPES = frozenset({"ST", "OP"})

# A transaction line carries a ticker in parens, an asset-type code in brackets, a single-letter
# transaction type standing alone before the first date, an MM/DD/YYYY date and a dollar range.
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]*)\)")
_ASSET_TYPE_RE = re.compile(r"\[([A-Z]{2})\]")
_TXN_TYPE_RE = re.compile(r"(?<![A-Za-z])([PSE])(?![A-Za-z])")
_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_AMOUNT_RE = re.compile(r"\$([\d,]+)\s*-\s*\$([\d,]+)")
_NAME_RE = re.compile(r"Name:\s*(.+)")
_STATE_DISTRICT_RE = re.compile(r"State/District:\s*([A-Za-z0-9]+)")
_FILING_ID_RE = re.compile(r"Filing ID #?\s*(\d+)")


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    """A parsed ``<Member>`` row from the year index that is an e-filed PTR."""

    doc_id: str
    year: int
    filing_date: datetime
    name: str


@dataclass(frozen=True, slots=True)
class _Transaction:
    """One parsed transaction row from a PTR PDF."""

    ticker: str
    asset_type: str
    sentiment: float
    transaction_date: datetime
    amount_low: float
    amount_high: float


class HouseDisclosuresSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        politicians: PoliticianRepositoryPort,
        sector_map: SectorMapPort,
        clock: ClockPort,
        lookback_days: int = 30,
        max_filings: int = 50,
        years: tuple[int, ...] | None = None,
    ) -> None:
        self._client = client
        self._politicians = politicians
        self._sector_map = sector_map
        self._clock = clock
        self._lookback_days = lookback_days
        self._max_filings = max_filings
        self._years = years

    @property
    def name(self) -> str:
        return "house_disclosures"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        entries = await self._fresh_entries(since)
        # Process the newest filings first and cap per poll to stay courteous (FR-ING-4).
        entries.sort(key=lambda entry: entry.filing_date, reverse=True)
        entries = entries[: self._max_filings]

        signals: list[Signal] = []
        for entry in entries:
            politician = self._politicians.find_in_text(entry.name)
            if politician is None:
                # Member not in the roster -> we cannot weight the trade, so skip the filing.
                continue
            text = await self._fetch_ptr_text(entry)
            if text is None:
                continue
            for txn in _parse_transactions(text):
                signals.append(self._to_signal(entry, politician, txn))
        return signals

    async def _fresh_entries(self, since: datetime) -> list[_IndexEntry]:
        """Parse every requested year index, keeping e-filed PTRs filed after ``since``."""
        entries: list[_IndexEntry] = []
        for year in self._target_years():
            xml_bytes = await self._fetch_index_xml(year)
            if xml_bytes is None:
                continue
            for entry in _parse_index(xml_bytes):
                if entry.filing_date > since:
                    entries.append(entry)
        return entries

    def _target_years(self) -> tuple[int, ...]:
        if self._years is not None:
            return self._years
        now = self._clock.now()
        # Early in the year, last year's PTRs are still being filed, so include it too.
        if now.month <= 3:
            return (now.year, now.year - 1)
        return (now.year,)

    async def _fetch_index_xml(self, year: int) -> bytes | None:
        """Download ``{year}FD.ZIP`` and return the contained ``{year}FD.xml`` bytes."""
        response = await self._client.get(_INDEX_URL.format(year=year))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            member = f"{year}FD.xml"
            names = archive.namelist()
            if member in names:
                target: str | None = member
            else:
                target = next((name for name in names if name.endswith(".xml")), None)
            if target is None:
                return None
            return archive.read(target)

    async def _fetch_ptr_text(self, entry: _IndexEntry) -> str | None:
        """Download a PTR PDF and return its concatenated text layer (or None on failure)."""
        url = _PTR_URL.format(year=entry.year, doc_id=entry.doc_id)
        response = await self._client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return _extract_pdf_text(response.content)

    def _to_signal(self, entry: _IndexEntry, politician: Politician, txn: _Transaction) -> Signal:
        # The FilingDate is this lagged leg's available timestamp (when the public could act).
        created_at = entry.filing_date
        base_weight = politician_weight_for(politician, txn.ticker, self._sector_map)
        raw_payload: dict[str, Any] = {
            "doc_id": entry.doc_id,
            "year": entry.year,
            "name": entry.name,
            "ticker": txn.ticker,
            "asset_type": txn.asset_type,
            "transaction_date": txn.transaction_date.date().isoformat(),
            "filing_date": entry.filing_date.date().isoformat(),
            "amount_low": txn.amount_low,
            "amount_high": txn.amount_high,
        }
        return Signal(
            id=f"house-{entry.doc_id}-{txn.ticker}-{int(txn.transaction_date.timestamp())}",
            created_at=created_at,
            source_type=SignalSource.DISCLOSED_TRADE,
            ticker=txn.ticker,
            politician_id=politician.id,
            sentiment=txn.sentiment,
            politician_weight=base_weight * _amount_weight(txn.amount_low, txn.amount_high),
            source_reliability=RELIABILITY_DISCLOSED_TRADE,
            raw_payload=raw_payload,
            # REAL only if the matched politician is real (fictional roster -> SYNTHETIC).
            provenance=(
                Provenance.REAL
                if politician.provenance is Provenance.REAL
                else Provenance.SYNTHETIC
            ),
        )


def _parse_index(xml_bytes: bytes) -> list[_IndexEntry]:
    """Parse the year index XML into the e-filed PTR entries worth fetching."""
    root = ElementTree.fromstring(xml_bytes)
    entries: list[_IndexEntry] = []
    for member in root.findall("Member"):
        if _text(member, "FilingType") != _PTR_FILING_TYPE:
            continue
        doc_id = _text(member, "DocID")
        if not _is_efiled(doc_id):
            continue
        filing_date = _parse_us_date(_text(member, "FilingDate"))
        if filing_date is None:
            continue
        name = _member_name(member)
        if not name:
            continue
        year = _to_int(_text(member, "Year")) or filing_date.year
        entries.append(_IndexEntry(doc_id=doc_id, year=year, filing_date=filing_date, name=name))
    return entries


def _is_efiled(doc_id: str) -> bool:
    return doc_id.startswith(_EFILED_DOC_ID_PREFIX) and len(doc_id) >= _EFILED_DOC_ID_MIN_LEN


def _member_name(member: ElementTree.Element) -> str:
    parts = (
        _text(member, "Prefix"),
        _text(member, "First"),
        _text(member, "Last"),
        _text(member, "Suffix"),
    )
    return " ".join(part for part in parts if part)


def _text(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _extract_pdf_text(pdf_bytes: bytes) -> str | None:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception:
        # A malformed or image-only PDF must not crash the poll; skip it.
        return None
    text = "\n".join(pages).strip()
    return text or None


def _parse_transactions(text: str) -> list[_Transaction]:
    """Parse all kept (stock/option, purchase/sale) transactions out of a PTR text layer.

    Each transaction spans the line carrying its asset-type bracket (``[ST]``/``[OP]``); the
    ticker may sit on that line or the line above it, so we join each bracket line with its
    predecessor before parsing.
    """
    lines = text.splitlines()
    transactions: list[_Transaction] = []
    for index, line in enumerate(lines):
        asset_match = _ASSET_TYPE_RE.search(line)
        if asset_match is None:
            continue
        asset_type = asset_match.group(1)
        if asset_type not in _KEPT_ASSET_TYPES:
            continue
        block = f"{lines[index - 1]} {line}" if index > 0 else line
        txn = _parse_transaction_block(block, asset_type)
        if txn is not None:
            transactions.append(txn)
    return transactions


def _parse_transaction_block(block: str, asset_type: str) -> _Transaction | None:
    ticker_match = _TICKER_RE.search(block)
    if ticker_match is None:
        return None
    ticker = ticker_match.group(1).upper()

    sentiment = _transaction_sentiment(block)
    if sentiment == 0.0:
        return None

    transaction_date = _first_date(block)
    if transaction_date is None:
        return None

    amount_low, amount_high = _parse_amount_range(block)
    return _Transaction(
        ticker=ticker,
        asset_type=asset_type,
        sentiment=sentiment,
        transaction_date=transaction_date,
        amount_low=amount_low,
        amount_high=amount_high,
    )


def _transaction_sentiment(block: str) -> float:
    """Map the standalone single-letter transaction code to a sentiment (E=exchange -> skip)."""
    match = _TXN_TYPE_RE.search(block)
    if match is None:
        return 0.0
    code = match.group(1)
    if code == "P":
        return _BUY_SENTIMENT
    if code == "S":
        return _SELL_SENTIMENT
    return 0.0


def _first_date(block: str) -> datetime | None:
    match = _DATE_RE.search(block)
    if match is None:
        return None
    month, day, year = (int(group) for group in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _parse_amount_range(block: str) -> tuple[float, float]:
    match = _AMOUNT_RE.search(block)
    if match is None:
        return (0.0, 0.0)
    low = _to_float(match.group(1).replace(",", ""))
    high = _to_float(match.group(2).replace(",", ""))
    return (low, high)


def _amount_weight(amount_low: float, amount_high: float) -> float:
    """Map a transaction's dollar-range midpoint to a multiplier in roughly [1.0, 2.0].

    Modeled on :func:`usaspending._amount_weight`: a log-scale bump so a $1M trade reads more
    convincing than a $1K one, saturating so a single huge trade cannot dominate.
    """
    midpoint = (amount_low + amount_high) / 2.0
    if midpoint <= 0:
        return 1.0
    return min(2.0, 1.0 + max(0.0, math.log10(midpoint) - 3.0) / 3.0)


def _parse_us_date(value: str) -> datetime | None:
    """Parse an M/D/YYYY FilingDate into a UTC datetime."""
    match = _DATE_RE.fullmatch(value.strip())
    if match is None:
        return None
    month, day, year = (int(group) for group in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
