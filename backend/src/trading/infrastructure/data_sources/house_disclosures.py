"""U.S. House Clerk financial-disclosure adapter (FR-ING-2, FR-POL-6) — STOCK Act PTRs.

Reads Periodic Transaction Reports (PTRs) straight from the official House Clerk financial
disclosure site — free, no API key, real, authoritative.

Two independent fetch paths share the same index/PDF/transaction-parsing core (:func:`_parse_index`,
:func:`_parse_transactions`, :func:`_extract_pdf_text`) and a single timestamp-agnostic signal
builder (:func:`_build_signal`); only the CALLER decides which timestamp and which roster to use:

  * ``fetch_signals`` (LIVE, unchanged behavior) — polls the current year's index for PTRs filed
    after a cursor, resolves the filer against the roster passed to the constructor (current
    members only — unchanged), and stamps ``created_at`` = the index's ``FilingDate`` (midnight
    UTC) — already the PIT-correct timestamp here (the filing date is when the PTR became public;
    the underlying trade date is retained separately in ``raw_payload["transaction_date"]``).
  * ``fetch_historical`` (BACKFILL, Task 04b of docs/RESEARCH-PLAN.md) — walks every requested
    year's index for PTRs whose ``FilingDate`` falls in ``[start, end]``, resolves filers against
    a COMBINED current + historical roster (retired members' PTRs would otherwise silently vanish
    for a deep 2008+ backfill — see ``infrastructure/politicians/congress_legislators.py``), and
    stamps ``created_at`` the same way (index ``FilingDate``, midnight UTC) — the identical PIT
    rule, just applied over a wide window instead of a rolling cursor.

Both paths keep only PTRs (``FilingType == 'P'``) that are e-filed (``DocID`` starts with ``'2'``
and is at least 8 chars) — those PDFs carry an extractable text layer; older/paper filings are
scanned images we cannot parse, so they are skipped (and, on the historical path, COUNTED — the
skip rate is itself research data, RESEARCH-PLAN §2.4). Each kept PTR PDF is opened with
``pdfplumber`` and parsed into transaction rows: ticker (token in parens, if present), asset-type
code (``[ST]`` stock / ``[OP]`` option — both kept), transaction type (P=purchase bullish,
S=sale bearish, E=exchange skipped), transaction date, and dollar-amount range (log-scaled into
the politician weight, modeled on :func:`usaspending._amount_weight`).

**Optional ticker (historical path only).** A kept transaction line's ticker regex can miss (a
foreign private issuer, an unusual asset description, ...). The LIVE path's behavior is
unchanged: such a transaction is simply skipped (as it always was). The HISTORICAL path instead
runs the injected entity resolver over the transaction's raw text block; if that ALSO fails to
resolve a ticker, the signal is still emitted with ``ticker=None`` (RESEARCH-PLAN §2.4 —
survivorship/unresolved entities are research data, never silently dropped) and counted.

Like the congressional disclosed-trade leg, PTRs are lagged up to 45 days, so this is a
corroborating signal, never a primary real-time trigger. The matched member determines
provenance on the live path: a Signal is ``Provenance.REAL`` only if the matched politician is
REAL (a fictional seed roster yields ``SYNTHETIC``), mirroring :mod:`congress_trades`. The
historical path always stamps ``Provenance.REAL`` (real disclosure data; the backfill USE CASE
re-stamps to ``BACKFILLED`` regardless — see ``application/backfill_signals.py``) and skips (and
counts) filings whose member is not in the combined roster.

Throttling (historical path): sequential PDF fetches through the shared
:class:`~trading.infrastructure.data_sources.throttle.RateLimiter` (same helper Task 04a's
``sec_form4`` historical fetcher uses) — no parallel fan-out against the Clerk's servers.

Index:   https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{YEAR}FD.ZIP
PTR PDF: https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{YEAR}/{DocID}.pdf
"""

import asyncio
import io
import logging
import math
import re
import zipfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree

import httpx
import pdfplumber

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.politicians import Politician
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_DISCLOSED_TRADE
from trading.infrastructure.data_sources.throttle import RateLimiter
from trading.infrastructure.data_sources.weighting import politician_weight_for
from trading.infrastructure.politicians.name_matching import resolve_filer

logger = logging.getLogger(__name__)

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
_NAME_RE = re.compile(r"Name:\s*(.+)")
_STATE_DISTRICT_RE = re.compile(r"State/District:\s*([A-Za-z0-9]+)")
_FILING_ID_RE = re.compile(r"Filing ID #?\s*(\d+)")

# The Clerk's CLOSED amount-bracket vocabulary (Task 07, docs/RESEARCH-PLAN.md). Every PTR amount
# cell is one of these ten strings, verbatim (a fixed schema, not free-form dollar figures), so
# each bracket gets its own tolerant pattern rather than one generic "$X - $Y" regex: a PDF's text
# layer can wrap the cell across lines/columns, interleaving the asset-type bracket or other cell
# text between the low and high figures (e.g. "$15,001 -\n[ST] $50,000"), so patterns match
# anywhere in a widened row block with `re.DOTALL` and tolerate arbitrary whitespace/newlines (and
# any other intervening characters) between the two figures, keyed on the two dollar amounts
# themselves being present and correctly ordered — no other bracket's numbers exist in between.
_AMOUNT_BRACKETS: tuple[tuple[float, float], ...] = (
    (1_001, 15_000),
    (15_001, 50_000),
    (50_001, 100_000),
    (100_001, 250_000),
    (250_001, 500_000),
    (500_001, 1_000_000),
    (1_000_001, 5_000_000),
    (5_000_001, 25_000_000),
    (25_000_001, 50_000_000),
)


def _bracket_pattern(low: float, high: float) -> re.Pattern[str]:
    """A tolerant regex for one closed bracket: the exact low/high figures, any whitespace or
    other intervening text (including newlines, from a wrapped PDF layout) between the '$low'
    and '- $high' halves. Non-greedy so it cannot accidentally bridge past a DIFFERENT bracket's
    figures on an adjacent row."""
    low_txt = f"{int(low):,}"
    high_txt = f"{int(high):,}"
    return re.compile(rf"\${low_txt}\s*-\s*(?:\S+\s+)*?\${high_txt}", re.DOTALL)


_BRACKET_PATTERNS: tuple[tuple[re.Pattern[str], float, float], ...] = tuple(
    (_bracket_pattern(low, high), low, high) for low, high in _AMOUNT_BRACKETS
)
# The open-top bracket has no upper figure; tolerate the Clerk's wrapped rendering the same way
# the closed brackets do ("over $50,000,000" possibly split across a line break after "over", or
# with other cell text such as an asset-type bracket interleaved before the dollar figure).
_OPEN_TOP_RE = re.compile(r"over\s*(?:\S+\s+)*?\$50,000,000", re.IGNORECASE | re.DOTALL)
# Sentinel pair for the open-top bracket: distinct from the (0.0, 0.0) parse-miss sentinel and
# from every closed bracket's high figure. `_amount_weight`'s existing log-scale-of-midpoint
# formula already saturates at 2.0x well below this midpoint, so this sentinel does not change
# `_amount_weight`'s semantics (Task 07 scope: parsing only, `_amount_weight` untouched).
_OPEN_TOP_LOW = 50_000_001.0
_OPEN_TOP_HIGH = 100_000_000.0

# Historical-path throttle: polite sequential PDF fetches against the Clerk's servers (same
# ceiling style as Task 04a's SEC fair-access limiter, no formal published rate limit here).
_MAX_REQUESTS_PER_SECOND = 5.0
_MIN_REQUEST_INTERVAL = 1.0 / _MAX_REQUESTS_PER_SECOND


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    """A parsed ``<Member>`` row from the year index that is an e-filed PTR."""

    doc_id: str
    year: int
    filing_date: datetime
    name: str


@dataclass(frozen=True, slots=True)
class _Transaction:
    """One parsed transaction row from a PTR PDF, timestamp-agnostic (shared by both paths).

    ``ticker`` is ``None`` when the regex found no ticker in parens on this line — the LIVE path
    filters these out (unchanged behavior); the HISTORICAL path attempts entity resolution on
    ``raw_block`` and, failing that, emits the signal with ``ticker=None`` (Task 04b).
    """

    ticker: str | None
    raw_block: str
    asset_type: str
    sentiment: float
    transaction_date: datetime
    amount_low: float
    amount_high: float


def _build_signal(
    *,
    entry: _IndexEntry,
    politician: Politician,
    txn: _Transaction,
    ticker: str | None,
    created_at: datetime,
    source_uid: str | None,
    sector_map: SectorMapPort,
    provenance: Provenance,
    row_index: int | None = None,
) -> Signal:
    """Shared signal construction: everything that does NOT depend on which path is calling.

    ``ticker`` is passed explicitly (rather than read off ``txn``) so a historical caller can
    supply an entity-resolver-recovered ticker distinct from the regex match (or ``None`` when
    both fail to resolve one — Task 04b), while ``created_at`` is supplied by the caller so the
    PIT rule (index FilingDate, never the transaction date) lives in exactly one place regardless
    of which path is asking. The politician-weighting lookup needs SOME ticker to key the sector
    map on, so an unresolved ticker weights as if sector-unclassified (``sector_for`` -> None ->
    neutral committee-topic relevance) rather than being skipped.

    ``row_index`` (historical path only) is folded into the Signal ``id``: two same-ticker,
    same-transaction-date rows in ONE PTR would otherwise share an id, and ``rm_signals`` is
    keyed on ``signal_id``, so the second row would silently vanish from the read model during a
    deep backfill — unrecoverable by rebuild, since the collision is baked into the recorded
    events and ``backfill_seen`` blocks re-emission. The live path passes no ``row_index`` and
    its id format is unchanged.
    """
    base_weight = politician_weight_for(politician, ticker or "", sector_map)
    raw_payload: dict[str, Any] = {
        "doc_id": entry.doc_id,
        "year": entry.year,
        "name": entry.name,
        "ticker": ticker,
        "asset_type": txn.asset_type,
        "transaction_date": txn.transaction_date.date().isoformat(),
        "filing_date": entry.filing_date.date().isoformat(),
        "amount_low": txn.amount_low,
        "amount_high": txn.amount_high,
    }
    if source_uid is not None:
        raw_payload["source_uid"] = source_uid
    row_part = "" if row_index is None else f"{row_index}-"
    return Signal(
        id=f"house-{entry.doc_id}-{row_part}"
        f"{ticker or 'unresolved'}-{int(txn.transaction_date.timestamp())}",
        created_at=created_at,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=politician.id,
        sentiment=txn.sentiment,
        politician_weight=base_weight * _amount_weight(txn.amount_low, txn.amount_high),
        source_reliability=RELIABILITY_DISCLOSED_TRADE,
        raw_payload=raw_payload,
        provenance=provenance,
    )


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
        entity_resolver: EntityResolverPort | None = None,
        historical_politicians: Callable[[], Awaitable[PoliticianRepositoryPort]] | None = None,
        sleep: Any = None,
    ) -> None:
        self._client = client
        self._politicians = politicians
        self._sector_map = sector_map
        self._clock = clock
        self._lookback_days = lookback_days
        self._max_filings = max_filings
        self._years = years
        # --- Historical-path-only collaborators (Task 04b); unused by fetch_signals -----------
        self._entity_resolver = entity_resolver
        self._historical_politicians_loader = historical_politicians
        self._historical_politicians_cache: PoliticianRepositoryPort | None = None
        self._limiter = RateLimiter(
            min_interval=_MIN_REQUEST_INTERVAL, sleep=sleep or asyncio.sleep
        )
        # Diagnostic counters from the most recent fetch_historical() call (Task 07,
        # docs/RESEARCH-PLAN.md), exposed via last_run_summary()
        # (SummarizingHistoricalSource, application/ports/data_source.py).
        self._last_run_summary: dict[str, object] | None = None

    @property
    def name(self) -> str:
        return "house_disclosures"

    def last_run_summary(self) -> Mapping[str, object] | None:
        """Diagnostic counters from the most recent ``fetch_historical`` call (Task 07):
        ``unmatched_filers`` (by reason: ``no_candidate``/``ambiguous``) and
        ``amount_parse_misses``. ``None`` until ``fetch_historical`` has run at least once on
        this instance. Implements
        :class:`~trading.application.ports.data_source.SummarizingHistoricalSource`.
        """
        return self._last_run_summary

    # --- Live path (unchanged behavior) -----------------------------------------------------

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        entries = await self._fresh_entries(since)
        # Process the newest filings first and cap per poll to stay courteous (FR-ING-4).
        entries.sort(key=lambda entry: entry.filing_date, reverse=True)
        entries = entries[: self._max_filings]

        signals: list[Signal] = []
        for entry in entries:
            politician, _reason = resolve_filer(entry.name, self._politicians.all())
            if politician is None:
                # Member not in the roster (or ambiguous) -> we cannot weight the trade, so skip
                # the filing. The live path has no per-run summary to attribute a reason to.
                continue
            text = await self._fetch_ptr_text(entry)
            if text is None:
                continue
            for txn in _parse_transactions(text, doc_id=entry.doc_id):
                if txn.ticker is None:
                    # Unchanged live behavior: a transaction line with no regex-matched ticker
                    # is skipped (no entity-resolver fallback on the live path).
                    continue
                signals.append(
                    _build_signal(
                        entry=entry,
                        politician=politician,
                        txn=txn,
                        ticker=txn.ticker,
                        created_at=entry.filing_date,
                        source_uid=None,
                        sector_map=self._sector_map,
                        # REAL only if the matched politician is real (fictional roster ->
                        # SYNTHETIC), mirroring congress_trades.
                        provenance=(
                            Provenance.REAL
                            if politician.provenance is Provenance.REAL
                            else Provenance.SYNTHETIC
                        ),
                    )
                )
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

    # --- Historical path (Task 04b, docs/RESEARCH-PLAN.md) -----------------------------------

    async def fetch_historical(self, start: datetime, end: datetime) -> Sequence[Signal]:
        """Reconstruct historical House PTR signals filed within ``[start, end]`` (inclusive).

        Walks every year's index the window spans, keeps e-filed PTRs whose ``FilingDate`` falls
        in the window, resolves each filer against the COMBINED current + historical roster
        (lazily fetched once and cached on this instance — see ``_politician_repository``), and
        fetches/parses each kept filing's PDF SEQUENTIALLY (throttled) into one Signal per
        transaction line. ``created_at`` = the index's ``FilingDate`` (midnight UTC) — the moment
        the PTR became public, never the underlying trade date (retained in
        ``raw_payload["transaction_date"]``). ``source_uid`` = ``"{DocID}:{row_index}"``.
        """
        politicians = await self._politician_repository()
        roster = politicians.all()

        paper_filings_skipped = 0
        unmatched_filer_skipped = 0
        unmatched_filers_no_candidate = 0
        unmatched_filers_ambiguous = 0
        unresolved_ticker = 0
        unparseable_pdfs = 0
        amount_parse_misses = 0
        signals: list[Signal] = []

        for year in _years_covering(start, end):
            xml_bytes = await self._fetch_index_xml_throttled(year)
            if xml_bytes is None:
                continue
            paper_filings_skipped += _count_paper_ptrs(xml_bytes)
            for entry in _parse_index(xml_bytes):
                if not (start <= entry.filing_date <= end):
                    continue
                politician, reason = resolve_filer(entry.name, roster)
                if politician is None:
                    unmatched_filer_skipped += 1
                    if reason == "ambiguous":
                        unmatched_filers_ambiguous += 1
                    else:
                        unmatched_filers_no_candidate += 1
                    logger.info(
                        "house_disclosures historical: filer not in roster, skipped "
                        "(doc_id=%s, reason=%s)",
                        entry.doc_id,
                        reason,
                    )
                    continue

                text = await self._fetch_ptr_text_throttled(entry)
                if text is None:
                    unparseable_pdfs += 1
                    logger.warning(
                        "house_disclosures historical: unparseable/unfetchable PDF, "
                        "skipped (doc_id=%s)",
                        entry.doc_id,
                    )
                    continue

                for row_index, txn in enumerate(_parse_transactions(text, doc_id=entry.doc_id)):
                    if txn.amount_low == 0.0 and txn.amount_high == 0.0:
                        amount_parse_misses += 1
                    ticker = txn.ticker or self._resolve_ticker(txn.raw_block)
                    if ticker is None:
                        unresolved_ticker += 1
                        logger.info(
                            "house_disclosures historical: unresolved ticker, "
                            "signal emitted with ticker=None (doc_id=%s, row=%d)",
                            entry.doc_id,
                            row_index,
                        )
                    signals.append(
                        _build_signal(
                            entry=entry,
                            politician=politician,
                            txn=txn,
                            ticker=ticker,
                            created_at=entry.filing_date,
                            source_uid=f"{entry.doc_id}:{row_index}",
                            sector_map=self._sector_map,
                            # Real disclosure data; the backfill use case re-stamps to
                            # BACKFILLED regardless (Task 02) — this adapter never pre-empts it.
                            provenance=Provenance.REAL,
                            row_index=row_index,
                        )
                    )

        self._last_run_summary = {
            "signals": len(signals),
            "paper_filings_skipped": paper_filings_skipped,
            "unmatched_filers": {
                "no_candidate": unmatched_filers_no_candidate,
                "ambiguous": unmatched_filers_ambiguous,
            },
            "unresolved_ticker": unresolved_ticker,
            "unparseable_pdfs": unparseable_pdfs,
            "amount_parse_misses": amount_parse_misses,
        }
        logger.info(
            "house_disclosures historical walk: %d signal(s), %d paper filing(s) skipped, "
            "%d filer(s) not in roster (%d no_candidate, %d ambiguous), %d unresolved "
            "ticker(s), %d unparseable PDF(s), %d amount-bracket parse miss(es)",
            len(signals),
            paper_filings_skipped,
            unmatched_filer_skipped,
            unmatched_filers_no_candidate,
            unmatched_filers_ambiguous,
            unresolved_ticker,
            unparseable_pdfs,
            amount_parse_misses,
        )
        return signals

    def _resolve_ticker(self, raw_block: str) -> str | None:
        """Entity-resolver fallback when the ticker regex misses (historical path only)."""
        if self._entity_resolver is None:
            return None
        candidates = self._entity_resolver.resolve_tickers(raw_block)
        return candidates[0] if candidates else None

    async def _politician_repository(self) -> PoliticianRepositoryPort:
        """The combined current+historical roster, fetched lazily on first use and cached.

        CRITICAL startup constraint (Task 04b): constructing this source must never touch the
        network — the loader callable is only invoked here, on the first ``fetch_historical``
        call, never at import/construction time. Falls back to the LIVE roster (current members
        only) if no historical loader was wired, so the source still degrades gracefully rather
        than crashing.
        """
        if self._historical_politicians_cache is not None:
            return self._historical_politicians_cache
        if self._historical_politicians_loader is None:
            return self._politicians
        repo = await self._historical_politicians_loader()
        self._historical_politicians_cache = repo
        return repo

    async def _fetch_index_xml_throttled(self, year: int) -> bytes | None:
        await self._limiter.wait()
        return await self._fetch_index_xml(year)

    async def _fetch_ptr_text_throttled(self, entry: _IndexEntry) -> str | None:
        await self._limiter.wait()
        return await self._fetch_ptr_text(entry)


def _years_covering(start: datetime, end: datetime) -> tuple[int, ...]:
    """Every calendar year the ``[start, end]`` window spans (inclusive)."""
    return tuple(range(start.year, end.year + 1))


def _count_paper_ptrs(xml_bytes: bytes) -> int:
    """Count PTR filings in the index that are scanned/paper (not e-filed) — research data."""
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return 0
    count = 0
    for member in root.findall("Member"):
        if _text(member, "FilingType") != _PTR_FILING_TYPE:
            continue
        if not _is_efiled(_text(member, "DocID")):
            count += 1
    return count


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


# How many lines before/after the asset-type-bracket line to include in the AMOUNT window (Task
# 07, Scope B). Ticker/sentiment/date parsing keep using the narrower 2-line `block` (unchanged,
# still correct for those fields); only the amount-bracket search widens, since wide/wrapped PDF
# layouts can push the low or high figure a line further away than the immediate predecessor.
_AMOUNT_WINDOW_BEFORE = 2
_AMOUNT_WINDOW_AFTER = 1


def _parse_transactions(text: str, *, doc_id: str = "") -> list[_Transaction]:
    """Parse all kept (stock/option, purchase/sale) transactions out of a PTR text layer.

    Each transaction spans the line carrying its asset-type bracket (``[ST]``/``[OP]``); the
    ticker may sit on that line or the line above it, so we join each bracket line with its
    predecessor before parsing (unchanged). A transaction is returned even when no ticker
    regex-matches (``ticker=None``) — the LIVE caller filters these out; the HISTORICAL caller
    attempts entity resolution and, failing that, still emits the signal (Task 04b).

    ``doc_id`` (historical path only, best-effort) is passed through purely so an amount-parse
    miss can be logged with the filing it came from (Task 07, Scope B); the live path does not
    supply it and simply logs without a DocID.
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
        amount_window = "\n".join(
            lines[max(0, index - _AMOUNT_WINDOW_BEFORE) : index + _AMOUNT_WINDOW_AFTER + 1]
        )
        txn = _parse_transaction_block(
            block, asset_type, amount_window=amount_window, doc_id=doc_id
        )
        if txn is not None:
            transactions.append(txn)
    return transactions


def _parse_transaction_block(
    block: str, asset_type: str, *, amount_window: str, doc_id: str
) -> _Transaction | None:
    sentiment = _transaction_sentiment(block)
    if sentiment == 0.0:
        return None

    transaction_date = _first_date(block)
    if transaction_date is None:
        return None

    ticker_match = _TICKER_RE.search(block)
    ticker = ticker_match.group(1).upper() if ticker_match else None

    amount_low, amount_high = _parse_amount_range(amount_window, doc_id=doc_id)
    return _Transaction(
        ticker=ticker,
        raw_block=block,
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


def _parse_amount_range(block: str, *, doc_id: str = "") -> tuple[float, float]:
    """Extract the Clerk's amount bracket from ``block`` (Task 07, Scope B).

    Tries each closed bracket's tolerant pattern (see ``_BRACKET_PATTERNS``) against the FULL
    ``block`` text, then the open-top bracket. ``(0.0, 0.0)`` is kept STRICTLY as the parse-miss
    sentinel — no closed or open-top bracket matched — and is logged with ``doc_id`` (when the
    caller has one) so misses are visible per run, never silent.
    """
    for pattern, low, high in _BRACKET_PATTERNS:
        if pattern.search(block):
            return (low, high)
    if _OPEN_TOP_RE.search(block):
        return (_OPEN_TOP_LOW, _OPEN_TOP_HIGH)
    logger.info(
        "house_disclosures: amount-bracket parse miss (doc_id=%s)",
        doc_id or "unknown",
    )
    return (0.0, 0.0)


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
