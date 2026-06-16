"""U.S. Senate Electronic Financial Disclosures (eFD) adapter (FR-ING-2, FR-POL-6).

Pulls Senators' **Periodic Transaction Reports** (PTRs) straight from the authoritative,
free government portal at https://efdsearch.senate.gov/search/ and emits one disclosed-trade
:class:`~trading.domain.signals.Signal` per transaction. This is the Senate analogue of the
House :class:`~trading.infrastructure.data_sources.congress_trades.CongressTradesSource`: the
STOCK Act lags filings up to 45 days, so this leg is corroboration only, never a primary
real-time trigger.

eFD has no JSON API; it is a Django site guarded by a CSRF token and a one-time "prohibition
agreement" acceptance. The handshake (run once per ``fetch_signals`` using the injected client)
is:

1. ``GET /search/`` — Django sets a ``csrftoken`` cookie; that token is also the value POSTed
   back as ``csrfmiddlewaretoken``.
2. ``POST /search/home/`` with ``prohibition_agreement=1`` + ``csrfmiddlewaretoken`` — accepts
   the agreement and establishes the session.
3. ``POST /search/report/data/`` with DataTables paging params (``draw``/``start``/``length``)
   plus ``report_types=[11]`` (PTR) and a submitted-date range — returns a JSON ``data`` array.
   Each row's filer link points either at an **electronic** report detail page (an HTML
   transactions table we parse) or at a **paper** report (a scanned PDF, which we SKIP).
4. ``GET <report detail URL>`` for each electronic PTR — yields the transactions table whose
   rows give us ticker, transaction type, transaction date, and amount range.

Each parsed transaction becomes a Signal under the SAME rules as the House adapter: ticker
upper-cased, the filing Senator resolved via ``politicians.find_in_text`` (skip if unmatched),
``provenance`` REAL only when the matched politician is REAL (otherwise SYNTHETIC, fail-closed),
Purchase ``+0.5`` / Sale ``-0.5`` sentiment (Exchange skipped), ``politician_weight`` from
``politician_weight_for`` scaled by the amount range, and the per-source reliability prior.

No commercial use of this free government data (spec section 2.2 non-goals).
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.politicians import Politician
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_DISCLOSED_TRADE
from trading.infrastructure.data_sources.weighting import politician_weight_for

# Portal roots. The handshake and search endpoints are all under this host.
_BASE_URL = "https://efdsearch.senate.gov"
_SEARCH_URL = f"{_BASE_URL}/search/"
_HOME_URL = f"{_BASE_URL}/search/home/"
_REPORT_DATA_URL = f"{_BASE_URL}/search/report/data/"

# eFD report-type code for Periodic Transaction Reports (the stock-trade filings we want).
_PTR_REPORT_TYPE = "[11]"
# Filer-type code for Senators (the eFD UI's "Senator" checkbox).
_SENATOR_FILER_TYPE = "[1]"

# Purchases read bullish, sales bearish; magnitude is modest since the leg is lagged. Exchanges
# (one security swapped for another) carry no clean directional read, so we skip them.
_BUY_SENTIMENT = 0.5
_SELL_SENTIMENT = -0.5

# Some Senators file the agreement-accept step under a session cookie; a browser-like UA and a
# Referer keep eFD's bot-mitigation from short-circuiting the handshake.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": _SEARCH_URL,
}

# Amount-range upper bounds (in USD) for the eFD disclosure brackets, ascending. The bracket a
# transaction falls in scales the politician weight on a log-ish curve (mirrors USAspending's
# _amount_weight: a small odd-lot buy should not weigh like a seven-figure position).
_AMOUNT_BRACKETS: tuple[tuple[int, float], ...] = (
    (1_000, 1.0),
    (15_000, 1.1),
    (50_000, 1.2),
    (100_000, 1.3),
    (250_000, 1.4),
    (500_000, 1.5),
    (1_000_000, 1.6),
    (5_000_000, 1.8),
    (25_000_000, 1.9),
    (50_000_000, 2.0),
)


class SenateEfdSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        politicians: PoliticianRepositoryPort,
        sector_map: SectorMapPort,
        clock: ClockPort,
        lookback_days: int = 30,
        max_filings: int = 50,
    ) -> None:
        self._client = client
        self._politicians = politicians
        self._sector_map = sector_map
        self._clock = clock
        self._lookback_days = lookback_days
        self._max_filings = max_filings

    @property
    def name(self) -> str:
        return "senate_efd"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        token = await self._handshake()
        rows = await self._search_ptr_rows(token)

        signals: list[Signal] = []
        for row in rows[: self._max_filings]:
            filing = _parse_search_row(row)
            # Paper (scanned PDF) PTRs have no machine-readable transactions table -> skip; only
            # electronic reports expose the HTML table we can parse.
            if filing is None or not filing.is_electronic:
                continue
            politician = self._politicians.find_in_text(filing.filer_name)
            if politician is None:
                continue
            detail_html = await self._fetch_detail(filing.detail_url, token)
            for txn in _parse_transactions(detail_html):
                signal = self._to_signal(txn, filing, politician, since)
                if signal is not None:
                    signals.append(signal)
        return signals

    async def _handshake(self) -> str:
        """Run the GET-search -> POST-agreement dance and return the CSRF token."""
        await self._client.get(_SEARCH_URL, headers=_HEADERS)
        token = self._csrf_token()
        await self._client.post(
            _HOME_URL,
            data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token},
            headers={**_HEADERS, "X-CSRFToken": token},
        )
        # The accept POST may rotate the token; re-read so subsequent POSTs carry the live value.
        return self._csrf_token() or token

    def _csrf_token(self) -> str:
        return self._client.cookies.get("csrftoken") or ""

    async def _search_ptr_rows(self, token: str) -> list[list[Any]]:
        response = await self._client.post(
            _REPORT_DATA_URL,
            data=self._search_form(token),
            headers={**_HEADERS, "X-CSRFToken": token},
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        data = body.get("data", [])
        return [row for row in data if isinstance(row, list)]

    def _search_form(self, token: str) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).strftime(
            "%m/%d/%Y"
        )
        end = self._clock.now().date().strftime("%m/%d/%Y")
        return {
            "draw": "1",
            "start": "0",
            "length": str(self._max_filings),
            "report_types": _PTR_REPORT_TYPE,
            "filer_types": _SENATOR_FILER_TYPE,
            "submitted_start_date": f"{start} 00:00:00",
            "submitted_end_date": f"{end} 23:59:59",
            "search_field": "",
            "first_name": "",
            "last_name": "",
            "csrfmiddlewaretoken": token,
        }

    async def _fetch_detail(self, detail_url: str, token: str) -> str:
        response = await self._client.get(detail_url, headers={**_HEADERS, "X-CSRFToken": token})
        response.raise_for_status()
        return response.text

    def _to_signal(
        self,
        txn: "_Transaction",
        filing: "_Filing",
        politician: Politician,
        since: datetime,
    ) -> Signal | None:
        ticker = txn.ticker.upper()
        if not ticker:
            return None
        sentiment = _direction_sentiment(txn.transaction_type)
        if sentiment == 0.0:
            return None
        created_at = txn.transaction_date or filing.received or self._clock.now()
        if created_at <= since:
            return None
        weight = politician_weight_for(politician, ticker, self._sector_map)
        payload: dict[str, Any] = {
            "filer": filing.filer_name,
            "report_url": filing.detail_url,
            "ticker": ticker,
            "asset_name": txn.asset_name,
            "transaction_type": txn.transaction_type,
            "transaction_date": txn.transaction_date_raw,
            "amount": txn.amount_raw,
        }
        return Signal(
            id=f"senate-efd-{politician.id}-{ticker}-{int(created_at.timestamp())}",
            created_at=created_at,
            source_type=SignalSource.DISCLOSED_TRADE,
            ticker=ticker,
            politician_id=politician.id,
            sentiment=sentiment,
            politician_weight=weight * _amount_multiplier(txn.amount_raw),
            source_reliability=RELIABILITY_DISCLOSED_TRADE,
            raw_payload=payload,
            # REAL only if the matched senator is real (synthetic seed roster -> SYNTHETIC).
            provenance=(
                Provenance.REAL
                if politician.provenance is Provenance.REAL
                else Provenance.SYNTHETIC
            ),
        )


class _Filing:
    """One PTR row from the search results: who filed, where the detail lives, and its kind."""

    __slots__ = ("detail_url", "filer_name", "is_electronic", "received")

    def __init__(
        self, *, filer_name: str, detail_url: str, is_electronic: bool, received: datetime | None
    ) -> None:
        self.filer_name = filer_name
        self.detail_url = detail_url
        self.is_electronic = is_electronic
        self.received = received


class _Transaction:
    """One parsed row from a PTR's HTML transactions table."""

    __slots__ = (
        "amount_raw",
        "asset_name",
        "ticker",
        "transaction_date",
        "transaction_date_raw",
        "transaction_type",
    )

    def __init__(
        self,
        *,
        ticker: str,
        asset_name: str,
        transaction_type: str,
        transaction_date: datetime | None,
        transaction_date_raw: str,
        amount_raw: str,
    ) -> None:
        self.ticker = ticker
        self.asset_name = asset_name
        self.transaction_type = transaction_type
        self.transaction_date = transaction_date
        self.transaction_date_raw = transaction_date_raw
        self.amount_raw = amount_raw


def _parse_search_row(row: list[Any]) -> _Filing | None:
    """Normalize a DataTables search row.

    eFD returns each filing as a list whose cells are HTML fragments. The canonical column order
    is [first_name, last_name, filer_link_html, report_type_link_html, date_received]. The filer
    link carries the report detail href; the report-type link's path (``/search/view/ptr/...``
    for electronic vs ``/search/view/paper/...`` for scanned) tells us whether it is parseable.
    """
    if len(row) < 5:
        return None
    first = _strip_tags(str(row[0]))
    last = _strip_tags(str(row[1]))
    filer_name = f"{first} {last}".strip() or _strip_tags(str(row[2]))
    href = _first_href(str(row[3])) or _first_href(str(row[2]))
    if not href:
        return None
    detail_url = href if href.startswith("http") else f"{_BASE_URL}{href}"
    is_electronic = "/paper/" not in href.lower()
    received = _parse_datetime(_strip_tags(str(row[4])))
    return _Filing(
        filer_name=filer_name,
        detail_url=detail_url,
        is_electronic=is_electronic,
        received=received,
    )


def _parse_transactions(html: str) -> list[_Transaction]:
    """Parse the electronic PTR detail page's transactions table into rows.

    The detail page renders a single transactions ``<table>`` whose body rows have cells in the
    order [#, Transaction Date, Owner, Ticker, Asset Name, Asset Type, Type, Amount, Comment].
    We read positionally (header text varies across filing years) and tolerate a missing trailing
    Comment cell.
    """
    parser = _TransactionTableParser()
    parser.feed(html)
    transactions: list[_Transaction] = []
    for cells in parser.rows:
        # A valid transaction row has at least through the Amount column (index 7).
        if len(cells) < 8:
            continue
        ticker = cells[3].strip()
        # eFD prints "--" for assets with no ticker (mutual funds, etc.); those are not tradeable.
        if ticker in {"", "--"}:
            continue
        raw_date = cells[1].strip()
        transactions.append(
            _Transaction(
                ticker=ticker,
                asset_name=cells[4].strip(),
                transaction_type=cells[6].strip(),
                transaction_date=_parse_datetime(raw_date),
                transaction_date_raw=raw_date,
                amount_raw=cells[7].strip(),
            )
        )
    return transactions


class _TransactionTableParser(HTMLParser):
    """Collect ``<tbody>`` rows as lists of cell text from the PTR transactions table.

    Stdlib-only (no lxml/bs4 dependency). We only capture cells inside ``<tbody>`` so the header
    row never pollutes the data; each ``<td>``'s text is the concatenation of its text nodes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._in_body = False
        self._in_cell = False
        self._row: list[str] | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tbody":
            self._in_body = True
        elif tag == "tr" and self._in_body:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tbody":
            self._in_body = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag in {"td", "th"} and self._in_cell and self._row is not None:
            self._row.append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)


class _TagStripper(HTMLParser):
    """Flatten an HTML fragment to its visible text and capture the first anchor href."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.first_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and self.first_href is None:
            for key, value in attrs:
                if key == "href" and value:
                    self.first_href = value
                    break

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)


def _strip_tags(fragment: str) -> str:
    parser = _TagStripper()
    parser.feed(fragment)
    return " ".join("".join(parser.text_parts).split())


def _first_href(fragment: str) -> str | None:
    parser = _TagStripper()
    parser.feed(fragment)
    return parser.first_href


def _direction_sentiment(transaction_type: str) -> float:
    lowered = transaction_type.lower()
    if "purchase" in lowered or "buy" in lowered:
        return _BUY_SENTIMENT
    if "sale" in lowered or "sell" in lowered:
        return _SELL_SENTIMENT
    return 0.0


def _amount_multiplier(amount_raw: str) -> float:
    """Map an eFD amount range (e.g. "$50,001 - $100,000") to a weight multiplier in [1.0, 2.0].

    We use the bracket's upper bound: a larger disclosed position is a stronger conviction tell.
    Unparseable or empty ranges fall back to the neutral 1.0 multiplier.
    """
    upper = _amount_upper_bound(amount_raw)
    if upper <= 0:
        return 1.0
    for bound, multiplier in _AMOUNT_BRACKETS:
        if upper <= bound:
            return multiplier
    return 2.0


def _amount_upper_bound(amount_raw: str) -> int:
    """Pull the largest integer out of an eFD amount string ("$50,001 - $100,000" -> 100000)."""
    digits: list[str] = []
    numbers: list[int] = []
    for char in amount_raw.replace(",", ""):
        if char.isdigit():
            digits.append(char)
        elif digits:
            numbers.append(int("".join(digits)))
            digits.clear()
    if digits:
        numbers.append(int("".join(digits)))
    return max(numbers) if numbers else 0


def _parse_datetime(value: str) -> datetime | None:
    """Parse the date formats eFD uses: ISO (YYYY-MM-DD) and US (MM/DD/YYYY), to a UTC datetime."""
    cleaned = value.strip()[:10]
    for parser in (_iso_date, _us_date):
        parsed = parser(cleaned)
        if parsed is not None:
            return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
    return None


def _iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _us_date(value: str) -> date | None:
    parts = value.split("/")
    if len(parts) != 3:
        return None
    try:
        month, day, year = (int(part) for part in parts)
        return date(year, month, day)
    except ValueError:
        return None
