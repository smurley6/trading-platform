"""openFDA Drugs@FDA drug-approval adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Queries the public Drugs@FDA endpoint (``/drug/drugsfda.json``) for recently *approved*
applications, resolves the sponsoring company to a ticker, and emits a bullish catalyst
Signal. A new FDA drug approval is a strong, market-moving event for the sponsoring issuer,
so the signal is positive and stamped high-reliability. There is no politician attached to a
regulatory event, so ``politician_id`` is None.

Why Drugs@FDA (``/drug/drugsfda.json``) over the other openFDA drug endpoints: it is the only
endpoint that ties the three things this catalyst needs together in one record — an *approval
status with a date* and the *sponsor company* and the *drug*. Each result carries
``sponsor_name`` and a ``submissions`` array; an approval is a submission whose
``submission_status == "AP"`` with a concrete ``submission_status_date`` (``YYYYMMDD``). The
drug labeling endpoint (``/drug/label.json``) has the drug but no decision/date; the
enforcement endpoint (``/drug/enforcement.json``) is *recalls* (a bearish event with no
approval semantics). So Drugs@FDA is the correct, reliable ``(event_date, sponsor, drug)``
source for approval catalysts.

The endpoint paginates with ``skip``/``limit`` (openFDA's offset paging); ``meta.results.total``
bounds the set. We walk pages up to a sane cap (``max_pages``) so a single poll ingests more
than ``limit`` approvals without unbounded fetching (FR-ING-4 rate-limit courtesy). Results are
filtered server-side to approvals whose status date falls in the lookback window; the ``since``
cursor then drops any approval at or before the cursor (already-ingested events) client-side so
a poll never re-emits.

A drug approval is real federal regulatory data with no politician dependency, so every emitted
Signal is stamped ``Provenance.REAL`` (the anti-placeholder guard requires REAL to drive
decisions).

API reference: https://open.fda.gov/apis/drug/drugsfda/ and https://api.fda.gov/drug/drugsfda.json
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

# Source-type reliability prior (the source_reliability term in conviction scoring). FDA approval
# data is an authoritative federal fact, but the *market reaction* is less mechanical than a
# contract award (priced-in expectations, label scope), so it sits below RELIABILITY_CONTRACT.
# Lives here (not in data_sources/__init__.py) per the build constraints for this adapter.
RELIABILITY_REGULATORY = 0.7

_ENDPOINT = "https://api.fda.gov/drug/drugsfda.json"
# A drug approval is a strong but bounded bullish catalyst for the sponsoring issuer.
_APPROVAL_SENTIMENT = 0.6
# A modest constant influence weight: an approval is significant but, unlike a contract award,
# carries no dollar amount to scale by, so every approval gets the same baseline weight (>= 0).
_APPROVAL_WEIGHT = 1.0
# openFDA submission_status value meaning "Approved".
_STATUS_APPROVED = "AP"


class FdaApprovalsSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        lookback_days: int = 30,
        api_key: str | None = None,
        page_limit: int = 100,
        max_pages: int = 10,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._lookback_days = lookback_days
        self._api_key = api_key
        self._page_limit = page_limit
        self._max_pages = max_pages

    @property
    def name(self) -> str:
        return "fda_approvals"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        signals: list[Signal] = []
        for page in range(self._max_pages):
            skip = page * self._page_limit
            response = await self._client.get(_ENDPOINT, params=self._params(skip))
            if response.status_code == httpx.codes.NOT_FOUND:
                # openFDA returns 404 when a search matches zero records (no approvals in the
                # window). That is a normal empty result, not an error: stop the walk.
                break
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            results: list[dict[str, Any]] = body.get("results", [])

            for record in results:
                approved = _approval_date(record)
                if approved is None or approved <= since:
                    # No in-record approval, or already ingested (the cursor filters at or before
                    # itself): skip just this record.
                    continue
                sponsor = str(record.get("sponsor_name", ""))
                tickers = self._entity_resolver.resolve_tickers(sponsor)
                if not tickers:
                    continue
                application = str(record.get("application_number", ""))
                for ticker in tickers:
                    signals.append(
                        Signal(
                            id=f"fda-approval-{application or ticker}-{ticker}",
                            created_at=approved,
                            source_type=SignalSource.STATEMENT,
                            ticker=ticker,
                            politician_id=None,
                            sentiment=_APPROVAL_SENTIMENT,
                            politician_weight=_APPROVAL_WEIGHT,
                            source_reliability=RELIABILITY_REGULATORY,
                            raw_payload=record,
                            # Real federal regulatory data, no politician dependency -> REAL.
                            provenance=Provenance.REAL,
                        )
                    )

            # A short page (fewer rows than the limit) or an empty page is the last page; stop.
            if len(results) < self._page_limit:
                break
        return signals

    def _params(self, skip: int) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).strftime("%Y%m%d")
        end = self._clock.now().date().strftime("%Y%m%d")
        # Server-side filter: an approved submission whose status date is in the lookback window.
        search = (
            f"submissions.submission_status:{_STATUS_APPROVED}"
            f" AND submissions.submission_status_date:[{start} TO {end}]"
        )
        params: dict[str, Any] = {"search": search, "limit": self._page_limit, "skip": skip}
        if self._api_key is not None:
            # An optional openFDA key raises the rate limit; passed as the 'api_key' query param.
            params["api_key"] = self._api_key
        return params


def _approval_date(record: dict[str, Any]) -> datetime | None:
    """The most recent approved submission date in a Drugs@FDA record, or None.

    A record's ``submissions`` array can hold the original approval plus later supplements; we
    take the latest ``AP`` submission so an approved supplement (a fresh catalyst) is dated to
    when it happened rather than the original filing.
    """
    submissions = record.get("submissions")
    if not isinstance(submissions, list):
        return None
    latest: datetime | None = None
    for submission in submissions:
        if not isinstance(submission, dict):
            continue
        if str(submission.get("submission_status", "")) != _STATUS_APPROVED:
            continue
        parsed = _parse_fda_date(str(submission.get("submission_status_date", "")))
        if parsed is not None and (latest is None or parsed > latest):
            latest = parsed
    return latest


def _parse_fda_date(value: str) -> datetime | None:
    """Parse an openFDA ``YYYYMMDD`` date into a UTC datetime, or None if malformed."""
    if len(value) != 8 or not value.isdigit():
        return None
    try:
        parsed: date = date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
