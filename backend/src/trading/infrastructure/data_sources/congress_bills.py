"""Congress.gov Bill Actions adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Pulls recently-updated bills from the free, official Congress.gov v3 API
(``https://api.congress.gov/v3/bill``), then walks each bill's *legislative actions* feed
(``/v3/bill/{congress}/{type}/{number}/actions``) and emits one Signal per significant action
whose bill title resolves to a tracked ticker. A bill's floor passage, committee referral, or a
recorded vote is a market-relevant legislative catalyst for the companies/sectors the bill
touches, so each action becomes a directionally-scored :class:`~trading.domain.signals.Signal`.

Why two hops (list bills by date, then fetch per-bill actions): the ``/v3/bill`` list endpoint is
the only one that filters by an ``updateDate`` window (``fromDateTime``/``toDateTime``), but it
carries only the *latest* action inline. The full, dated action history — the thing we score —
lives on the per-bill ``/actions`` sub-resource. So we page the date-filtered bill list, then
fetch each bill's actions, exactly as the openFDA leg pages a search then reads each record's
submissions.

Ticker resolution mirrors the SEC/FDA legs: the bill *title* is run through the injected
:class:`EntityResolverPort`; a bill whose title resolves to no tracked ticker is dropped (logged),
so this leg never emits an unattributable signal. Direction is read from the action text/type:
passage reads bullish, a failed motion bearish, a referral mildly positive, other committee/
procedural actions mildly negative. Every emitted Signal is stamped :class:`Provenance.REAL`
(official federal legislative record, no politician dependency beyond an optional sponsor) and
classified :data:`SignalSource.STATEMENT` (a bill action is an official legislative statement, not
a disclosed trade or a contract award).

Only actions strictly after the ``since`` cursor are emitted, so a poll never re-emits an action
it has already seen. Default OFF: an unconfigured adapter (``enabled=False``) yields no signals and
makes no network call — construction never opens a socket, and the API key is injected, never
hardcoded.

API reference: https://api.congress.gov/ and https://gpo.congress.gov/#/
"""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

logger = logging.getLogger(__name__)

# Per-source-type reliability prior for legislative-record data. Congress.gov actions are
# authoritative, immutable federal facts once recorded, so this sits near the top of the range
# (just under a hard contract award). Lives HERE per this adapter's build constraints.
RELIABILITY_BILL_ACTIONS = 0.9

# Directional sentiment priors for the action types we score (all within [-1, 1]).
SENTIMENT_PASSAGE = 0.7  # a bill passing is a bullish catalyst for the companies it touches
SENTIMENT_REFERRAL = 0.3  # a committee referral is mildly positive (legislative interest)
SENTIMENT_FAILURE = -0.4  # a failed/recommitted motion is bearish (opposition/delay)
SENTIMENT_COMMITTEE = -0.2  # other committee/procedural friction is neutral-to-negative

_BILL_ENDPOINT = "https://api.congress.gov/v3/bill"

# Every resolved action gets the same baseline influence weight (>= 0): a bill action carries no
# dollar amount to scale by, mirroring the FDA-approval leg's flat weight.
_ACTION_WEIGHT = 1.0

# Congress.gov offset/limit paging cap (limit max is 250; we default well under that).
_MAX_LIMIT = 250


class CongressBillsSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        api_key: str | None = None,
        enabled: bool = False,
        lookback_days: int = 30,
        page_limit: int = 100,
        max_pages: int = 10,
        max_actions_per_bill: int = 250,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._api_key = api_key
        self._enabled = enabled
        self._lookback_days = lookback_days
        self._page_limit = min(page_limit, _MAX_LIMIT)
        self._max_pages = max_pages
        self._max_actions_per_bill = max_actions_per_bill

    @property
    def name(self) -> str:
        return "congress_bills"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        # Default OFF: an unconfigured/disabled adapter makes no network call and yields nothing.
        if not self._enabled:
            return []

        signals: list[Signal] = []
        for page in range(self._max_pages):
            offset = page * self._page_limit
            body = await self._get_bill_page(offset)
            bills: list[dict[str, Any]] = body.get("bills", []) if body is not None else []

            for bill in bills:
                tickers = self._resolve_bill_tickers(bill)
                if not tickers:
                    logger.debug(
                        "congress_bills: bill %s has no resolvable ticker; skipping",
                        _bill_ref(bill),
                    )
                    continue
                actions = await self._fetch_actions(bill)
                for action in actions:
                    created_at = _action_datetime(action)
                    if created_at is None or created_at <= since:
                        # No parseable date, or already ingested (cursor filters at or before
                        # itself) -> skip just this action.
                        continue
                    for ticker in tickers:
                        signals.append(self._to_signal(bill, action, ticker, created_at))

            pagination = (body or {}).get("pagination", {})
            has_next = isinstance(pagination, dict) and bool(pagination.get("next"))
            # A short page or the absence of a `next` link is the last page; stop the walk.
            if len(bills) < self._page_limit or not has_next:
                break
        logger.info("congress_bills: emitted %d signal(s)", len(signals))
        return signals

    async def _get_bill_page(self, offset: int) -> dict[str, Any] | None:
        response = await self._client.get(
            _BILL_ENDPOINT, params=self._bill_params(offset), timeout=30.0
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            # A window with no matching bills reads as 404 on some Congress.gov paths; that is a
            # normal empty result, not an error.
            return None
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body

    async def _fetch_actions(self, bill: dict[str, Any]) -> list[dict[str, Any]]:
        congress = bill.get("congress")
        bill_type = str(bill.get("type", "")).lower()
        number = bill.get("number")
        if congress is None or not bill_type or number is None:
            return []
        url = f"{_BILL_ENDPOINT}/{congress}/{bill_type}/{number}/actions"
        response = await self._client.get(url, params=self._actions_params(), timeout=20.0)
        if response.status_code == httpx.codes.NOT_FOUND:
            return []
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        actions = body.get("actions", [])
        if not isinstance(actions, list):
            return []
        return [a for a in actions if isinstance(a, dict)][: self._max_actions_per_bill]

    def _to_signal(
        self, bill: dict[str, Any], action: dict[str, Any], ticker: str, created_at: datetime
    ) -> Signal:
        raw_payload: dict[str, Any] = {
            "congress": bill.get("congress"),
            "type": bill.get("type"),
            "number": bill.get("number"),
            "title": bill.get("title"),
            "policyArea": bill.get("policyArea"),
            "sponsors": bill.get("sponsors"),
            "ticker": ticker,
            "action": action,
        }
        sponsor_id = _sponsor_id(bill)
        return Signal(
            id=_signal_id(bill, action, ticker),
            created_at=created_at,
            source_type=SignalSource.STATEMENT,
            ticker=ticker,
            politician_id=sponsor_id,
            sentiment=_action_sentiment(action),
            politician_weight=_ACTION_WEIGHT,
            source_reliability=RELIABILITY_BILL_ACTIONS,
            raw_payload=raw_payload,
            # Real federal legislative record, no politician dependency required -> REAL.
            provenance=Provenance.REAL,
        )

    def _resolve_bill_tickers(self, bill: dict[str, Any]) -> Sequence[str]:
        title = str(bill.get("title") or "")
        tickers = list(self._entity_resolver.resolve_tickers(title))
        if tickers:
            return tickers
        # Fall back to the bill's policy area name (e.g. "Energy") for a sector-keyword match.
        policy_area = bill.get("policyArea")
        if isinstance(policy_area, dict):
            name = str(policy_area.get("name") or "")
            if name:
                return list(self._entity_resolver.resolve_tickers(name))
        return []

    def _bill_params(self, offset: int) -> dict[str, Any]:
        now = self._clock.now()
        start = now - timedelta(days=self._lookback_days)
        params: dict[str, Any] = {
            "format": "json",
            "offset": offset,
            "limit": self._page_limit,
            "sort": "updateDate+desc",
            "fromDateTime": _to_congress_dt(start),
            "toDateTime": _to_congress_dt(now),
        }
        if self._api_key is not None:
            # The free api.data.gov key raises the rate limit; passed as the ?api_key query param.
            params["api_key"] = self._api_key
        return params

    def _actions_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"format": "json", "limit": min(self._max_actions_per_bill, 250)}
        if self._api_key is not None:
            params["api_key"] = self._api_key
        return params


def _bill_ref(bill: dict[str, Any]) -> str:
    return f"{bill.get('congress')}-{bill.get('type')}-{bill.get('number')}"


def _signal_id(bill: dict[str, Any], action: dict[str, Any], ticker: str) -> str:
    congress = bill.get("congress")
    bill_type = str(bill.get("type", "")).lower()
    number = bill.get("number")
    # Prefer the stable action code; fall back to the action date so IDs stay deterministic.
    marker = str(action.get("actionCode") or action.get("actionDate") or "")
    return f"congress-{congress}-{bill_type}-{number}-{marker}-{ticker}"


def _sponsor_id(bill: dict[str, Any]) -> str | None:
    """The primary sponsor's BioGuide ID, if the bill record carries sponsors; else None."""
    sponsors = bill.get("sponsors")
    if isinstance(sponsors, list) and sponsors:
        first = sponsors[0]
        if isinstance(first, dict):
            bioguide = first.get("bioguideId")
            if bioguide:
                return str(bioguide)
    return None


def _action_sentiment(action: dict[str, Any]) -> float:
    """Score a bill action's market direction from its type and text (result within [-1, 1])."""
    text = str(action.get("text") or "")
    lowered = text.lower()
    action_type = str(action.get("type") or "")
    if "failed" in lowered or "recommit" in lowered:
        return SENTIMENT_FAILURE
    if "passed" in lowered or "passage" in lowered:
        return SENTIMENT_PASSAGE
    if action_type == "Referral" or "referred to" in lowered:
        return SENTIMENT_REFERRAL
    if action_type == "Committee" or "committee" in lowered:
        return SENTIMENT_COMMITTEE
    # Any other recorded action is a mild positive (legislative movement on the bill).
    return SENTIMENT_REFERRAL


def _action_datetime(action: dict[str, Any]) -> datetime | None:
    """Combine an action's ``actionDate`` (YYYY-MM-DD) and optional ``actionTime`` (HH:MM:SS)
    into a UTC datetime; missing time defaults to 00:00:00Z. Returns None if the date is absent
    or malformed."""
    raw_date = str(action.get("actionDate") or "")
    try:
        parsed_date = date.fromisoformat(raw_date)
    except ValueError:
        return None
    raw_time = str(action.get("actionTime") or "")
    parsed_time = _parse_time(raw_time)
    return datetime.combine(parsed_date, parsed_time, tzinfo=UTC)


def _parse_time(value: str) -> time:
    try:
        return time.fromisoformat(value) if value else time(0, 0, 0)
    except ValueError:
        return time(0, 0, 0)


def _to_congress_dt(value: datetime) -> str:
    """Format a datetime as the ISO-8601-with-Z window bound Congress.gov requires."""
    aware = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.strftime("%Y-%m-%dT%H:%M:%SZ")
