"""ClinicalTrials.gov API v2 study-update adapter (FR-ING-2, FR-ING-3, FR-ING-4).

Queries the free, keyless ClinicalTrials.gov v2 ``/studies`` endpoint for studies whose record was
last posted in the lookback window, resolves the sponsoring (or collaborating) company to a ticker,
and emits a clinical-catalyst :class:`~trading.domain.signals.Signal`. A trial phase/status change
for a tradeable biotech/pharma sponsor is a market catalyst for that issuer; the *direction* depends
on the trial's phase and overall status (a late-phase "Completed" trial reads bullish; a
"Terminated"/"Withdrawn"/"Suspended" trial reads bearish).

Why the v2 ``/studies`` endpoint: it is the single free, keyless, authoritative feed that ties a
*study record with a last-update date* to the *sponsor/collaborator named on the study*. Each result
is a nested study object; we request only the modules this catalyst needs via ``fields`` so retained
payloads stay bounded — the NCT id, brief title, overall status, phase(s), study type, lead
sponsor + collaborator names, and the ``lastUpdatePostDateStruct`` (a ``{"date": "YYYY-MM-DD"}``
struct). The lead-sponsor name and every collaborator name are concatenated and handed to the
injected :class:`~trading.application.ports.nlp.EntityResolverPort`, as the FDA / Federal Register
legs resolve a named entity; studies whose sponsor/collaborators resolve to no tracked ticker carry
no market relevance and are dropped (logged) — biotech/pharma is the catalyst here, not academia.

Direction (sentiment) is derived from the study's *status* and *phase*, not from prose:
  * A terminal-negative status (``TERMINATED`` / ``WITHDRAWN`` / ``SUSPENDED``) -> ``-0.6`` (a
    bearish read; a stopped trial is a setback for the sponsor).
  * ``COMPLETED`` at Phase 3 or Phase 4 -> ``+0.6`` (a finished late-phase trial is a strong
    bullish read toward approval); ``COMPLETED`` at any earlier/unknown phase -> ``+0.3`` (a
    weaker milestone).
  * Any other status (recruiting, active, not-yet-recruiting, unknown) -> ``0.0`` (neutral; a
    trial merely in progress is not yet a directional catalyst).

The endpoint paginates with an opaque ``nextPageToken`` echoed back as ``pageToken``; we follow it
up to a sane cap (``max_pages``) so a single poll ingests more than one page without unbounded
fetching (FR-ING-4 rate-limit courtesy). Server-side we filter to studies updated in the lookback
window via ``filter.advanced=AREA[LastUpdatePostDate]RANGE[<start>,MAX]``, newest first; the
``since`` cursor then drops any study at or before it client-side so a poll never re-emits.

ClinicalTrials.gov data is live, authoritative federal data with no politician dependency, so every
emitted Signal is stamped ``Provenance.REAL`` (the anti-placeholder guard requires REAL to drive
real decisions). The adapter is default OFF: an adapter constructed with ``enabled=False`` (the
default) yields no signals and makes no network call.

API reference: https://clinicaltrials.gov/data-api/api and
https://clinicaltrials.gov/api/v2/studies
"""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

logger = logging.getLogger(__name__)

# Source-type reliability prior (the source_reliability term in conviction scoring). A trial
# phase/status change is an authoritative clinical fact, but its *market reaction* is noisier than
# a finalized FDA approval (a completed trial is not yet an approval; the read can miss), so it sits
# a touch below the FDA regulatory prior. Defined HERE per this adapter's brief.
RELIABILITY_CLINICAL = 0.6

_ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"

# The record modules this catalyst needs; requesting only these keeps retained payloads bounded
# (a full study record is ~45 KB). The v2 API accepts these piped field/aggregate names.
_FIELDS = (
    "NCTId|BriefTitle|OverallStatus|Phase|StudyType"
    "|LeadSponsorName|CollaboratorName|LastUpdatePostDateStruct"
)

# Status/phase-conditioned sentiment priors (see the module docstring).
_TERMINAL_NEGATIVE_SENTIMENT = -0.6
_COMPLETED_LATE_PHASE_SENTIMENT = 0.6
_COMPLETED_OTHER_SENTIMENT = 0.3
_NEUTRAL_SENTIMENT = 0.0

# A study carries no dollar amount to scale by, so every resolved study gets the same baseline
# influence weight (>= 0), like the FDA approval / Federal Register document legs.
_STUDY_WEIGHT = 1.0

# Overall-status labels the v2 API emits that read as a setback for the sponsor.
_TERMINAL_NEGATIVE_STATUSES = frozenset({"TERMINATED", "WITHDRAWN", "SUSPENDED"})
_STATUS_COMPLETED = "COMPLETED"
# Phase labels the v2 API emits that count as "late phase" (strongest completion read).
_LATE_PHASES = frozenset({"PHASE3", "PHASE4"})


class ClinicalTrialsSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        entity_resolver: EntityResolverPort,
        clock: ClockPort,
        enabled: bool = False,
        lookback_days: int = 14,
        page_size: int = 100,
        max_pages: int = 5,
    ) -> None:
        self._client = client
        self._entity_resolver = entity_resolver
        self._clock = clock
        self._enabled = enabled
        self._lookback_days = lookback_days
        self._page_size = page_size
        self._max_pages = max_pages

    @property
    def name(self) -> str:
        return "clinical_trials"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        # Default OFF: a disabled adapter yields nothing and never opens a socket.
        if not self._enabled:
            return []

        signals: list[Signal] = []
        page_token: str | None = None
        for _page in range(self._max_pages):
            response = await self._client.get(_ENDPOINT, params=self._params(page_token))
            if response.status_code == httpx.codes.NOT_FOUND:
                # A zero-match search is a normal empty result, not an error: stop the walk.
                break
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            studies = body.get("studies")
            if not isinstance(studies, list):
                break

            for study in studies:
                if not isinstance(study, dict):
                    continue
                signals.extend(self._study_to_signals(study, since))

            next_token = body.get("nextPageToken")
            page_token = next_token if isinstance(next_token, str) and next_token else None
            if page_token is None:
                break

        return signals

    def _study_to_signals(self, study: dict[str, Any], since: datetime) -> list[Signal]:
        protocol = study.get("protocolSection")
        if not isinstance(protocol, dict):
            return []
        status_module = _as_dict(protocol.get("statusModule"))

        posted = _parse_ct_date(
            str(_as_dict(status_module.get("lastUpdatePostDateStruct")).get("date", ""))
        )
        if posted is None or posted <= since:
            # No usable date, or already ingested (the cursor filters at or before itself): skip.
            return []

        sponsor_text = _sponsor_text(protocol)
        tickers = self._entity_resolver.resolve_tickers(sponsor_text)
        if not tickers:
            # No tradeable sponsor/collaborator -> no market relevance, drop (like the FDA leg).
            logger.debug("clinical_trials: no ticker resolved for %r; skipping", sponsor_text[:80])
            return []

        status = str(status_module.get("overallStatus", "")).upper()
        phases = _phases(protocol)
        sentiment = _sentiment_for(status, phases)
        nct_id = str(_as_dict(protocol.get("identificationModule")).get("nctId", ""))

        signals: list[Signal] = []
        for ticker in tickers:
            signals.append(
                Signal(
                    id=f"clinical-trial-{nct_id or ticker}-{ticker}",
                    created_at=posted,
                    source_type=SignalSource.STATEMENT,
                    ticker=ticker,
                    politician_id=None,
                    sentiment=sentiment,
                    politician_weight=_STUDY_WEIGHT,
                    source_reliability=RELIABILITY_CLINICAL,
                    raw_payload=study,
                    # Real federal clinical data, no politician dependency -> REAL.
                    provenance=Provenance.REAL,
                )
            )
        return signals

    def _params(self, page_token: str | None) -> dict[str, Any]:
        start = (self._clock.now().date() - timedelta(days=self._lookback_days)).strftime(
            "%Y-%m-%d"
        )
        params: dict[str, Any] = {
            "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{start},MAX]",
            "sort": "LastUpdatePostDate:desc",
            "pageSize": self._page_size,
            "fields": _FIELDS,
        }
        if page_token is not None:
            # The opaque cursor from a prior page; the other params are re-sent unchanged.
            params["pageToken"] = page_token
        return params


def _sponsor_text(protocol: dict[str, Any]) -> str:
    """Lead-sponsor name plus every collaborator name, joined for entity resolution.

    The catalyst can attach to either the sponsor (Pfizer runs a trial) or a named collaborator
    (an academic center runs a trial with Pfizer), so both are resolved.
    """
    module = _as_dict(protocol.get("sponsorCollaboratorsModule"))
    names: list[str] = []
    lead = _as_dict(module.get("leadSponsor")).get("name")
    if isinstance(lead, str) and lead:
        names.append(lead)
    collaborators = module.get("collaborators")
    if isinstance(collaborators, list):
        for collaborator in collaborators:
            name = _as_dict(collaborator).get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return " ".join(names)


def _phases(protocol: dict[str, Any]) -> frozenset[str]:
    """The uppercased phase labels on a study (e.g. ``PHASE3``), or an empty set."""
    phases = _as_dict(protocol.get("designModule")).get("phases")
    if not isinstance(phases, list):
        return frozenset()
    return frozenset(str(phase).upper() for phase in phases if isinstance(phase, str))


def _sentiment_for(status: str, phases: frozenset[str]) -> float:
    if status in _TERMINAL_NEGATIVE_STATUSES:
        # A stopped trial is a setback for the sponsor.
        return _TERMINAL_NEGATIVE_SENTIMENT
    if status == _STATUS_COMPLETED:
        # A finished late-phase trial reads strongest toward approval; an earlier completion weaker.
        return (
            _COMPLETED_LATE_PHASE_SENTIMENT if phases & _LATE_PHASES else _COMPLETED_OTHER_SENTIMENT
        )
    # In-progress / unknown -> not yet a directional catalyst.
    return _NEUTRAL_SENTIMENT


def _as_dict(value: Any) -> dict[str, Any]:
    """The value if it is a dict, else an empty dict (defensive nested-record access)."""
    return value if isinstance(value, dict) else {}


def _parse_ct_date(value: str) -> datetime | None:
    """Parse a ClinicalTrials.gov ``YYYY-MM-DD`` date into a UTC datetime at midnight, or None."""
    try:
        parsed: date = date.fromisoformat(value.strip())
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
