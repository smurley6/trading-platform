"""Real congressional-roster adapter (FR-POL-3) — unitedstates/congress-legislators.

Replaces the fictional seed roster (`StaticPoliticianRepository._seed`) with ACTUAL sitting
members of Congress, sourced from the FREE, public, no-key ``unitedstates/congress-legislators``
dataset on GitHub. Every :class:`~trading.domain.politicians.Politician` built here is stamped
``provenance=Provenance.REAL`` so that the news/disclosed-trade legs it backs can produce REAL
signals — the anti-placeholder guard
(:mod:`trading.application.data_integrity`) blocks SYNTHETIC data from any real decision path,
so an un-stamped roster would silently neuter every downstream signal.

The module is split into three honest layers so parsing is unit-testable with zero network:

* :func:`parse_roster` — a PURE function: three raw YAML strings in, real Politicians out.
* :func:`fetch_roster` / :func:`fetch_and_parse_roster` — thin async I/O over
  ``httpx.AsyncClient`` that GETs the three YAML files (only invoked when the orchestrator
  enables the roster; the network is never touched by tests).
* :class:`CongressLegislatorsRepository` — a :class:`PoliticianRepositoryPort` seeded from the
  parsed politicians.

Role classification (no executive branch exists in this dataset, so EXECUTIVE is never emitted):

* a member who holds the *Chair* title on any committee  -> ``COMMITTEE_CHAIR``;
* otherwise, a *Ranking Member* OR a clearly senior member (>= ``_SENIOR_TERM_THRESHOLD``
  served terms) -> ``SENIOR``;
* everyone else -> ``MEMBER``.

``track_record_score`` is set to the NEUTRAL value ``1.0`` for every real member. A genuine
predictiveness score would require a historical congressional-trades corpus the platform does
not yet have; rather than fabricate signal, we decline to up/down-weight: ``influence_weight``
(FR-POL-3) reduces to ``base_weight x topic_relevance``. This is the single honest
placeholder-but-neutral value in the adapter and is documented as such.

**Historical roster (Task 04b, docs/RESEARCH-PLAN.md).** ``legislators-historical.yaml`` (same
repo, same per-member schema as ``legislators-current.yaml``) covers every RETIRED/former member
back to 1789 — without it, a deep House-disclosures backfill (2008+) would silently drop every
PTR filed by a since-retired member, because the live roster only ever covers CURRENT sitting
members. :func:`parse_historical_roster` classifies retired members from ``terms`` ALONE (there
is no current committee-membership file for someone no longer serving): >= ``_SENIOR_TERM_
THRESHOLD`` terms served -> ``SENIOR``, else ``MEMBER``; ``committees=()`` (empty — a former
member holds no active committee seat) and ``track_record_score=1.0`` (same neutral convention
as the current-roster parser), still stamped ``Provenance.REAL`` (this is genuine historical
public-record data, not a placeholder). The composition root is responsible for combining
current + historical into one :class:`CongressLegislatorsRepository` for the historical House
source — see ``main.py``'s lazy historical-roster loader (fetched only on first use, never at
startup, so a bare app boot stays network-silent).

**Nicknames (Task 07, docs/RESEARCH-PLAN.md).** Both parsers also carry ``name.nickname`` (e.g.
official first name "Richard" / nickname "Rick") onto :class:`Politician.nickname` — a filer
string using the informal form ("Rick Allen") would otherwise never contiguously match the
official full name. See :mod:`trading.infrastructure.politicians.name_matching` for how this
is consumed.
"""

from collections.abc import Sequence
from typing import Final

import httpx
import yaml

from trading.domain.politicians import (
    DEFAULT_BASE_WEIGHTS,
    Politician,
    PoliticianRole,
)
from trading.domain.provenance import Provenance
from trading.infrastructure.politicians.name_matching import find_politician

# Raw GitHub URLs (main branch) for the three FREE, public, no-key source files.
_RAW_BASE: Final = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main"
LEGISLATORS_URL: Final = f"{_RAW_BASE}/legislators-current.yaml"
COMMITTEES_URL: Final = f"{_RAW_BASE}/committees-current.yaml"
COMMITTEE_MEMBERSHIP_URL: Final = f"{_RAW_BASE}/committee-membership-current.yaml"
# Same repo, same per-member schema; every RETIRED/former member back to 1789 (Task 04b). No
# committee-membership counterpart exists for this file (retired members hold no active seat).
LEGISLATORS_HISTORICAL_URL: Final = f"{_RAW_BASE}/legislators-historical.yaml"

# A member who has served at least this many terms is treated as "clearly senior".
_SENIOR_TERM_THRESHOLD: Final = 6

# NEUTRAL track-record score: declines to up/down-weight (see module docstring). Multiplying by
# 1.0 leaves FR-POL-3's influence_weight at base_weight x topic_relevance.
_NEUTRAL_TRACK_RECORD: Final = 1.0

# Committee-membership titles that signal role (matched case-insensitively). The dataset uses
# "Chair"/"Chairman"/"Chairwoman" for the gavel-holder; "Vice Chair" is NOT a full chair and is
# deliberately excluded so we don't over-weight them. "Ranking Member" is the senior minority.
_CHAIR_TITLES: Final = frozenset({"chair", "chairman", "chairwoman"})
_RANKING_TITLE: Final = "ranking member"


def parse_roster(
    legislators_yaml: str,
    committees_yaml: str,
    membership_yaml: str,
) -> tuple[Politician, ...]:
    """Build REAL :class:`Politician` objects from the three congress-legislators YAML files.

    Pure and offline: identical inputs yield identical outputs, no I/O. See the module docstring
    for the role-classification and neutral-track-record rules (FR-POL-3).

    Args:
        legislators_yaml: contents of ``legislators-current.yaml``.
        committees_yaml: contents of ``committees-current.yaml``.
        membership_yaml: contents of ``committee-membership-current.yaml``.

    Returns:
        A tuple of Politicians, each stamped ``provenance=Provenance.REAL``, ordered by the
        underlying legislators file.
    """
    committee_names = _committee_display_names(committees_yaml)
    memberships = _parse_membership(membership_yaml, committee_names)

    politicians: list[Politician] = []
    legislators = yaml.safe_load(legislators_yaml) or []
    for entry in legislators:
        bioguide = _bioguide(entry)
        name = _official_full(entry)
        if not bioguide or not name:
            continue

        terms = entry.get("terms") or []
        latest = terms[-1] if terms else {}
        party = str(latest.get("party", "") or "")

        member_committees = memberships.get(bioguide, _MemberCommittees())
        role = _classify_role(
            is_chair=member_committees.is_chair,
            is_ranking_member=member_committees.is_ranking_member,
            term_count=len(terms),
        )
        politicians.append(
            Politician(
                id=bioguide,
                name=name,
                role=role,
                committees=member_committees.display_names,
                party=party,
                base_weight=DEFAULT_BASE_WEIGHTS[role],
                track_record_score=_NEUTRAL_TRACK_RECORD,
                provenance=Provenance.REAL,
                nickname=_nickname(entry),
            )
        )
    return tuple(politicians)


def parse_historical_roster(legislators_historical_yaml: str) -> tuple[Politician, ...]:
    """Build REAL :class:`Politician` objects from ``legislators-historical.yaml`` (Task 04b).

    Pure and offline, same schema as :func:`parse_roster`'s legislators file, but with NO
    committee-membership counterpart: a retired member holds no active committee seat, so role
    classification here relies on ``terms`` alone (>= ``_SENIOR_TERM_THRESHOLD`` served terms ->
    ``SENIOR``, else ``MEMBER``; a *Chair*/*Ranking Member* title cannot be inferred without a
    membership file, so ``COMMITTEE_CHAIR`` is never emitted here) and ``committees=()``.
    ``track_record_score`` stays the same NEUTRAL ``1.0`` as the current-roster parser.

    Args:
        legislators_historical_yaml: contents of ``legislators-historical.yaml``.

    Returns:
        A tuple of Politicians, each stamped ``provenance=Provenance.REAL``, ordered by the
        underlying legislators-historical file.
    """
    politicians: list[Politician] = []
    legislators = yaml.safe_load(legislators_historical_yaml) or []
    for entry in legislators:
        bioguide = _bioguide(entry)
        name = _official_full(entry)
        if not bioguide or not name:
            continue

        terms = entry.get("terms") or []
        latest = terms[-1] if terms else {}
        party = str(latest.get("party", "") or "")

        role = _classify_role(is_chair=False, is_ranking_member=False, term_count=len(terms))
        politicians.append(
            Politician(
                id=bioguide,
                name=name,
                role=role,
                committees=(),
                party=party,
                base_weight=DEFAULT_BASE_WEIGHTS[role],
                track_record_score=_NEUTRAL_TRACK_RECORD,
                provenance=Provenance.REAL,
                nickname=_nickname(entry),
            )
        )
    return tuple(politicians)


async def fetch_roster(client: httpx.AsyncClient) -> tuple[str, str, str]:
    """GET the three source YAML files and return their raw text.

    Kept deliberately separate from :func:`parse_roster` so parsing stays unit-testable with no
    network. Only called when the orchestrator enables the roster (e.g. via a
    ``ENABLE_CONGRESS_ROSTER`` flag).

    Returns:
        ``(legislators_yaml, committees_yaml, membership_yaml)``.
    """
    legislators = await _get_text(client, LEGISLATORS_URL)
    committees = await _get_text(client, COMMITTEES_URL)
    membership = await _get_text(client, COMMITTEE_MEMBERSHIP_URL)
    return legislators, committees, membership


async def fetch_and_parse_roster(client: httpx.AsyncClient) -> tuple[Politician, ...]:
    """Convenience: :func:`fetch_roster` followed by :func:`parse_roster`."""
    legislators, committees, membership = await fetch_roster(client)
    return parse_roster(legislators, committees, membership)


async def fetch_historical_roster(client: httpx.AsyncClient) -> str:
    """GET ``legislators-historical.yaml`` and return its raw text (Task 04b).

    Kept separate from :func:`parse_historical_roster` for the same reason as the current-roster
    split: parsing stays unit-testable with no network. Only called by the House disclosures
    historical source's LAZY roster loader (never at startup — see ``main.py``).
    """
    return await _get_text(client, LEGISLATORS_HISTORICAL_URL)


async def fetch_and_parse_historical_roster(client: httpx.AsyncClient) -> tuple[Politician, ...]:
    """Convenience: :func:`fetch_historical_roster` followed by :func:`parse_historical_roster`."""
    legislators_historical = await fetch_historical_roster(client)
    return parse_historical_roster(legislators_historical)


async def fetch_and_parse_combined_roster(client: httpx.AsyncClient) -> tuple[Politician, ...]:
    """Current + historical members in one call (Task 04b): current + retired, current first.

    The convenience the historical House source's lazy loader uses to resolve filers across
    both sitting and since-retired members in a single :class:`CongressLegislatorsRepository`.
    """
    current = await fetch_and_parse_roster(client)
    historical = await fetch_and_parse_historical_roster(client)
    return current + historical


async def _get_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


class CongressLegislatorsRepository:
    """:class:`PoliticianRepositoryPort` backed by the real congress-legislators roster.

    Seeded from parsed Politicians (use :func:`fetch_and_parse_roster` to populate from the
    live feed, or :func:`parse_roster` directly in tests). Name resolution delegates to the
    shared :func:`trading.infrastructure.politicians.name_matching.find_politician`.
    """

    def __init__(self, politicians: Sequence[Politician]) -> None:
        self._politicians: tuple[Politician, ...] = tuple(politicians)
        self._by_id: dict[str, Politician] = {p.id: p for p in self._politicians}

    def get(self, politician_id: str) -> Politician | None:
        return self._by_id.get(politician_id)

    def find_in_text(self, text: str) -> Politician | None:
        return find_politician(text, self._politicians)

    def all(self) -> Sequence[Politician]:
        return self._politicians


class _MemberCommittees:
    """Accumulated committee facts for one bioguide id while parsing the membership file."""

    __slots__ = ("_names", "is_chair", "is_ranking_member")

    def __init__(self) -> None:
        self._names: list[str] = []
        self.is_chair: bool = False
        self.is_ranking_member: bool = False

    def add(self, display_name: str, title: str) -> None:
        if display_name not in self._names:
            self._names.append(display_name)
        normalized = title.strip().lower()
        if normalized in _CHAIR_TITLES:
            self.is_chair = True
        elif normalized == _RANKING_TITLE:
            self.is_ranking_member = True

    @property
    def display_names(self) -> tuple[str, ...]:
        return tuple(self._names)


def _committee_display_names(committees_yaml: str) -> dict[str, str]:
    """Map ``thomas_id`` -> committee display ``name`` (top-level committees only).

    Subcommittee membership in the membership file is keyed by ``<parent>+<sub>`` thomas ids;
    we resolve those to the parent committee's display name so the emitted committee strings
    stay at full-committee granularity (matching the sector-relevance taxonomy).
    """
    mapping: dict[str, str] = {}
    committees = yaml.safe_load(committees_yaml) or []
    for committee in committees:
        thomas_id = committee.get("thomas_id")
        name = committee.get("name")
        if isinstance(thomas_id, str) and isinstance(name, str):
            mapping[thomas_id] = name
    return mapping


def _parse_membership(
    membership_yaml: str,
    committee_names: dict[str, str],
) -> dict[str, _MemberCommittees]:
    """Map ``bioguide id`` -> :class:`_MemberCommittees` from the membership file."""
    by_member: dict[str, _MemberCommittees] = {}
    membership = yaml.safe_load(membership_yaml) or {}
    for thomas_id, members in membership.items():
        display_name = committee_names.get(_parent_thomas_id(str(thomas_id)))
        if display_name is None or not isinstance(members, list):
            continue
        for member in members:
            bioguide = member.get("bioguide")
            if not isinstance(bioguide, str):
                continue
            record = by_member.setdefault(bioguide, _MemberCommittees())
            record.add(display_name, str(member.get("title", "") or ""))
    return by_member


def _parent_thomas_id(thomas_id: str) -> str:
    """Reduce a subcommittee id (``HSAG15``) to its parent committee id (``HSAG``).

    Top-level committee ids are a 4-character alpha code; subcommittees append numeric digits.
    """
    parent = thomas_id[:4]
    return parent if parent else thomas_id


def _classify_role(
    *,
    is_chair: bool,
    is_ranking_member: bool,
    term_count: int,
) -> PoliticianRole:
    """Classify a member into the EXISTING :class:`PoliticianRole` (never EXECUTIVE)."""
    if is_chair:
        return PoliticianRole.COMMITTEE_CHAIR
    if is_ranking_member or term_count >= _SENIOR_TERM_THRESHOLD:
        return PoliticianRole.SENIOR
    return PoliticianRole.MEMBER


def _bioguide(entry: dict[str, object]) -> str:
    ids = entry.get("id")
    if isinstance(ids, dict):
        bioguide = ids.get("bioguide")
        if isinstance(bioguide, str):
            return bioguide
    return ""


def _official_full(entry: dict[str, object]) -> str:
    name = entry.get("name")
    if isinstance(name, dict):
        official_full = name.get("official_full")
        if isinstance(official_full, str):
            return official_full
    return ""


def _nickname(entry: dict[str, object]) -> str | None:
    """The ``name.nickname`` field (Task 07, docs/RESEARCH-PLAN.md) — a preferred given name
    distinct from the official first name (e.g. official "Richard" / nickname "Rick"), used by
    :mod:`trading.infrastructure.politicians.name_matching` as an alternate first-name token when
    matching filer strings that use the informal form. ``None`` when absent (most members)."""
    name = entry.get("name")
    if isinstance(name, dict):
        nickname = name.get("nickname")
        if isinstance(nickname, str) and nickname.strip():
            return nickname.strip()
    return None
