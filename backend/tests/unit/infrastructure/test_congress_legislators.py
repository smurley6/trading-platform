"""Real congressional-roster parser (FR-POL-3) — offline, vendored inline YAML fixtures.

Exercises :func:`parse_roster` against SMALL inline fixtures modelled on the actual
``unitedstates/congress-legislators`` schema, with 4 real current members (a committee chair, a
ranking member, a senior-by-tenure senator, and a junior rank-and-file rep), across both
chambers. Asserts every emitted Politician is stamped ``Provenance.REAL`` with the correct
role, committees, and party. ZERO network calls.
"""

from collections.abc import Callable

import httpx
import pytest

from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.domain.politicians import DEFAULT_BASE_WEIGHTS, Politician, PoliticianRole
from trading.domain.provenance import Provenance
from trading.infrastructure.politicians.congress_legislators import (
    CongressLegislatorsRepository,
    fetch_and_parse_roster,
    parse_roster,
)

Handler = Callable[[httpx.Request], httpx.Response]

# --- Vendored inline fixtures (real members; trimmed to the fields the parser reads) ---------

# Glenn Thompson chairs the House Committee on Agriculture; Jasmine Crockett is a junior House
# member (one term); Maria Cantwell is a long-tenured senator (>= 6 terms here); Roger Wicker is
# Ranking Member of the Senate Armed Services Committee in the fixture.
_LEGISLATORS_YAML = """
- id:
    bioguide: T000467
  name:
    first: Glenn
    last: Thompson
    official_full: Glenn Thompson
  terms:
    - type: rep
      state: PA
      party: Republican
    - type: rep
      state: PA
      party: Republican
- id:
    bioguide: C001067
  name:
    first: Jasmine
    last: Crockett
    official_full: Jasmine Crockett
  terms:
    - type: rep
      state: TX
      party: Democrat
- id:
    bioguide: C000127
  name:
    first: Maria
    last: Cantwell
    official_full: Maria Cantwell
  terms:
    - type: rep
      state: WA
      party: Democrat
    - type: sen
      state: WA
      party: Democrat
    - type: sen
      state: WA
      party: Democrat
    - type: sen
      state: WA
      party: Democrat
    - type: sen
      state: WA
      party: Democrat
    - type: sen
      state: WA
      party: Democrat
- id:
    bioguide: W000437
  name:
    first: Roger
    last: Wicker
    official_full: Roger F. Wicker
  terms:
    - type: sen
      state: MS
      party: Republican
"""

_COMMITTEES_YAML = """
- type: house
  name: House Committee on Agriculture
  thomas_id: HSAG
  subcommittees:
    - name: Conservation, Research, and Biotechnology
      thomas_id: "15"
- type: senate
  name: Senate Committee on Commerce, Science, and Transportation
  thomas_id: SSCM
- type: senate
  name: Senate Committee on Armed Services
  thomas_id: SSAS
"""

_MEMBERSHIP_YAML = """
HSAG:
  - name: Glenn Thompson
    party: majority
    rank: 1
    title: Chair
    bioguide: T000467
  - name: Jasmine Crockett
    party: minority
    rank: 12
    bioguide: C001067
HSAG15:
  - name: Glenn Thompson
    party: majority
    rank: 1
    bioguide: T000467
SSCM:
  - name: Maria Cantwell
    party: majority
    rank: 1
    bioguide: C000127
SSAS:
  - name: Roger F. Wicker
    party: minority
    rank: 1
    title: Ranking Member
    bioguide: W000437
"""


@pytest.fixture
def roster() -> dict[str, Politician]:
    politicians = parse_roster(_LEGISLATORS_YAML, _COMMITTEES_YAML, _MEMBERSHIP_YAML)
    return {p.id: p for p in politicians}


def test_parse_roster_yields_four_real_politicians() -> None:
    politicians = parse_roster(_LEGISLATORS_YAML, _COMMITTEES_YAML, _MEMBERSHIP_YAML)
    assert len(politicians) == 4
    # The whole point of the adapter: every real member is stamped REAL (anti-placeholder guard).
    assert all(p.provenance is Provenance.REAL for p in politicians)


def test_committee_chair_is_classified_and_carries_full_committee_name(
    roster: dict[str, Politician],
) -> None:
    thompson = roster["T000467"]
    assert thompson.name == "Glenn Thompson"
    assert thompson.party == "Republican"
    assert thompson.role is PoliticianRole.COMMITTEE_CHAIR
    assert thompson.base_weight == DEFAULT_BASE_WEIGHTS[PoliticianRole.COMMITTEE_CHAIR]
    # Subcommittee membership (HSAG15) collapses to the parent display name; no duplicates.
    assert thompson.committees == ("House Committee on Agriculture",)
    assert thompson.track_record_score == 1.0  # neutral


def test_ranking_member_is_senior(roster: dict[str, Politician]) -> None:
    wicker = roster["W000437"]
    assert wicker.name == "Roger F. Wicker"
    assert wicker.role is PoliticianRole.SENIOR
    assert wicker.base_weight == DEFAULT_BASE_WEIGHTS[PoliticianRole.SENIOR]
    assert wicker.committees == ("Senate Committee on Armed Services",)


def test_long_tenure_member_is_senior(roster: dict[str, Politician]) -> None:
    cantwell = roster["C000127"]
    # Six served terms with no title still reads SENIOR via the tenure threshold.
    assert cantwell.role is PoliticianRole.SENIOR
    assert cantwell.committees == ("Senate Committee on Commerce, Science, and Transportation",)


def test_junior_rank_and_file_is_member(roster: dict[str, Politician]) -> None:
    crockett = roster["C001067"]
    assert crockett.role is PoliticianRole.MEMBER
    assert crockett.party == "Democrat"
    assert crockett.committees == ("House Committee on Agriculture",)
    assert crockett.base_weight == DEFAULT_BASE_WEIGHTS[PoliticianRole.MEMBER]


def test_parse_roster_is_pure_and_deterministic() -> None:
    first = parse_roster(_LEGISLATORS_YAML, _COMMITTEES_YAML, _MEMBERSHIP_YAML)
    second = parse_roster(_LEGISLATORS_YAML, _COMMITTEES_YAML, _MEMBERSHIP_YAML)
    assert first == second


def test_repository_conforms_to_port() -> None:
    politicians = parse_roster(_LEGISLATORS_YAML, _COMMITTEES_YAML, _MEMBERSHIP_YAML)
    # Structural conformance to the port is checked by mypy via this annotation.
    repo: PoliticianRepositoryPort = CongressLegislatorsRepository(politicians)
    assert isinstance(repo, CongressLegislatorsRepository)


def test_repository_lookup_and_find_in_text() -> None:
    politicians = parse_roster(_LEGISLATORS_YAML, _COMMITTEES_YAML, _MEMBERSHIP_YAML)
    repo = CongressLegislatorsRepository(politicians)
    assert repo.get("T000467") is not None
    assert repo.get("missing") is None
    assert len(repo.all()) == 4
    found = repo.find_in_text("Rep. Glenn Thompson announced new farm subsidies today.")
    assert found is not None
    assert found.id == "T000467"
    assert found.provenance is Provenance.REAL


async def test_fetch_and_parse_roster_uses_mock_transport() -> None:
    bodies = {
        "legislators-current.yaml": _LEGISLATORS_YAML,
        "committees-current.yaml": _COMMITTEES_YAML,
        "committee-membership-current.yaml": _MEMBERSHIP_YAML,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        filename = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, text=bodies[filename])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        politicians = await fetch_and_parse_roster(client)

    assert len(politicians) == 4
    assert all(p.provenance is Provenance.REAL for p in politicians)
