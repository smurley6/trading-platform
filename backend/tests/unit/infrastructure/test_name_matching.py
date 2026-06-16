"""Tests for robust politician name matching (FR-POL-1).

Covers full-name matching, honorific + unambiguous surname, refusal to guess on ambiguous
or bare surnames, and deterministic tie-breaking.
"""

from collections.abc import Sequence

import pytest

from trading.domain.politicians import Politician, PoliticianRole
from trading.domain.provenance import Provenance
from trading.infrastructure.politicians.name_matching import find_politician


def _member(pid: str, name: str) -> Politician:
    return Politician(
        id=pid,
        name=name,
        role=PoliticianRole.MEMBER,
        committees=(),
        party="D",
        base_weight=1.0,
        track_record_score=0.5,
        provenance=Provenance.REAL,
    )


@pytest.fixture
def roster() -> Sequence[Politician]:
    # Two "Johnson"s share a surname (ambiguous); "Warren"/"Murkowski" are unique.
    return (
        _member("pol-warren", "Elizabeth Warren"),
        _member("pol-johnson-a", "Marcia Johnson"),
        _member("pol-johnson-b", "Hank Johnson"),
        _member("pol-murkowski", "Lisa Murkowski"),
    )


def test_full_name_match_case_insensitive_on_word_boundary(roster: Sequence[Politician]) -> None:
    result = find_politician("Today ELIZABETH WARREN spoke on the floor.", roster)
    assert result is not None
    assert result.id == "pol-warren"


def test_full_name_resolves_shared_surname(roster: Sequence[Politician]) -> None:
    # Full name disambiguates between the two Johnsons.
    result = find_politician("A statement from Hank Johnson today.", roster)
    assert result is not None
    assert result.id == "pol-johnson-b"


def test_honorific_plus_unique_surname_matches(roster: Sequence[Politician]) -> None:
    cases = {
        "Sen. Warren introduced a bill.": "pol-warren",
        "Senator Warren introduced a bill.": "pol-warren",
        "Rep. Murkowski commented.": "pol-murkowski",
        "Representative Murkowski commented.": "pol-murkowski",
        "Congresswoman Warren spoke.": "pol-warren",
        "Congressman Murkowski spoke.": "pol-murkowski",
    }
    for text, expected_id in cases.items():
        result = find_politician(text, roster)
        assert result is not None, text
        assert result.id == expected_id, text


def test_honorific_plus_ambiguous_surname_returns_none(roster: Sequence[Politician]) -> None:
    # "Johnson" is shared, so an honorific + bare surname must NOT guess.
    assert find_politician("Sen. Johnson voted yes.", roster) is None
    assert find_politician("Representative Johnson objected.", roster) is None


def test_ambiguous_surname_resolves_when_first_name_present(roster: Sequence[Politician]) -> None:
    # Adding the first name (full name) lets it resolve even past an honorific.
    result = find_politician("Sen. Marcia Johnson voted yes.", roster)
    assert result is not None
    assert result.id == "pol-johnson-a"


def test_bare_common_surname_does_not_match(roster: Sequence[Politician]) -> None:
    # "warren" as an ordinary word (a rabbit warren) with no honorific / first name.
    assert find_politician("The rabbits dug a warren under the hill.", roster) is None
    # Even an unambiguous surname alone is not enough without an honorific.
    assert find_politician("Murkowski is also a place name.", roster) is None


def test_empty_and_no_match_return_none(roster: Sequence[Politician]) -> None:
    assert find_politician("", roster) is None
    assert find_politician("   ...   ", roster) is None
    assert find_politician("No politicians mentioned here at all.", roster) is None


def test_determinism_on_tie() -> None:
    # Two distinct members whose identical full name both appear; the earliest stable key
    # (id) wins, independent of roster ordering and stable across repeated calls.
    a = _member("pol-aaa", "Sam Stone")
    b = _member("pol-zzz", "Sam Stone")  # identical name, different id
    text = "Sam Stone made the news."

    forward = find_politician(text, (a, b))
    reversed_ = find_politician(text, (b, a))
    assert forward is not None
    assert reversed_ is not None
    assert forward.id == "pol-aaa"  # lexicographically earliest id wins the tie
    assert reversed_.id == "pol-aaa"  # roster order does not change the result

    # Idempotent across repeated calls.
    ids: set[str] = set()
    for _ in range(5):
        again = find_politician(text, (a, b))
        assert again is not None
        ids.add(again.id)
    assert ids == {"pol-aaa"}


def test_full_name_beats_honorific_surname_when_both_present() -> None:
    # If one member's full name and another member's "honorific + unique surname" both
    # appear, the full-name match (more specific) wins regardless of ordering.
    warren = _member("pol-warren", "Elizabeth Warren")
    murkowski = _member("pol-murkowski", "Lisa Murkowski")
    roster = (warren, murkowski)
    text = "Elizabeth Warren met Sen. Murkowski."
    result = find_politician(text, roster)
    assert result is not None
    assert result.id == "pol-warren"
