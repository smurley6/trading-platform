import pytest

from trading.domain.sectors import (
    COMMITTEE_SECTOR_RELEVANCE,
    Sector,
    committee_topic_relevance,
)

# --- Back-compat: the original behaviour must be preserved exactly --------------------------


def test_direct_committee_oversight_scores_highest() -> None:
    assert committee_topic_relevance(["armed services"], Sector.DEFENSE) == 1.0


def test_committee_without_oversight_scores_low() -> None:
    relevance = committee_topic_relevance(["armed services"], Sector.HEALTHCARE)
    assert relevance == 0.3


def test_unknown_sector_is_neutral() -> None:
    assert committee_topic_relevance(["armed services"], None) == 0.5


def test_no_committees_is_neutral() -> None:
    assert committee_topic_relevance([], Sector.DEFENSE) == 0.5


def test_committee_name_is_normalized() -> None:
    assert committee_topic_relevance(["  Armed Services  "], Sector.DEFENSE) == 1.0


def test_any_matching_committee_counts() -> None:
    relevance = committee_topic_relevance(["agriculture", "financial services"], Sector.FINANCIALS)
    assert relevance == 1.0


# --- Case / punctuation / full-display-name robustness (FR-POL-3) ---------------------------


def test_matching_is_case_insensitive() -> None:
    assert committee_topic_relevance(["ARMED SERVICES"], Sector.DEFENSE) == 1.0


def test_full_house_display_name_resolves() -> None:
    """The unitedstates/congress-legislators dataset spelling must resolve."""
    assert committee_topic_relevance(["House Committee on Armed Services"], Sector.DEFENSE) == 1.0


def test_full_senate_display_name_resolves() -> None:
    assert committee_topic_relevance(["Senate Committee on Armed Services"], Sector.DEFENSE) == 1.0


def test_intelligence_select_committee_prefix_resolves() -> None:
    assert (
        committee_topic_relevance(
            ["House Permanent Select Committee on Intelligence"], Sector.TECHNOLOGY
        )
        == 1.0
    )
    assert (
        committee_topic_relevance(["Senate Select Committee on Intelligence"], Sector.DEFENSE)
        == 1.0
    )


def test_internal_punctuation_is_ignored() -> None:
    """Commas in the real Senate names must not break the match."""
    assert (
        committee_topic_relevance(["Banking, Housing, and Urban Affairs"], Sector.FINANCIALS) == 1.0
    )
    assert (
        committee_topic_relevance(["Health, Education, Labor, and Pensions"], Sector.HEALTHCARE)
        == 1.0
    )


# --- The required committee -> sector primary mappings --------------------------------------


@pytest.mark.parametrize(
    ("committee", "sector"),
    [
        ("Armed Services", Sector.DEFENSE),
        ("Homeland Security", Sector.DEFENSE),
        ("Intelligence", Sector.DEFENSE),
        ("Intelligence", Sector.TECHNOLOGY),
        ("Financial Services", Sector.FINANCIALS),
        ("Banking, Housing, and Urban Affairs", Sector.FINANCIALS),
        ("Energy and Commerce", Sector.ENERGY),
        ("Energy and Natural Resources", Sector.ENERGY),
        ("Health, Education, Labor, and Pensions", Sector.HEALTHCARE),
        ("Judiciary", Sector.TECHNOLOGY),
        ("Ways and Means", Sector.FINANCIALS),
        ("Finance", Sector.FINANCIALS),
        ("Transportation and Infrastructure", Sector.INDUSTRIALS),
        ("Commerce, Science, and Transportation", Sector.TECHNOLOGY),
        ("Natural Resources", Sector.ENERGY),
        ("Natural Resources", Sector.MATERIALS),
        ("Agriculture", Sector.CONSUMER),
        ("Agriculture, Nutrition, and Forestry", Sector.CONSUMER),
    ],
)
def test_required_primary_mappings_score_direct(committee: str, sector: Sector) -> None:
    assert committee_topic_relevance([committee], sector) == 1.0


# --- Graded secondary relevance ------------------------------------------------------------


def test_secondary_sector_scores_between_baseline_and_direct() -> None:
    """Energy & Commerce touches healthcare/telecom but less than its primary energy remit."""
    primary = committee_topic_relevance(["Energy and Commerce"], Sector.ENERGY)
    secondary = committee_topic_relevance(["Energy and Commerce"], Sector.HEALTHCARE)
    assert primary == 1.0
    assert 0.3 < secondary < 1.0


def test_armed_services_has_graded_industrial_nexus() -> None:
    relevance = committee_topic_relevance(["Armed Services"], Sector.INDUSTRIALS)
    assert 0.3 < relevance < 1.0


def test_foreign_affairs_touches_defense_and_energy() -> None:
    assert 0.3 < committee_topic_relevance(["Foreign Affairs"], Sector.DEFENSE) < 1.0
    assert 0.3 < committee_topic_relevance(["Foreign Relations"], Sector.ENERGY) < 1.0


def test_unmapped_committee_falls_back_to_low_baseline() -> None:
    """A committee with no entry at all contributes only the low baseline."""
    assert committee_topic_relevance(["House Committee on Ethics"], Sector.TECHNOLOGY) == 0.3


def test_mapped_committee_unrelated_sector_falls_back_to_low_baseline() -> None:
    assert committee_topic_relevance(["Armed Services"], Sector.REAL_ESTATE) == 0.3


# --- Broad-market committees ---------------------------------------------------------------


@pytest.mark.parametrize("sector", list(Sector))
def test_appropriations_has_a_relevance_for_every_sector(sector: Sector) -> None:
    relevance = committee_topic_relevance(["Appropriations"], sector)
    assert 0.3 < relevance < 1.0


@pytest.mark.parametrize("sector", list(Sector))
def test_budget_has_a_relevance_for_every_sector(sector: Sector) -> None:
    relevance = committee_topic_relevance(["Budget"], sector)
    assert 0.3 < relevance < 1.0


# --- Aggregation across multiple committees ------------------------------------------------


def test_strongest_relevance_across_committees_wins() -> None:
    """A direct-oversight committee should beat a merely-secondary one for the same sector."""
    # Appropriations gives a broad (sub-1.0) relevance to healthcare; HELP gives direct 1.0.
    relevance = committee_topic_relevance(
        ["Appropriations", "Health, Education, Labor, and Pensions"], Sector.HEALTHCARE
    )
    assert relevance == 1.0


def test_max_not_sum_so_relevance_stays_in_unit_interval() -> None:
    relevance = committee_topic_relevance(
        ["Armed Services", "Homeland Security", "Intelligence"], Sector.DEFENSE
    )
    assert relevance == 1.0


# --- Invariants on the published taxonomy --------------------------------------------------


def test_all_relevances_are_in_unit_interval() -> None:
    for sectors in COMMITTEE_SECTOR_RELEVANCE.values():
        for relevance in sectors.values():
            assert 0.0 < relevance <= 1.0


def test_every_returned_relevance_is_in_unit_interval() -> None:
    committees = [
        "Armed Services",
        "Energy and Commerce",
        "Judiciary",
        "Appropriations",
        "Unmapped Committee",
    ]
    for committee in committees:
        for sector in Sector:
            relevance = committee_topic_relevance([committee], sector)
            assert 0.0 <= relevance <= 1.0
