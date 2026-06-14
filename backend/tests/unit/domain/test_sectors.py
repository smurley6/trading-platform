from trading.domain.sectors import Sector, committee_topic_relevance


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
