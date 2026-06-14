"""Sectors and committee-to-sector relevance (the topic_relevance term in FR-POL-3)."""

from collections.abc import Sequence
from enum import StrEnum
from types import MappingProxyType

# Relevance applied when a politician sits on a committee that oversees the ticker's sector.
_DIRECT_RELEVANCE = 1.0
# Politician has committees, but none oversee this sector.
_NO_MATCH_RELEVANCE = 0.3
# We can't place the ticker in a sector, or the politician has no committees on record.
_NEUTRAL_RELEVANCE = 0.5


class Sector(StrEnum):
    TECHNOLOGY = "technology"
    DEFENSE = "defense"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    FINANCIALS = "financials"
    INDUSTRIALS = "industrials"
    CONSUMER = "consumer"
    COMMUNICATIONS = "communications"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


# Congressional committees (normalized lower-case names) to the sectors they most directly
# oversee. Deliberately coarse; this is the gimmicky heart of the strategy, not gospel.
COMMITTEE_SECTOR_RELEVANCE = MappingProxyType(
    {
        "armed services": frozenset({Sector.DEFENSE, Sector.INDUSTRIALS}),
        "homeland security": frozenset({Sector.DEFENSE, Sector.TECHNOLOGY}),
        "intelligence": frozenset({Sector.DEFENSE, Sector.TECHNOLOGY}),
        "energy and commerce": frozenset(
            {Sector.ENERGY, Sector.HEALTHCARE, Sector.COMMUNICATIONS, Sector.TECHNOLOGY}
        ),
        "energy and natural resources": frozenset(
            {Sector.ENERGY, Sector.UTILITIES, Sector.MATERIALS}
        ),
        "commerce science and transportation": frozenset(
            {Sector.TECHNOLOGY, Sector.COMMUNICATIONS, Sector.INDUSTRIALS}
        ),
        "financial services": frozenset({Sector.FINANCIALS, Sector.REAL_ESTATE}),
        "banking housing and urban affairs": frozenset({Sector.FINANCIALS, Sector.REAL_ESTATE}),
        "health education labor and pensions": frozenset({Sector.HEALTHCARE}),
        "ways and means": frozenset({Sector.HEALTHCARE, Sector.FINANCIALS}),
        "agriculture": frozenset({Sector.MATERIALS, Sector.CONSUMER}),
    }
)


def committee_topic_relevance(committees: Sequence[str], sector: Sector | None) -> float:
    """How relevant a politician's committee assignments are to a ticker's sector (0-1).

    Direct oversight scores highest; an unknown sector or a politician with no committees
    on record falls back to a neutral baseline so the signal is neither boosted nor buried.
    """
    if sector is None or not committees:
        return _NEUTRAL_RELEVANCE
    for committee in committees:
        sectors = COMMITTEE_SECTOR_RELEVANCE.get(committee.strip().lower())
        if sectors and sector in sectors:
            return _DIRECT_RELEVANCE
    return _NO_MATCH_RELEVANCE
