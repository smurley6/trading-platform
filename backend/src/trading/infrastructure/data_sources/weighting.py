"""Shared politician influence weighting for data-source adapters (FR-POL-3)."""

from trading.application.ports.reference_data import SectorMapPort
from trading.domain.politicians import Politician
from trading.domain.sectors import committee_topic_relevance


def politician_weight_for(politician: Politician, ticker: str, sector_map: SectorMapPort) -> float:
    """base_weight (role) x topic_relevance (committee<->sector) x track_record_score."""
    sector = sector_map.sector_for(ticker)
    relevance = committee_topic_relevance(politician.committees, sector)
    return politician.influence_weight(relevance)
