"""Politicians and influence weighting (FR-POL-3)."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class PoliticianRole(StrEnum):
    EXECUTIVE = "executive"
    COMMITTEE_CHAIR = "committee_chair"
    SENIOR = "senior"
    MEMBER = "member"


# Default base weights by role; overridable per politician via `base_weight`.
DEFAULT_BASE_WEIGHTS = MappingProxyType(
    {
        PoliticianRole.EXECUTIVE: 1.5,
        PoliticianRole.COMMITTEE_CHAIR: 1.3,
        PoliticianRole.SENIOR: 1.1,
        PoliticianRole.MEMBER: 1.0,
    }
)


@dataclass(frozen=True, slots=True)
class Politician:
    id: str
    name: str
    role: PoliticianRole
    committees: tuple[str, ...]
    party: str
    base_weight: float
    track_record_score: float

    def __post_init__(self) -> None:
        if self.base_weight < 0:
            raise ValueError("base_weight must be non-negative")
        if self.track_record_score < 0:
            raise ValueError("track_record_score must be non-negative")

    def influence_weight(self, topic_relevance: float) -> float:
        """FR-POL-3: base_weight (by role) x topic_relevance x track_record_score."""
        if not 0.0 <= topic_relevance <= 1.0:
            raise ValueError("topic_relevance must be in [0, 1]")
        return self.base_weight * topic_relevance * self.track_record_score
