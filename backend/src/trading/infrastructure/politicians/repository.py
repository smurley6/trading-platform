"""Static politician repository (FR-POL-3 inputs).

Seed roster used to weight signals by role and committee. Replace with a real data adapter
(e.g. a congressional-membership feed) behind `PoliticianRepositoryPort` without touching
callers. `base_weight` is seeded from the politician's role (see DEFAULT_BASE_WEIGHTS).
"""

from collections.abc import Sequence

from trading.domain.politicians import (
    DEFAULT_BASE_WEIGHTS,
    Politician,
    PoliticianRole,
)


def _seed() -> tuple[Politician, ...]:
    raw = [
        # (id, name, role, committees, party, track_record_score)
        ("pol-exec-1", "Dale Whitman", PoliticianRole.EXECUTIVE, (), "I", 0.7),
        (
            "pol-chair-armed",
            "Margaret Holt",
            PoliticianRole.COMMITTEE_CHAIR,
            ("armed services",),
            "R",
            0.8,
        ),
        (
            "pol-chair-fin",
            "Theodore Pierce",
            PoliticianRole.COMMITTEE_CHAIR,
            ("financial services",),
            "D",
            0.75,
        ),
        (
            "pol-chair-energy",
            "Carolyn Stacy",
            PoliticianRole.COMMITTEE_CHAIR,
            ("energy and commerce",),
            "D",
            0.7,
        ),
        (
            "pol-senior-armed",
            "Frank Delgado",
            PoliticianRole.SENIOR,
            ("armed services", "intelligence"),
            "R",
            0.65,
        ),
        (
            "pol-member-health",
            "Priya Nadkarni",
            PoliticianRole.MEMBER,
            ("health education labor and pensions",),
            "D",
            0.6,
        ),
        ("pol-member-1", "Walter Boyd", PoliticianRole.MEMBER, (), "R", 0.5),
    ]
    return tuple(
        Politician(
            id=pid,
            name=name,
            role=role,
            committees=committees,
            party=party,
            base_weight=DEFAULT_BASE_WEIGHTS[role],
            track_record_score=track,
        )
        for (pid, name, role, committees, party, track) in raw
    )


class StaticPoliticianRepository:
    def __init__(self, politicians: Sequence[Politician] | None = None) -> None:
        self._politicians = tuple(politicians) if politicians is not None else _seed()
        self._by_id = {p.id: p for p in self._politicians}
        # Longest names first so "Margaret Holt" wins over a hypothetical "Holt".
        self._by_name = sorted(self._politicians, key=lambda p: len(p.name), reverse=True)

    def get(self, politician_id: str) -> Politician | None:
        return self._by_id.get(politician_id)

    def find_in_text(self, text: str) -> Politician | None:
        lowered = text.lower()
        for politician in self._by_name:
            if politician.name.lower() in lowered:
                return politician
        return None

    def all(self) -> Sequence[Politician]:
        return self._politicians
