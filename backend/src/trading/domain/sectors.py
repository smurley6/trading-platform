"""Sectors and committee-to-sector relevance (the topic_relevance term in FR-POL-3).

FR-POL-3 weights a politician's market influence by ``base_weight (role) x
topic_relevance x track_record_score``. This module owns ``topic_relevance``: how much a
politician's *real* congressional committee assignments plausibly bear on a given equity
sector. The mapping below covers the standing committees of the U.S. Congress, keyed by the
committee display names as the ``unitedstates/congress-legislators`` dataset spells them
(House and Senate), matched case- and punctuation-insensitively.

This is DOMAIN code: pure, deterministic, no I/O.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType

# Relevance applied when a politician sits on a committee that *directly* oversees the
# ticker's sector. Kept at 1.0 so a committee's primary sector saturates the signal (and so
# the historical binary behaviour is preserved for back-compat).
_DIRECT_RELEVANCE = 1.0
# Politician has committees, but none touch this sector at all: a low, non-zero baseline so
# the signal is dampened rather than erased.
_NO_MATCH_RELEVANCE = 0.3
# We can't place the ticker in a sector, or the politician has no committees on record: a
# neutral baseline so the signal is neither boosted nor buried.
_NEUTRAL_RELEVANCE = 0.5

# Graded weights for sectors a committee influences *secondarily* (real but partial nexus,
# e.g. a broad fiscal/tax committee that touches every sector, or a committee whose remit
# only partly overlaps the sector). All lie strictly between the no-match baseline and the
# direct-oversight ceiling, so a secondary nexus still beats "no nexus" without saturating.
_STRONG_SECONDARY = 0.7
_MODERATE_SECONDARY = 0.55
_BROAD_MARKET = 0.45


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


# Congressional standing committees -> the sectors they influence, each with a relevance in
# (0, 1]. Keys are the topical core of the committee's display name, already normalized by
# `_normalize` (lower-case, punctuation stripped, chamber prefix removed). Because the names
# below feed `_normalize` themselves, they may be written in human-readable form; both the
# bare topical name ("Armed Services") and the dataset's full display name ("House Committee
# on Armed Services" / "Senate Committee on Armed Services") normalize to the same key.
#
# A committee's *primary* sector(s) score `_DIRECT_RELEVANCE` (1.0); sectors it influences
# more partially score a graded secondary weight. Deliberately coarse: this is the gimmicky
# heart of the strategy (FR-POL-3), not gospel.
_RAW_COMMITTEE_SECTOR_RELEVANCE: Mapping[str, Mapping[Sector, float]] = {
    # --- National security ---------------------------------------------------------------
    "Armed Services": {
        Sector.DEFENSE: _DIRECT_RELEVANCE,
        Sector.INDUSTRIALS: _STRONG_SECONDARY,
        Sector.TECHNOLOGY: _MODERATE_SECONDARY,
    },
    "Homeland Security": {
        Sector.DEFENSE: _DIRECT_RELEVANCE,
        Sector.TECHNOLOGY: _STRONG_SECONDARY,
    },
    "Homeland Security and Governmental Affairs": {  # Senate counterpart
        Sector.DEFENSE: _DIRECT_RELEVANCE,
        Sector.TECHNOLOGY: _STRONG_SECONDARY,
    },
    "Intelligence": {  # House Permanent Select / Senate Select
        Sector.DEFENSE: _DIRECT_RELEVANCE,
        Sector.TECHNOLOGY: _DIRECT_RELEVANCE,
    },
    # --- Foreign policy: defense procurement + global energy ------------------------------
    "Foreign Affairs": {  # House
        Sector.DEFENSE: _STRONG_SECONDARY,
        Sector.ENERGY: _MODERATE_SECONDARY,
    },
    "Foreign Relations": {  # Senate
        Sector.DEFENSE: _STRONG_SECONDARY,
        Sector.ENERGY: _MODERATE_SECONDARY,
    },
    # --- Finance / fiscal ----------------------------------------------------------------
    "Financial Services": {  # House
        Sector.FINANCIALS: _DIRECT_RELEVANCE,
        Sector.REAL_ESTATE: _STRONG_SECONDARY,
    },
    "Banking, Housing, and Urban Affairs": {  # Senate counterpart
        Sector.FINANCIALS: _DIRECT_RELEVANCE,
        Sector.REAL_ESTATE: _STRONG_SECONDARY,
    },
    "Ways and Means": {  # House: tax, trade, health entitlements
        Sector.FINANCIALS: _DIRECT_RELEVANCE,
        Sector.HEALTHCARE: _STRONG_SECONDARY,
    },
    "Finance": {  # Senate counterpart of Ways and Means
        Sector.FINANCIALS: _DIRECT_RELEVANCE,
        Sector.HEALTHCARE: _STRONG_SECONDARY,
    },
    # Appropriations and Budget steer federal spending across the whole market, so they
    # carry a broad, even relevance to every sector rather than a single primary nexus.
    "Appropriations": dict.fromkeys(Sector, _BROAD_MARKET),
    "Budget": dict.fromkeys(Sector, _BROAD_MARKET),
    # --- Energy / resources / environment ------------------------------------------------
    "Energy and Commerce": {  # House: sprawling remit (energy + health + telecom + tech)
        Sector.ENERGY: _DIRECT_RELEVANCE,
        Sector.HEALTHCARE: _STRONG_SECONDARY,
        Sector.COMMUNICATIONS: _STRONG_SECONDARY,
        Sector.TECHNOLOGY: _MODERATE_SECONDARY,
        Sector.UTILITIES: _MODERATE_SECONDARY,
    },
    "Energy and Natural Resources": {  # Senate
        Sector.ENERGY: _DIRECT_RELEVANCE,
        Sector.UTILITIES: _STRONG_SECONDARY,
        Sector.MATERIALS: _STRONG_SECONDARY,
    },
    "Natural Resources": {  # House: public lands, mining, drilling
        Sector.ENERGY: _DIRECT_RELEVANCE,
        Sector.MATERIALS: _DIRECT_RELEVANCE,
        Sector.UTILITIES: _MODERATE_SECONDARY,
    },
    "Environment and Public Works": {  # Senate
        Sector.UTILITIES: _DIRECT_RELEVANCE,
        Sector.ENERGY: _STRONG_SECONDARY,
        Sector.MATERIALS: _MODERATE_SECONDARY,
        Sector.INDUSTRIALS: _MODERATE_SECONDARY,
    },
    # --- Health / labor ------------------------------------------------------------------
    "Health, Education, Labor, and Pensions": {  # Senate (HELP): pharma + health
        Sector.HEALTHCARE: _DIRECT_RELEVANCE,
    },
    "Education and the Workforce": {  # House: labor/pensions touch broad employers + health
        Sector.HEALTHCARE: _MODERATE_SECONDARY,
        Sector.FINANCIALS: _MODERATE_SECONDARY,
    },
    # --- Technology / antitrust ----------------------------------------------------------
    "Judiciary": {  # antitrust + IP: bears most on big tech / platforms
        Sector.TECHNOLOGY: _DIRECT_RELEVANCE,
        Sector.COMMUNICATIONS: _STRONG_SECONDARY,
    },
    # --- Commerce / industry / infrastructure --------------------------------------------
    "Commerce, Science, and Transportation": {  # Senate
        Sector.TECHNOLOGY: _DIRECT_RELEVANCE,
        Sector.COMMUNICATIONS: _STRONG_SECONDARY,
        Sector.INDUSTRIALS: _STRONG_SECONDARY,
    },
    "Transportation and Infrastructure": {  # House
        Sector.INDUSTRIALS: _DIRECT_RELEVANCE,
        Sector.MATERIALS: _MODERATE_SECONDARY,
    },
    "Science, Space, and Technology": {  # House
        Sector.TECHNOLOGY: _DIRECT_RELEVANCE,
        Sector.INDUSTRIALS: _MODERATE_SECONDARY,
    },
    "Small Business": {  # broad, mild nexus to consumer-facing and financial firms
        Sector.CONSUMER: _MODERATE_SECONDARY,
        Sector.FINANCIALS: _MODERATE_SECONDARY,
    },
    # --- Agriculture ---------------------------------------------------------------------
    "Agriculture": {  # House
        Sector.CONSUMER: _DIRECT_RELEVANCE,
        Sector.MATERIALS: _STRONG_SECONDARY,
    },
    "Agriculture, Nutrition, and Forestry": {  # Senate counterpart
        Sector.CONSUMER: _DIRECT_RELEVANCE,
        Sector.MATERIALS: _STRONG_SECONDARY,
    },
    # --- Veterans: chiefly health provision ----------------------------------------------
    "Veterans' Affairs": {
        Sector.HEALTHCARE: _STRONG_SECONDARY,
    },
}

# Chamber / boilerplate prefixes stripped from a committee name during normalization so the
# dataset's full display names collapse onto the bare topical keys above. Ordered longest
# first so the most specific prefix wins.
_NAME_PREFIXES: tuple[str, ...] = (
    "house permanent select committee on",
    "senate select committee on",
    "house select committee on",
    "joint committee on",
    "house committee on",
    "senate committee on",
    "permanent select committee on",
    "select committee on",
    "committee on",
)


def _normalize(name: str) -> str:
    """Reduce a committee name to a punctuation- and chamber-agnostic match key.

    Lower-cases, drops every character that is not a letter, digit or space, collapses
    runs of whitespace, then strips a leading chamber/boilerplate prefix and a trailing or
    leading bare "the". This lets both the bare topical name ("Armed Services") and the
    dataset's full display name ("House Committee on Armed Services") resolve to one key,
    satisfying the FR-POL-3 requirement that matching be robust to case and punctuation.
    """
    cleaned = "".join(c if (c.isalnum() or c.isspace()) else " " for c in name.lower())
    collapsed = " ".join(cleaned.split())
    for prefix in _NAME_PREFIXES:
        if collapsed.startswith(prefix):
            collapsed = collapsed[len(prefix) :].strip()
            break
    if collapsed.startswith("the "):
        collapsed = collapsed[len("the ") :].strip()
    return collapsed


# Public, normalized committee -> {sector: relevance} table. Built once at import time and
# frozen; this is the authoritative committee-to-sector taxonomy for FR-POL-3.
COMMITTEE_SECTOR_RELEVANCE: Mapping[str, Mapping[Sector, float]] = MappingProxyType(
    {
        _normalize(name): MappingProxyType(dict(sectors))
        for name, sectors in _RAW_COMMITTEE_SECTOR_RELEVANCE.items()
    }
)


def committee_topic_relevance(committees: Sequence[str], sector: Sector | None) -> float:
    """How relevant a politician's committee assignments are to a ticker's sector (0-1).

    FR-POL-3 ``topic_relevance`` term. A politician's real congressional committees are
    matched against :data:`COMMITTEE_SECTOR_RELEVANCE`; the strongest sector-specific
    relevance across all of their committees wins. Direct oversight scores
    ``_DIRECT_RELEVANCE`` (1.0); a partial nexus scores a graded value between the no-match
    baseline and 1.0; a politician who has committees but none touching the sector scores
    ``_NO_MATCH_RELEVANCE``. An unknown sector or a politician with no committees on record
    falls back to ``_NEUTRAL_RELEVANCE`` so the signal is neither boosted nor buried.

    Pure and deterministic: the return value depends only on the arguments.
    """
    if sector is None or not committees:
        return _NEUTRAL_RELEVANCE
    best = _NO_MATCH_RELEVANCE
    for committee in committees:
        sectors = COMMITTEE_SECTOR_RELEVANCE.get(_normalize(committee))
        if sectors is None:
            continue
        relevance = sectors.get(sector)
        if relevance is not None and relevance > best:
            best = relevance
            if best >= _DIRECT_RELEVANCE:
                break
    return best
