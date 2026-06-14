"""Reference data: the seed company directory and ticker-to-sector classification.

Static seed data, not secrets or tunable policy, so it lives in code rather than env.
Replace with a real provider adapter (e.g. an exchange listing feed) without touching
callers — both the entity resolver and sector map depend only on the shapes here.
"""

from types import MappingProxyType

from trading.domain.sectors import Sector

# Canonical company directory: ticker -> (sector, recognizable name aliases).
# Aliases are matched case-insensitively against free text during entity resolution.
COMPANY_DIRECTORY = MappingProxyType(
    {
        "AAPL": (Sector.TECHNOLOGY, ("apple", "apple inc")),
        "MSFT": (Sector.TECHNOLOGY, ("microsoft",)),
        "NVDA": (Sector.TECHNOLOGY, ("nvidia",)),
        "GOOGL": (Sector.COMMUNICATIONS, ("google", "alphabet")),
        "META": (Sector.COMMUNICATIONS, ("meta", "facebook")),
        "LMT": (Sector.DEFENSE, ("lockheed", "lockheed martin")),
        "RTX": (Sector.DEFENSE, ("raytheon", "rtx")),
        "NOC": (Sector.DEFENSE, ("northrop", "northrop grumman")),
        "GD": (Sector.DEFENSE, ("general dynamics",)),
        "BA": (Sector.INDUSTRIALS, ("boeing",)),
        "PFE": (Sector.HEALTHCARE, ("pfizer",)),
        "MRNA": (Sector.HEALTHCARE, ("moderna",)),
        "UNH": (Sector.HEALTHCARE, ("unitedhealth", "united health")),
        "XOM": (Sector.ENERGY, ("exxon", "exxonmobil", "exxon mobil")),
        "CVX": (Sector.ENERGY, ("chevron",)),
        "JPM": (Sector.FINANCIALS, ("jpmorgan", "jp morgan", "jpmorgan chase")),
        "GS": (Sector.FINANCIALS, ("goldman", "goldman sachs")),
    }
)
