"""Reference data: the seed company directory and ticker-to-sector classification.

Static seed data, not secrets or tunable policy, so it lives in code rather than env.
Replace with a real provider adapter (e.g. an exchange listing feed) without touching
callers — both the entity resolver and sector map depend only on the shapes here.
"""

from types import MappingProxyType

from trading.domain.sectors import Sector

# Canonical company directory: ticker -> (sector, recognizable name aliases).
# Aliases are matched case-insensitively, on word boundaries, against free text during entity
# resolution (news mentions, $cashtags, and uppercased USAspending legal recipient names alike).
# Aliases therefore omit corporate suffixes (CORPORATION/COMPANY/INC/LLC/SYSTEMS/HOLDINGS/etc.)
# so "LOCKHEED MARTIN" matches "LOCKHEED MARTIN CORPORATION". Only publicly-traded issuers are
# listed (the strategy trades the parent's stock); the alias must be specific enough to avoid
# spurious hits on unrelated recipients.
COMPANY_DIRECTORY = MappingProxyType(
    {
        "AAPL": (Sector.TECHNOLOGY, ("apple", "apple inc")),
        "MSFT": (Sector.TECHNOLOGY, ("microsoft",)),
        "NVDA": (Sector.TECHNOLOGY, ("nvidia",)),
        "GOOGL": (Sector.COMMUNICATIONS, ("google", "alphabet")),
        "META": (Sector.COMMUNICATIONS, ("meta", "facebook")),
        "LMT": (Sector.DEFENSE, ("lockheed", "lockheed martin")),
        "RTX": (Sector.DEFENSE, ("raytheon", "rtx", "rtx corporation")),
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
        # --- Major publicly-traded federal contractors (USAspending recipient names) ----------
        "HII": (Sector.DEFENSE, ("huntington ingalls",)),
        "LHX": (Sector.DEFENSE, ("l3harris", "l3 harris", "harris corporation")),
        "LDOS": (Sector.DEFENSE, ("leidos",)),
        "BAH": (Sector.DEFENSE, ("booz allen", "booz allen hamilton")),
        "SAIC": (Sector.DEFENSE, ("saic", "science applications international")),
        "CACI": (Sector.DEFENSE, ("caci", "caci international")),
        "PLTR": (Sector.TECHNOLOGY, ("palantir",)),
        "ORCL": (Sector.TECHNOLOGY, ("oracle",)),
        "AMZN": (Sector.CONSUMER, ("amazon", "amazon web services")),
        "IBM": (Sector.TECHNOLOGY, ("ibm", "international business machines")),
        "ACN": (Sector.TECHNOLOGY, ("accenture",)),
        "GE": (Sector.INDUSTRIALS, ("general electric", "ge aerospace")),
        "HON": (Sector.INDUSTRIALS, ("honeywell",)),
        "GDYN": (Sector.TECHNOLOGY, ("grid dynamics",)),
        "TXT": (Sector.INDUSTRIALS, ("textron",)),
        "CW": (Sector.INDUSTRIALS, ("curtiss wright", "curtiss-wright")),
        "KTOS": (Sector.DEFENSE, ("kratos",)),
        "AVAV": (Sector.DEFENSE, ("aerovironment",)),
        "MRCY": (Sector.DEFENSE, ("mercury systems",)),
        "BAESY": (Sector.DEFENSE, ("bae systems",)),
        "CSCO": (Sector.TECHNOLOGY, ("cisco", "cisco systems")),
        "DELL": (Sector.TECHNOLOGY, ("dell technologies", "dell inc")),
        "HPE": (Sector.TECHNOLOGY, ("hewlett packard enterprise",)),
        "DXC": (Sector.TECHNOLOGY, ("dxc technology", "dxc")),
        "DAL": (Sector.INDUSTRIALS, ("delta air lines",)),
        "FDX": (Sector.INDUSTRIALS, ("fedex", "federal express")),
        "UPS": (Sector.INDUSTRIALS, ("united parcel service",)),
        "CAT": (Sector.INDUSTRIALS, ("caterpillar",)),
        "DE": (Sector.INDUSTRIALS, ("deere", "john deere")),
        "MMM": (Sector.INDUSTRIALS, ("3m company",)),
        "MCK": (Sector.HEALTHCARE, ("mckesson",)),
        "CAH": (Sector.HEALTHCARE, ("cardinal health",)),
        "HUM": (Sector.HEALTHCARE, ("humana",)),
        "CI": (Sector.HEALTHCARE, ("cigna",)),
        "CVS": (Sector.HEALTHCARE, ("cvs health", "cvs caremark")),
        "VLO": (Sector.ENERGY, ("valero",)),
        "COP": (Sector.ENERGY, ("conocophillips", "conoco phillips")),
        # --- Broad-market issuers commonly named in political / financial news ----------------
        # Ambiguous common-word / single-letter tickers use ONLY distinctive multi-word aliases
        # (never the bare word) to avoid false positives; the $TICKER cashtag path still works.
        "TSLA": (Sector.CONSUMER, ("tesla",)),
        "INTC": (Sector.TECHNOLOGY, ("intel corporation", "intel corp")),
        "AMD": (Sector.TECHNOLOGY, ("advanced micro devices",)),
        "QCOM": (Sector.TECHNOLOGY, ("qualcomm",)),
        "MU": (Sector.TECHNOLOGY, ("micron",)),
        "AVGO": (Sector.TECHNOLOGY, ("broadcom",)),
        "TXN": (Sector.TECHNOLOGY, ("texas instruments",)),
        "CRM": (Sector.TECHNOLOGY, ("salesforce",)),
        "ADBE": (Sector.TECHNOLOGY, ("adobe",)),
        "MRK": (Sector.HEALTHCARE, ("merck",)),
        "JNJ": (Sector.HEALTHCARE, ("johnson & johnson", "johnson and johnson")),
        "LLY": (Sector.HEALTHCARE, ("eli lilly",)),
        "ABBV": (Sector.HEALTHCARE, ("abbvie",)),
        "OXY": (Sector.ENERGY, ("occidental petroleum",)),
        "NEE": (Sector.UTILITIES, ("nextera",)),
        "BAC": (Sector.FINANCIALS, ("bank of america",)),
        "WFC": (Sector.FINANCIALS, ("wells fargo",)),
        "MS": (Sector.FINANCIALS, ("morgan stanley",)),
        "C": (Sector.FINANCIALS, ("citigroup", "citibank")),
        "BLK": (Sector.FINANCIALS, ("blackrock",)),
        "MA": (Sector.FINANCIALS, ("mastercard",)),
        "F": (Sector.CONSUMER, ("ford motor",)),
        "GM": (Sector.CONSUMER, ("general motors",)),
        "T": (Sector.COMMUNICATIONS, ("at&t",)),
        "VZ": (Sector.COMMUNICATIONS, ("verizon",)),
        "TMUS": (Sector.COMMUNICATIONS, ("t-mobile", "t mobile")),
        "DIS": (Sector.COMMUNICATIONS, ("walt disney", "disney")),
        "UAL": (Sector.INDUSTRIALS, ("united airlines",)),
        "AAL": (Sector.INDUSTRIALS, ("american airlines",)),
        "LUV": (Sector.INDUSTRIALS, ("southwest airlines",)),
        "WMT": (Sector.CONSUMER, ("walmart",)),
        "KO": (Sector.CONSUMER, ("coca-cola", "coca cola")),
        "PEP": (Sector.CONSUMER, ("pepsico",)),
        "NKE": (Sector.CONSUMER, ("nike",)),
    }
)
