"""Data-source adapters (FR-ING-2): news/statements, USAspending contracts, congress trades.

Each implements `DataSourcePort`, normalizes provider records to the internal `Signal`
schema, and retains the raw payload (FR-ING-3). Adapters take an injected `httpx.AsyncClient`
so tests drive them with a mock transport rather than the network.
"""

# Per-source-type reliability priors (the source_reliability term in conviction scoring).
# Contracts are hard facts; statements are noisier; disclosed trades are lagged and indirect.
RELIABILITY_CONTRACT = 0.9
RELIABILITY_STATEMENT = 0.6
RELIABILITY_DISCLOSED_TRADE = 0.7
