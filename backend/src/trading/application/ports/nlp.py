"""NLP ports (FR-POL-1/2): entity resolution and sentiment, swappable behind interfaces.

M2 ships deterministic, dependency-light implementations; a model-backed adapter can drop
in later without touching the pipeline.
"""

from collections.abc import Sequence
from typing import Protocol


class EntityResolverPort(Protocol):
    def resolve_tickers(self, text: str) -> Sequence[str]:
        """Extract company mentions from text and resolve them to tickers (NER + lookup)."""
        ...


class SentimentAnalyzerPort(Protocol):
    def score(self, text: str, ticker: str) -> float:
        """Stance of `text` toward `ticker` as a sentiment in [-1, 1]."""
        ...
