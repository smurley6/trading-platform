"""News/RSS statement adapter (FR-ING-2, FR-POL-1/2).

Pulls RSS 2.0 feeds, scans each item for a known politician and resolvable tickers, and
emits a statement Signal per (politician, ticker) pair scored by sentiment and influence
weight. Items without a recognized politician or ticker are skipped (but a real deployment
may want to record raw misses too). RSS is parsed with the stdlib — no extra dependency.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from trading.application.ports.clock import ClockPort
from trading.application.ports.nlp import EntityResolverPort, SentimentAnalyzerPort
from trading.application.ports.politician_repository import PoliticianRepositoryPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.data_sources import RELIABILITY_STATEMENT
from trading.infrastructure.data_sources.weighting import politician_weight_for

logger = logging.getLogger(__name__)


class NewsSource:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        feed_urls: Sequence[str],
        entity_resolver: EntityResolverPort,
        sentiment: SentimentAnalyzerPort,
        politicians: PoliticianRepositoryPort,
        sector_map: SectorMapPort,
        clock: ClockPort,
    ) -> None:
        self._client = client
        self._feed_urls = list(feed_urls)
        self._entity_resolver = entity_resolver
        self._sentiment = sentiment
        self._politicians = politicians
        self._sector_map = sector_map
        self._clock = clock

    @property
    def name(self) -> str:
        return "news"

    async def fetch_signals(self, since: datetime) -> Sequence[Signal]:
        signals: list[Signal] = []
        for url in self._feed_urls:
            response = await self._client.get(url)
            response.raise_for_status()
            signals.extend(self._parse_feed(response.text, since))
        return signals

    def _parse_feed(self, body: str, since: datetime) -> list[Signal]:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError:
            logger.warning("could not parse RSS feed body; skipping")
            return []

        signals: list[Signal] = []
        for item in root.iter("item"):
            title = _text(item, "title")
            description = _text(item, "description")
            text = f"{title} {description}".strip()
            if not text:
                continue
            published = _parse_pubdate(_text(item, "pubDate")) or self._clock.now()
            if published <= since:
                continue
            politician = self._politicians.find_in_text(text)
            if politician is None:
                continue
            tickers = self._entity_resolver.resolve_tickers(text)
            for ticker in tickers:
                sentiment = self._sentiment.score(text, ticker)
                if sentiment == 0.0:
                    continue
                weight = politician_weight_for(politician, ticker, self._sector_map)
                signals.append(
                    Signal(
                        id=f"news-{politician.id}-{ticker}-{int(published.timestamp())}",
                        created_at=published,
                        source_type=SignalSource.STATEMENT,
                        ticker=ticker,
                        politician_id=politician.id,
                        sentiment=sentiment,
                        politician_weight=weight,
                        source_reliability=RELIABILITY_STATEMENT,
                        raw_payload={
                            "title": title,
                            "description": description,
                            "link": _text(item, "link"),
                            "pub_date": _text(item, "pubDate"),
                        },
                    )
                )
        return signals


def _text(item: ElementTree.Element, tag: str) -> str:
    child = item.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _parse_pubdate(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
