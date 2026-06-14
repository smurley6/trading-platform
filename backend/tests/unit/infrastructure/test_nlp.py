from trading.infrastructure.nlp.entity_resolution import GazetteerEntityResolver
from trading.infrastructure.nlp.sentiment import LexiconSentiment


class TestEntityResolution:
    def setup_method(self) -> None:
        self.resolver = GazetteerEntityResolver()

    def test_resolves_company_name_to_ticker(self) -> None:
        assert "AAPL" in self.resolver.resolve_tickers("Senator praises Apple's new factory")

    def test_resolves_multiword_alias(self) -> None:
        assert "LMT" in self.resolver.resolve_tickers("a new Lockheed Martin contract")

    def test_resolves_cashtag(self) -> None:
        assert self.resolver.resolve_tickers("bullish on $NVDA today") == ["NVDA"]

    def test_ignores_unknown_cashtag(self) -> None:
        assert self.resolver.resolve_tickers("what about $ZZZZ") == []

    def test_word_boundary_prevents_false_positive(self) -> None:
        # "applesauce" must not match "apple".
        assert "AAPL" not in self.resolver.resolve_tickers("the applesauce market")

    def test_deduplicates_repeated_mentions(self) -> None:
        assert self.resolver.resolve_tickers("Apple, Apple, $AAPL") == ["AAPL"]


class TestSentiment:
    def setup_method(self) -> None:
        self.sentiment = LexiconSentiment()

    def test_positive_statement(self) -> None:
        assert self.sentiment.score("strong support and new funding", "AAPL") > 0

    def test_negative_statement(self) -> None:
        assert self.sentiment.score("a probe and possible sanctions", "AAPL") < 0

    def test_negation_flips_polarity(self) -> None:
        assert self.sentiment.score("does not support the deal", "AAPL") < 0

    def test_neutral_statement_is_zero(self) -> None:
        assert self.sentiment.score("the committee met on Tuesday", "AAPL") == 0.0

    def test_score_is_bounded(self) -> None:
        loud = "support boost approve fund award win strong growth praise"
        assert -1.0 <= self.sentiment.score(loud, "AAPL") <= 1.0
