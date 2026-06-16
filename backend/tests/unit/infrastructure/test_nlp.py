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

    def test_resolves_expanded_companies(self) -> None:
        assert "TSLA" in self.resolver.resolve_tickers("a Tesla gigafactory tax break")
        assert "MRK" in self.resolver.resolve_tickers("Merck pricing hearing")

    def test_resolves_possessive(self) -> None:
        assert "NVDA" in self.resolver.resolve_tickers("concerns over Nvidia's exports")

    def test_resolves_multiple_companies(self) -> None:
        got = self.resolver.resolve_tickers("the bill affects Microsoft and Pfizer")
        assert "MSFT" in got
        assert "PFE" in got

    def test_common_words_do_not_false_positive(self) -> None:
        # Common-word / single-letter tickers (V, F, T, C, MA, GM, ...) use only distinctive
        # multi-word aliases, so ordinary prose must resolve to nothing.
        text = "All the key members met on time; it was a fine session at the capitol."
        assert self.resolver.resolve_tickers(text) == []


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

    def test_real_headline_positive(self) -> None:
        assert self.sentiment.score("Senator applauds Boeing investment", "BA") > 0

    def test_real_headline_negative(self) -> None:
        assert self.sentiment.score("Senator slams the deal and threatens a probe", "BA") < 0
