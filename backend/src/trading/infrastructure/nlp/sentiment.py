"""Lexicon sentiment (FR-POL-2): stance of a statement toward a ticker, in [-1, 1].

Deterministic bag-of-words polarity with simple negation handling. M2 applies whole-text
polarity to each resolved ticker; a ticker-windowed or model-backed scorer can replace
this behind `SentimentAnalyzerPort` without changing callers.
"""

import re

_POSITIVE = frozenset(
    {
        "support",
        "supports",
        "boost",
        "boosts",
        "approve",
        "approved",
        "fund",
        "funds",
        "funding",
        "award",
        "awarded",
        "win",
        "wins",
        "strong",
        "growth",
        "praise",
        "praised",
        "invest",
        "investment",
        "expand",
        "breakthrough",
        "advance",
        "champion",
        "backs",
        "back",
        "endorse",
        "endorsed",
        "favorable",
        "surge",
    }
)
_NEGATIVE = frozenset(
    {
        "oppose",
        "opposes",
        "ban",
        "bans",
        "banned",
        "investigate",
        "investigation",
        "probe",
        "sanction",
        "sanctions",
        "cut",
        "cuts",
        "weak",
        "decline",
        "criticize",
        "criticized",
        "criticism",
        "fraud",
        "fine",
        "fined",
        "penalty",
        "lawsuit",
        "sue",
        "sued",
        "concern",
        "concerns",
        "threat",
        "restrict",
        "restriction",
        "scrutiny",
        "halt",
        "block",
        "blocked",
    }
)
_NEGATORS = frozenset({"not", "no", "never", "without", "deny", "denies", "denied"})

_WORD = re.compile(r"[a-z']+")


class LexiconSentiment:
    def score(self, text: str, ticker: str) -> float:
        tokens = _WORD.findall(text.lower())
        score = 0
        negate = False
        for token in tokens:
            if token in _NEGATORS:
                negate = True
                continue
            polarity = 1 if token in _POSITIVE else -1 if token in _NEGATIVE else 0
            if polarity:
                score += -polarity if negate else polarity
            negate = False
        if score == 0:
            return 0.0
        # Squash to [-1, 1]; magnitude grows with corroborating words but saturates.
        return max(-1.0, min(1.0, score / 3.0))
