"""Robust politician name matching for the news pipeline (FR-POL-1).

Identify which REAL member of Congress (out of ~540) a piece of free text refers to.
The naive longest-name-first substring scan in
:mod:`trading.infrastructure.politicians.repository` is too crude at that scale: bare
surnames collide ("Johnson" appears many times in the roster *and* as an ordinary English
word), producing false positives and arbitrary guesses.

This module is a PURE, deterministic replacement. It matches in two ways, from most to
least specific:

1. **Full name** — the member's official full name on word boundaries, case-insensitively
   (e.g. ``"Elizabeth Warren"``).
2. **Honorific + surname** — an honorific (``Sen.``/``Senator``/``Rep.``/
   ``Representative``/``Congressman``/``Congresswoman``/``Congressmember``) immediately
   followed by a member's surname, but ONLY when that surname is UNAMBIGUOUS in the roster.

A bare surname with neither an honorific nor a first name never matches, so a common word
that happens to equal a surname (e.g. "the long johnson rod") is ignored. When several
candidates match, the longest (most word tokens) wins; ties break deterministically on a
stable sort key so identical input always yields identical output.

No I/O, no network, no mutation of inputs, full type annotations, no ``Any``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from trading.domain.politicians import Politician

# Honorifics that license a surname-only reference. Stored lowercased; a trailing period in
# the source text is stripped before lookup, so "sen." and "sen" both resolve here.
_HONORIFICS: frozenset[str] = frozenset(
    {
        "sen",
        "senator",
        "rep",
        "representative",
        "congressman",
        "congresswoman",
        "congressmember",
    }
)

# A token is a run of letters, digits, apostrophes or hyphens (so "O'Brien" / "Ortiz-Cruz"
# stay intact). Surrounding punctuation/whitespace forms the word boundaries.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens with trailing periods already excluded by the token regex."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(_tokenize(name))


def find_politician(text: str, politicians: Sequence[Politician]) -> Politician | None:
    """Return the member of Congress mentioned in ``text``, or ``None`` (FR-POL-1).

    Matching rules, in priority order:

    * **Full-name match** — a member's full official name appearing on word boundaries,
      case-insensitively.
    * **Honorific + surname** — an honorific token directly followed by a member's surname,
      *only* when that surname is unique across ``politicians``.

    A bare surname (no honorific, no accompanying first name) is deliberately ignored to
    avoid false positives from ordinary words. The most specific match wins (most name
    tokens), with a deterministic tie-break on ``(id, name)`` so the result is stable.
    """
    tokens = _tokenize(text)
    if not tokens:
        return None

    # Index full-name token sequences and surnames. A surname is "ambiguous" when more than
    # one *distinct* member shares it (the same member listed twice is not a collision).
    full_name_index: dict[tuple[str, ...], list[Politician]] = defaultdict(list)
    surname_index: dict[str, set[str]] = defaultdict(set)
    surname_owner: dict[str, Politician] = {}
    for politician in politicians:
        name_tokens = _name_tokens(politician.name)
        if not name_tokens:
            continue
        full_name_index[name_tokens].append(politician)
        surname = name_tokens[-1]
        surname_index[surname].add(politician.id)
        surname_owner[surname] = politician

    token_set = set(tokens)  # fast membership test for the sliding-window scans
    n = len(tokens)

    # Each match yields a (rank, owner) candidate; `min` on the rank picks the winner. Rank
    # is `(-kind, -span, id, name)`: full-name (kind 1) beats honorific+surname (kind 0);
    # then a longer span (more specific) wins; then ties break on the lexicographically
    # earliest `(id, name)`. Negating kind/span makes "better" the smaller rank.
    candidates: list[tuple[tuple[int, int, str, str], Politician]] = []

    # 1) Full-name matches (most specific). Only scan windows whose first token actually
    #    appears in the text, keeping this near-linear for long passages.
    for name_tokens, owners in full_name_index.items():
        span = len(name_tokens)
        if span > n or name_tokens[0] not in token_set:
            continue
        if any(tuple(tokens[s : s + span]) == name_tokens for s in range(n - span + 1)):
            for owner in owners:
                candidates.append(((-1, -span, owner.id, owner.name), owner))

    # 2) Honorific + unambiguous surname. A surname with no honorific and no first name is
    #    never matched here, so a common word equal to a surname cannot trigger a hit.
    for i in range(n - 1):
        if tokens[i] not in _HONORIFICS:
            continue
        surname = tokens[i + 1]
        if len(surname_index.get(surname, ())) != 1:
            continue  # absent or ambiguous surname: refuse to guess
        owner = surname_owner[surname]
        candidates.append(((0, -2, owner.id, owner.name), owner))

    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])[1]
