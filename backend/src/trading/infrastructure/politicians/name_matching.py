"""Robust politician name matching for the news pipeline (FR-POL-1) and filer-string resolution
(Task 07, docs/RESEARCH-PLAN.md).

Identify which REAL member of Congress (out of ~540) a piece of text refers to. Two distinct
matching problems live here, deliberately kept SEPARATE:

1. :func:`find_politician` — FREE TEXT (news articles, disclosed-trade filer names as they were
   originally used). The naive longest-name-first substring scan in
   :mod:`trading.infrastructure.politicians.repository` is too crude at ~540-member scale: bare
   surnames collide ("Johnson" appears many times in the roster *and* as an ordinary English
   word), producing false positives and arbitrary guesses. This function matches in two ways,
   from most to least specific:

   * **Full name** — the member's official full name on word boundaries, case-insensitively.
   * **Honorific + surname** — an honorific immediately followed by a member's surname, but ONLY
     when that surname is UNAMBIGUOUS in the roster.

   A bare surname with neither an honorific nor a first name never matches. When several
   candidates match, the longest (most word tokens) wins; ties break deterministically.

2. :func:`resolve_filer` — STRUCTURED FILER STRINGS from the House Clerk's disclosure index
   (``"Hon. Richard W. Allen"``, ``"Rob Bresnahan Jr"``, ``"Marjorie Taylor Mrs Greene"``, ...).
   These are short "Prefix First [Middle] Last [Suffix]"-shaped strings, not prose, so a
   different, PRECISION-FIRST algorithm applies: normalize both the filer string and every
   roster name (Unicode NFKD accent stripping, suffix tolerance, courtesy-title stripping
   anywhere in the string), then require the normalized (first OR nickname) token AND the
   normalized last-name token to both be present, ignoring any middle tokens. A filer string
   must resolve to EXACTLY ONE roster entry: zero or multiple surviving candidates is a
   deliberate non-match (see :class:`UnmatchedReason`) — fuzzy/edit-distance matching is NOT
   used here, because a false positive would attribute a non-member's PTR to a member, which is
   worse than a miss.

No I/O, no network, no mutation of inputs, full type annotations, no ``Any``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from trading.domain.politicians import Politician

# Honorifics that license a surname-only reference in `find_politician`, or that may appear
# anywhere in a `resolve_filer` filer string (stripped before matching). Stored lowercased; a
# trailing period in the source text is stripped before lookup, so "sen."/"sen" and "hon."/"hon"
# both resolve here.
_HONORIFICS: frozenset[str] = frozenset(
    {
        "sen",
        "senator",
        "rep",
        "representative",
        "congressman",
        "congresswoman",
        "congressmember",
        "hon",
    }
)

# Name suffixes tolerated (with or without punctuation) in `resolve_filer` — never significant
# for identity, so stripped from both roster names and filer strings before comparison.
_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv"})

# Courtesy titles that can appear ANYWHERE in a filer string (not just as a leading prefix, e.g.
# the Clerk's "Marjorie Taylor Mrs Greene" embeds one mid-string) — stripped wherever they occur.
_COURTESY_TITLES: frozenset[str] = frozenset({"mr", "mrs", "ms", "dr"})

# Tokens that carry no identity information in a filer string: honorifics, courtesy titles, and
# suffixes are all stripped the same way regardless of position.
_FILER_NOISE_TOKENS: frozenset[str] = _HONORIFICS | _COURTESY_TITLES | _SUFFIXES

# A token is a run of letters, digits, apostrophes or hyphens (so "O'Brien" / "Ortiz-Cruz"
# stay intact). Surrounding punctuation/whitespace forms the word boundaries.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

UnmatchedReason = Literal["no_candidate", "ambiguous"]


def _strip_accents(text: str) -> str:
    """Unicode NFKD accent stripping: "Sánchez" -> "Sanchez" (Task 07)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    """Lowercased, accent-stripped word tokens with trailing periods excluded by the token regex.

    Accents are stripped BEFORE tokenizing (not per-match) so an accented letter does not act as
    a word-boundary character — e.g. "Sánchez" must tokenize as one token ("sanchez"), not split
    into "s" / "nchez" at the "á".
    """
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(_strip_accents(text))]


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


def _filer_identity_tokens(text: str) -> tuple[str, ...]:
    """Tokenize a filer/roster name, dropping honorifics/courtesy-titles/suffixes anywhere.

    Accent-stripped and lowercased (via :func:`_tokenize`), then noise tokens are filtered out
    regardless of position — a Clerk filer string can embed a courtesy title mid-string
    ("Marjorie Taylor Mrs Greene"), not just as a leading prefix.
    """
    return tuple(token for token in _tokenize(text) if token not in _FILER_NOISE_TOKENS)


class _RosterCandidate:
    """A roster entry's normalized identity: acceptable first-name tokens + one last-name token."""

    __slots__ = ("first_tokens", "last_token", "politician")

    def __init__(
        self, politician: Politician, first_tokens: frozenset[str], last_token: str
    ) -> None:
        self.politician = politician
        self.first_tokens = first_tokens
        self.last_token = last_token


def _roster_candidate(politician: Politician) -> _RosterCandidate | None:
    """Build a :class:`_RosterCandidate` from a roster entry's official name (+ nickname).

    The LAST identity token of the normalized official name is treated as the surname; the
    FIRST identity token is treated as the given name. Any tokens in between are middle
    names/initials and are intentionally ignored (Task 07). The nickname (if present) is
    normalized the same way and added as an additional acceptable "first name" token.
    """
    identity = _filer_identity_tokens(politician.name)
    if not identity:
        return None
    first_tokens = {identity[0]}
    if politician.nickname:
        nickname_tokens = _filer_identity_tokens(politician.nickname)
        if nickname_tokens:
            # A nickname is itself a given name (e.g. "Rick"), never multi-token; use the
            # whole normalized nickname as one more acceptable first-name token.
            first_tokens.add(" ".join(nickname_tokens))
    last_token = identity[-1]
    return _RosterCandidate(politician, frozenset(first_tokens), last_token)


def resolve_filer(
    text: str, politicians: Sequence[Politician]
) -> tuple[Politician | None, UnmatchedReason | None]:
    """Resolve a House Clerk filer string to EXACTLY ONE roster entry, or refuse (Task 07).

    Normalizes both ``text`` and every roster name/nickname (accent stripping, honorific /
    courtesy-title / suffix removal anywhere in the string), then matches on normalized
    ``(first OR nickname) + last``, ignoring any middle tokens. Returns
    ``(politician, None)`` on exactly one surviving candidate, or ``(None, reason)`` where
    ``reason`` is ``"no_candidate"`` (zero matches) or ``"ambiguous"`` (more than one distinct
    politician survives normalization) — the caller (``house_disclosures.py``) increments an
    ``unmatched_filers`` counter keyed by this reason.

    Deliberately NOT fuzzy: no edit-distance/similarity scoring. A false positive would
    attribute a non-member's PTR to the wrong member, which is worse than a miss.
    """
    filer_tokens = _filer_identity_tokens(text)
    if len(filer_tokens) < 2:
        return None, "no_candidate"
    filer_first, filer_last = filer_tokens[0], filer_tokens[-1]

    matches: dict[str, Politician] = {}
    for politician in politicians:
        candidate = _roster_candidate(politician)
        if candidate is None:
            continue
        if candidate.last_token != filer_last:
            continue
        if filer_first not in candidate.first_tokens:
            continue
        matches[politician.id] = politician

    if not matches:
        return None, "no_candidate"
    if len(matches) > 1:
        return None, "ambiguous"
    return next(iter(matches.values())), None
