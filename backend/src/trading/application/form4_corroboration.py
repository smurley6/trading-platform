"""Form 4 corroboration hygiene for the proposals scan (task-11, Phase-7 buy-edge analysis).

Two application-layer guards keep repeated Form 4 filings from masquerading as independent
conviction:

1. **Episode dedupe** — 194 same-day insider rows on a ticker, or 25 lots inside one insider's
   filing, are ONE corroboration vote, not 194 / 25 aligned sources. :func:`collapse_form4_signals`
   collapses Form 4 *buy* signals sharing ``raw_payload['accession']`` (one filing → one row) and
   then, via the pure :func:`trading.domain.conviction.dedupe_form4_votes`, collapses
   ``(ticker, owner, filing-day)`` streams (30-day same-owner window) down to one representative
   signal per distinct filer stream — so downstream conviction counts distinct owners, not rows.

2. **One-off WATCH gate** — a lone insider buy with no *second* same-ticker Form 4 filer vote in
   the trailing 7 days is speculative and must not reach the TRADE tier on formula strength alone.
   :func:`form4_oneoff_capped_tickers` returns the tickers whose conviction the caller must cap at
   :data:`ONE_OFF_CONVICTION_CAP`. A second filer vote landing inside a later evaluation's trailing
   window lifts the cap on re-evaluation.

Both are APPLICATION concerns because they read ``raw_payload`` (owner, accession), which the pure
domain never sees. Only Form 4 *buys* are touched: sells are benched (reliability 0.0) and left
exactly as-is, and non-Form-4 signals pass through untouched. Trailing windows only — the gate
never looks forward.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

from trading.domain.conviction import Form4Vote, dedupe_form4_votes
from trading.domain.signals import Signal, SignalSource

# A lone one-off Form 4 buy is capped just below the TRADE threshold (70.0): high enough to stay
# a WATCH-tier "about to buy" candidate, never high enough to auto-propose a trade on its own.
ONE_OFF_CONVICTION_CAP = 69.0

# Trailing window for the one-off gate: a same-ticker second filer vote within this many days of
# the evaluation "now" lifts the cap.
_ONE_OFF_WINDOW_DAYS = 7


def _is_form4_buy(signal: Signal) -> bool:
    """True for a Form 4 open-market PURCHASE signal.

    Form 4 is the only adapter that stamps ``owner`` into ``raw_payload`` (congress/senate carry a
    ``politician_id`` and no owner; the EDGAR Atom feed carries an accession but no owner), so
    ``owner`` present on a ``DISCLOSED_TRADE`` signal uniquely identifies Form 4. Buys have positive
    sentiment (code ``P`` -> +0.5); sells (code ``S`` -> -0.5, benched) are deliberately excluded so
    the sales path is untouched.
    """
    return (
        signal.source_type is SignalSource.DISCLOSED_TRADE
        and "owner" in signal.raw_payload
        and signal.sentiment > 0.0
    )


def _normalized_owner(signal: Signal) -> str:
    """Normalize the reporting owner name: upper-cased, whitespace-collapsed. No fuzzy matching."""
    return " ".join(str(signal.raw_payload.get("owner", "")).split()).upper()


def _accession(signal: Signal) -> str | None:
    value = signal.raw_payload.get("accession")
    return str(value) if value is not None else None


def _votes(signals: Sequence[Signal]) -> list[Form4Vote]:
    """One :class:`Form4Vote` per ticker-bearing Form 4 buy (unresolved tickers never vote)."""
    return [
        Form4Vote(
            ticker=s.ticker,
            owner=_normalized_owner(s),
            created_at=s.created_at,
            signal_id=s.id,
        )
        for s in signals
        if _is_form4_buy(s) and s.ticker is not None
    ]


def form4_buy_votes(signals: Sequence[Signal]) -> list[Form4Vote]:
    """Public: one :class:`Form4Vote` per ticker-resolved Form 4 *buy* signal (sells/non-Form-4
    excluded, unresolved tickers excluded). The read-only episode/edge evaluator (task-13) uses this
    to build episodes without re-deriving the ``raw_payload`` (owner) parsing that lives here."""
    return _votes(signals)


def collapse_form4_signals(signals: Sequence[Signal]) -> list[Signal]:
    """Collapse repeated Form 4 buy filings so each distinct filer stream contributes one signal.

    Non-Form-4 signals (and Form 4 sells) pass through untouched. Form 4 buys are collapsed in two
    steps: first by shared ``accession`` (the many transaction rows of one filing → one row), then
    by ``(ticker, owner, filing-day)`` stream via the pure domain helper. The surviving signals are
    the earliest representative of each stream; ticker-less Form 4 buys are kept as-is (they never
    reach ticker-grouped scoring anyway). Order is not significant to any downstream consumer.
    """
    passthrough: list[Signal] = []
    form4_buys: list[Signal] = []
    for s in signals:
        (form4_buys if _is_form4_buy(s) else passthrough).append(s)

    if not form4_buys:
        return list(signals)

    # Step 1: one filing (accession) → one representative row. Rows with no accession (the live
    # feed) stay individual; ties resolve deterministically on (created_at, id).
    per_filing: list[Signal] = []
    reps_by_accession: dict[str, Signal] = {}
    for s in form4_buys:
        accession = _accession(s)
        if accession is None:
            per_filing.append(s)
            continue
        current = reps_by_accession.get(accession)
        if current is None or (s.created_at, s.id) < (current.created_at, current.id):
            reps_by_accession[accession] = s
    per_filing.extend(reps_by_accession.values())

    # Step 2: collapse (ticker, owner, day)+30d filer streams to one vote each (pure domain).
    by_id = {s.id: s for s in per_filing}
    kept_votes = dedupe_form4_votes(_votes(per_filing))
    collapsed_buys = [by_id[v.signal_id] for v in kept_votes]
    ticker_less = [s for s in per_filing if s.ticker is None]

    return [*passthrough, *ticker_less, *collapsed_buys]


def form4_oneoff_capped_tickers(
    signals: Sequence[Signal],
    *,
    now: datetime,
    window_days: int = _ONE_OFF_WINDOW_DAYS,
) -> frozenset[str]:
    """Tickers whose Form 4 buy corroboration is a lone one-off in the trailing window.

    A ticker is capped when, within the trailing ``(now - window_days, now]`` window, it has fewer
    than two DISTINCT filer votes (deduped the same way conviction sees them). Counting deduped
    votes — not raw rows — is deliberate: an isolated filer refiling the same position is not
    independent corroboration, so it must not lift its own gate. Trailing window only; a later
    re-evaluation with ``now`` advanced re-derives the set against the new window.
    """
    start = now - timedelta(days=window_days)
    recent = [s for s in signals if start < s.created_at <= now]
    votes = _votes(recent)

    by_ticker: dict[str, list[Form4Vote]] = {}
    for vote in votes:
        by_ticker.setdefault(vote.ticker, []).append(vote)

    return frozenset(
        ticker for ticker, group in by_ticker.items() if len(dedupe_form4_votes(group)) < 2
    )
