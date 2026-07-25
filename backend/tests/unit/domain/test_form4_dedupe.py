"""Form 4 episode dedupe (task-11): repeated filings from one filer stream are ONE vote.

Pure-domain coverage of :func:`dedupe_form4_votes` and :class:`Form4Vote`. The application-layer
signal→vote extraction and the one-off cap are covered in
``tests/unit/application/test_form4_corroboration.py``.
"""

from datetime import UTC, datetime, timedelta

import pytest

from trading.domain.conviction import (
    Form4Vote,
    dedupe_form4_votes,
    group_form4_vote_streams,
)

# The LUV 2024-10-02 replay anchor (194 same-day filings, 2 owners).
DAY = datetime(2024, 10, 2, 13, 30, tzinfo=UTC)


def _vote(owner: str, sid: str, *, ticker: str = "LUV", at: datetime = DAY) -> Form4Vote:
    return Form4Vote(ticker=ticker, owner=owner, created_at=at, signal_id=sid)


def test_luv_same_day_two_owners_collapse_to_two_votes() -> None:
    # 194 same-day filings, split across exactly two reporting owners → a 2-owner single-day event.
    votes = [_vote("ALPHA A", f"a{i:03d}") for i in range(97)]
    votes += [_vote("BETA B", f"b{i:03d}") for i in range(97)]
    kept = dedupe_form4_votes(votes)
    assert len(kept) == 2
    assert {v.owner for v in kept} == {"ALPHA A", "BETA B"}


def test_luv_adding_lots_3_to_194_changes_nothing() -> None:
    # The 2-owner event is fully determined by the two earliest (canonical) filings; piling on the
    # remaining ~192 same-day rows must not change which votes survive.
    canonical = [_vote("ALPHA A", "a000"), _vote("BETA B", "b000")]
    lots_3_to_194 = [_vote("ALPHA A", f"a{i:03d}") for i in range(1, 97)]
    lots_3_to_194 += [_vote("BETA B", f"b{i:03d}") for i in range(1, 98)]

    base = dedupe_form4_votes(canonical)
    full = dedupe_form4_votes([*canonical, *lots_3_to_194])
    assert [(v.owner, v.signal_id) for v in base] == [(v.owner, v.signal_id) for v in full]


def test_musk_single_owner_single_day_is_one_vote() -> None:
    # 25 lots, one owner, one day (one filing's transaction rows) → exactly one corroboration vote.
    votes = [_vote("MUSK ELON", f"m{i:02d}", ticker="TSLA") for i in range(25)]
    assert len(dedupe_form4_votes(votes)) == 1


def test_two_distinct_owners_same_ticker_different_days_within_7d_are_two_votes() -> None:
    votes = [
        _vote("ALPHA", "a", at=DAY),
        _vote("BETA", "b", at=DAY + timedelta(days=3)),
    ]
    kept = dedupe_form4_votes(votes)
    assert len(kept) == 2
    assert {v.owner for v in kept} == {"ALPHA", "BETA"}


def test_same_owner_within_stream_window_collapses() -> None:
    votes = [_vote("ALPHA", "a", at=DAY), _vote("ALPHA", "b", at=DAY + timedelta(days=20))]
    assert len(dedupe_form4_votes(votes)) == 1


def test_same_owner_beyond_stream_window_opens_a_new_vote() -> None:
    votes = [_vote("ALPHA", "a", at=DAY), _vote("ALPHA", "b", at=DAY + timedelta(days=31))]
    assert len(dedupe_form4_votes(votes)) == 2


def test_stream_window_is_rolling_not_anchored() -> None:
    # 0 → 20 → 40 days: each gap is within 30d, so the whole chain is ONE stream even though the
    # first and last filings are 40 days apart (the window trails the most-recent filing).
    votes = [
        _vote("ALPHA", "a", at=DAY),
        _vote("ALPHA", "b", at=DAY + timedelta(days=20)),
        _vote("ALPHA", "c", at=DAY + timedelta(days=40)),
    ]
    assert len(dedupe_form4_votes(votes)) == 1


def test_representative_is_the_earliest_filing() -> None:
    kept = dedupe_form4_votes(
        [_vote("ALPHA", "z", at=DAY + timedelta(days=2)), _vote("ALPHA", "a", at=DAY)]
    )
    assert [v.signal_id for v in kept] == ["a"]


def test_output_is_deterministically_ordered() -> None:
    kept = dedupe_form4_votes(
        [_vote("BETA", "b", ticker="TSLA"), _vote("ALPHA", "a", ticker="AAPL")]
    )
    assert [(v.ticker, v.owner) for v in kept] == [("AAPL", "ALPHA"), ("TSLA", "BETA")]


def test_empty_input_yields_no_votes() -> None:
    assert dedupe_form4_votes([]) == []


def test_negative_stream_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="stream_window_days"):
        dedupe_form4_votes([], stream_window_days=-1)


def test_filing_day_is_the_acceptance_date() -> None:
    assert _vote("ALPHA", "a").filing_day == DAY.date()


# --- group_form4_vote_streams (task-13: full stream membership for episode aggregation) --------


def test_group_streams_dedupe_is_the_earliest_of_each_stream() -> None:
    # dedupe_form4_votes MUST equal "take stream[0] of each group_form4_vote_streams stream" —
    # they share one partitioning implementation (task-13 reuse of the task-11 primitive).
    votes = [
        _vote("ALPHA", "a", at=DAY),
        _vote("ALPHA", "b", at=DAY + timedelta(days=20)),
        _vote("ALPHA", "c", at=DAY + timedelta(days=100)),
        _vote("BETA", "z", ticker="TSLA"),
    ]
    streams = group_form4_vote_streams(votes)
    reps = [s[0] for s in streams]
    assert [v.signal_id for v in reps] == [v.signal_id for v in dedupe_form4_votes(votes)]


def test_group_streams_preserves_all_members_in_chronological_order() -> None:
    votes = [
        _vote("ALPHA", "c", at=DAY + timedelta(days=20)),
        _vote("ALPHA", "a", at=DAY),
        _vote("ALPHA", "b", at=DAY + timedelta(days=10)),
    ]
    streams = group_form4_vote_streams(votes)
    assert len(streams) == 1
    assert [v.signal_id for v in streams[0]] == ["a", "b", "c"]


def test_group_streams_none_window_ignores_gaps_one_episode_per_ticker_owner() -> None:
    # stream_window_days=None: the WHOLE filer-on-name relationship is one episode regardless of
    # gaps (the task-13 episode unit) — 30-day splitting would make this two streams.
    votes = [
        _vote("ALPHA", "a", at=DAY),
        _vote("ALPHA", "b", at=DAY + timedelta(days=400)),
    ]
    assert len(group_form4_vote_streams(votes, stream_window_days=30)) == 2
    none_streams = group_form4_vote_streams(votes, stream_window_days=None)
    assert len(none_streams) == 1
    assert [v.signal_id for v in none_streams[0]] == ["a", "b"]


def test_group_streams_distinct_owners_never_merge_even_with_none_window() -> None:
    votes = [_vote("ALPHA", "a"), _vote("BETA", "b")]
    streams = group_form4_vote_streams(votes, stream_window_days=None)
    assert {s[0].owner for s in streams} == {"ALPHA", "BETA"}
    assert all(len(s) == 1 for s in streams)


def test_group_streams_negative_window_rejected() -> None:
    with pytest.raises(ValueError, match="stream_window_days"):
        group_form4_vote_streams([], stream_window_days=-1)
