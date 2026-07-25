"""Form 4 corroboration hygiene (task-11): episode collapse + one-off WATCH gate.

Application-layer coverage: signal→vote extraction, accession + (ticker, owner, day) collapse, the
one-off trailing-7d cap, and that sells / non-Form-4 signals are left untouched. The pure stream
grouping is covered in ``tests/unit/domain/test_form4_dedupe.py``.
"""

from datetime import UTC, datetime, timedelta

from trading.application.form4_corroboration import (
    ONE_OFF_CONVICTION_CAP,
    collapse_form4_signals,
    form4_oneoff_capped_tickers,
)
from trading.domain.conviction import ConvictionThresholds, compute_conviction
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource

DAY = datetime(2024, 10, 2, 13, 30, tzinfo=UTC)
THRESHOLDS = ConvictionThresholds(trade=70.0, watch=40.0)


def _form4(
    sid: str,
    *,
    ticker: str | None = "LUV",
    owner: str = "OWNER A",
    at: datetime = DAY,
    accession: str | None = None,
    sentiment: float = 0.5,
    weight: float = 1.0,
    reliability: float = 0.6,
) -> Signal:
    payload: dict[str, object] = {"owner": owner}
    if accession is not None:
        payload["accession"] = accession
    return Signal(
        id=sid,
        created_at=at,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=None,
        sentiment=sentiment,
        politician_weight=weight,
        source_reliability=reliability,
        raw_payload=payload,
        provenance=Provenance.REAL,
    )


def _non_form4(sid: str, *, ticker: str = "AAPL") -> Signal:
    return Signal(
        id=sid,
        created_at=DAY,
        source_type=SignalSource.STATEMENT,
        ticker=ticker,
        politician_id="pol-1",
        sentiment=0.8,
        politician_weight=1.0,
        source_reliability=0.9,
        raw_payload={"source": "news"},
        provenance=Provenance.REAL,
    )


# --- collapse: LUV replay + Musk single filing -------------------------------------------------


def test_luv_replay_collapses_194_filings_to_two_owner_votes() -> None:
    filings = [_form4(f"a{i:03d}", owner="ALPHA A", accession=f"acc-a-{i}") for i in range(97)]
    filings += [_form4(f"b{i:03d}", owner="BETA B", accession=f"acc-b-{i}") for i in range(97)]
    collapsed = collapse_form4_signals(filings)
    assert len(collapsed) == 2
    assert {s.raw_payload["owner"] for s in collapsed} == {"ALPHA A", "BETA B"}


def test_luv_replay_conviction_ignores_lots_3_to_194() -> None:
    owner_a = _form4("a000", owner="ALPHA A", accession="acc-a-0")
    owner_b = _form4("b000", owner="BETA B", accession="acc-b-0")
    extra = [_form4(f"a{i:03d}", owner="ALPHA A", accession=f"acc-a-{i}") for i in range(1, 97)]
    extra += [_form4(f"b{i:03d}", owner="BETA B", accession=f"acc-b-{i}") for i in range(1, 98)]

    two_owner_event = compute_conviction(
        "LUV", collapse_form4_signals([owner_a, owner_b]), window_days=21, thresholds=THRESHOLDS
    )
    full_replay = compute_conviction(
        "LUV",
        collapse_form4_signals([owner_a, owner_b, *extra]),
        window_days=21,
        thresholds=THRESHOLDS,
    )
    assert full_replay.score == two_owner_event.score
    assert full_replay.decayed_weighted_sum == two_owner_event.decayed_weighted_sum


def test_musk_25_lots_one_accession_is_one_vote() -> None:
    lots = [
        _form4(f"m{i:02d}", ticker="TSLA", owner="MUSK ELON", accession="acc-1") for i in range(25)
    ]
    collapsed = collapse_form4_signals(lots)
    assert len(collapsed) == 1

    single_filing = compute_conviction("TSLA", [lots[0]], window_days=21, thresholds=THRESHOLDS)
    twenty_five = compute_conviction("TSLA", collapsed, window_days=21, thresholds=THRESHOLDS)
    assert twenty_five.score == single_filing.score
    # One owner → one distinct source type → corroboration multiplier unchanged vs a single filing.
    assert twenty_five.corroboration_factor == single_filing.corroboration_factor


def test_collapse_leaves_non_form4_signals_untouched() -> None:
    news = _non_form4("n1")
    buy = _form4("f1", accession="acc-1")
    collapsed = collapse_form4_signals([news, buy])
    assert news in collapsed
    assert len(collapsed) == 2


def test_collapse_leaves_sells_untouched() -> None:
    # Two sells from one owner/day: benched (reliability 0.0) and NEGATIVE sentiment → not buys, so
    # neither is collapsed. Both survive verbatim.
    sells = [
        _form4("s1", sentiment=-0.5, reliability=0.0, accession="acc-1"),
        _form4("s2", sentiment=-0.5, reliability=0.0, accession="acc-2"),
    ]
    assert collapse_form4_signals(sells) == sells


def test_collapse_keeps_ticker_less_form4_buys_visible() -> None:
    unresolved = _form4("u1", ticker=None, accession="acc-1")
    collapsed = collapse_form4_signals([unresolved])
    assert collapsed == [unresolved]


# --- one-off WATCH gate -------------------------------------------------------------------------


def test_solo_one_off_is_capped() -> None:
    solo = _form4("f1", accession="acc-1")
    assert form4_oneoff_capped_tickers([solo], now=DAY) == frozenset({"LUV"})


def test_second_owner_filing_at_day_plus_5_lifts_the_cap() -> None:
    first = _form4("f1", owner="ALPHA", at=DAY, accession="acc-1")
    second = _form4("f2", owner="BETA", at=DAY + timedelta(days=5), accession="acc-2")
    # Re-evaluated at day+5: trailing 7d holds both filings → two votes → cap lifted.
    assert form4_oneoff_capped_tickers([first, second], now=DAY + timedelta(days=5)) == frozenset()


def test_second_owner_filing_at_day_plus_40_does_not_lift_the_cap() -> None:
    first = _form4("f1", owner="ALPHA", at=DAY, accession="acc-1")
    second = _form4("f2", owner="BETA", at=DAY + timedelta(days=40), accession="acc-2")
    # Re-evaluated at day+40: the day-0 filing has aged out of the trailing 7d → lone vote → capped.
    assert form4_oneoff_capped_tickers([first, second], now=DAY + timedelta(days=40)) == frozenset(
        {"LUV"}
    )


def test_two_owners_within_7d_is_not_capped() -> None:
    signals = [
        _form4("f1", owner="ALPHA", at=DAY, accession="acc-1"),
        _form4("f2", owner="BETA", at=DAY + timedelta(days=3), accession="acc-2"),
    ]
    assert form4_oneoff_capped_tickers(signals, now=DAY + timedelta(days=3)) == frozenset()


def test_single_owner_refiling_does_not_lift_its_own_gate() -> None:
    # Same owner filing twice within the window is still ONE vote — not independent corroboration.
    signals = [
        _form4("f1", owner="ALPHA", at=DAY, accession="acc-1"),
        _form4("f2", owner="ALPHA", at=DAY + timedelta(days=2), accession="acc-2"),
    ]
    assert form4_oneoff_capped_tickers(signals, now=DAY + timedelta(days=2)) == frozenset({"LUV"})


def test_sells_do_not_count_toward_the_gate() -> None:
    signals = [
        _form4("f1", owner="ALPHA", at=DAY, accession="acc-1"),
        _form4("s1", owner="BETA", sentiment=-0.5, reliability=0.0, accession="acc-2"),
    ]
    # The sell is not a buy vote, so the buy remains a lone one-off.
    assert form4_oneoff_capped_tickers(signals, now=DAY) == frozenset({"LUV"})


def test_cap_value_is_below_the_trade_threshold() -> None:
    assert THRESHOLDS.trade > ONE_OFF_CONVICTION_CAP
    assert THRESHOLDS.watch <= ONE_OFF_CONVICTION_CAP
