"""PIT bar table (task-rl-03; program doc §4.2 and §4.8 Test D). Stdlib only — no optional extras.

The bar's ``session_ts`` is Alpaca's opening-boundary stamp and is NEVER knowledge time; every bar
carries the moment its close became public (``known_at``). The frozen artifact's stored ``known_at``
is consumed and validated against the single rule (``session_known_at``), so the two cannot drift.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from trading.application.rl_dataset import (
    RlBar,
    RlDataset,
    rl_dataset_from_json,
    session_known_at,
)
from trading.domain.prices import PriceSeries
from trading.rl import bars as bars_module
from trading.rl.bars import Bar, BarTable, BarTableError

_FROZEN = Path(__file__).parents[3] / "data" / "rl" / "spy_daily.json"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _series(*pairs: tuple[str, str], symbol: str = "SPY") -> PriceSeries:
    return PriceSeries(series={symbol: tuple((_utc(ts), Decimal(c)) for ts, c in pairs)})


def _weekday_sessions(first: str, count: int) -> list[str]:
    """``count`` consecutive weekday opening-boundary stamps (EDT, 04:00Z) from ``first``."""
    day = _utc(first)
    out: list[str] = []
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat().replace("+00:00", "Z"))
        day += timedelta(days=1)
    return out


def _table(count: int = 20) -> BarTable:
    sessions = _weekday_sessions("2026-06-01T04:00:00Z", count)
    return BarTable.from_price_series(
        _series(*((ts, str(100 + i)) for i, ts in enumerate(sessions))), "SPY"
    )


# --- 1. the knowledge-time rule under both DST regimes ----------------------------------------


@pytest.mark.parametrize(
    ("session_ts", "expected_known_at"),
    [
        ("2026-01-02T05:00:00Z", "2026-01-02T21:15:00Z"),  # EST: 00:00 NY = 05:00Z
        ("2026-07-02T04:00:00Z", "2026-07-02T20:15:00Z"),  # EDT: 00:00 NY = 04:00Z
    ],
)
def test_known_at_is_after_the_session_close_under_both_dst_regimes(
    session_ts: str, expected_known_at: str
) -> None:
    table = BarTable.from_price_series(_series((session_ts, "600.00")), "SPY")

    (bar,) = table.bars
    assert bar.session_ts == _utc(session_ts)
    assert bar.known_at == _utc(expected_known_at)
    # The close (16:00 New York) is strictly before known_at; session_ts is the OPEN boundary.
    assert bar.known_at - bar.session_ts == timedelta(hours=16, minutes=15)


# --- 2. Bar invariants ------------------------------------------------------------------------


def test_bar_rejects_naive_datetimes() -> None:
    aware = _utc("2026-01-02T05:00:00Z")
    naive = aware.replace(tzinfo=None)
    with pytest.raises(BarTableError, match="session_ts must be timezone-aware"):
        Bar(session_ts=naive, close=Decimal(1), known_at=aware + timedelta(hours=17))
    with pytest.raises(BarTableError, match="known_at must be timezone-aware"):
        Bar(session_ts=aware, close=Decimal(1), known_at=naive + timedelta(hours=17))


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(hours=-1)])
def test_bar_rejects_known_at_before_session_ts(offset: timedelta) -> None:
    session_ts = _utc("2026-01-02T05:00:00Z")
    with pytest.raises(BarTableError, match="must be after session_ts"):
        Bar(session_ts=session_ts, close=Decimal(1), known_at=session_ts + offset)


@pytest.mark.parametrize("close", ["0", "-1", "NaN", "Infinity"])
def test_bar_rejects_a_close_that_cannot_be_a_ratio_denominator(close: str) -> None:
    session_ts = _utc("2026-01-02T05:00:00Z")
    with pytest.raises(BarTableError, match="finite and positive"):
        Bar(session_ts=session_ts, close=Decimal(close), known_at=session_known_at(session_ts))


# --- 3. from_price_series ---------------------------------------------------------------------


def test_from_price_series_raises_on_missing_symbol_or_non_monotonic_series() -> None:
    series = _series(("2026-07-01T04:00:00Z", "1"), ("2026-07-02T04:00:00Z", "2"))
    with pytest.raises(BarTableError, match="no bars for symbol 'QQQ'"):
        BarTable.from_price_series(series, "QQQ")
    with pytest.raises(BarTableError, match="no bars for symbol 'SPY'"):
        BarTable.from_price_series(PriceSeries(series={"SPY": ()}), "SPY")

    backwards = _series(("2026-07-02T04:00:00Z", "2"), ("2026-07-01T04:00:00Z", "1"))
    with pytest.raises(BarTableError, match="strictly ascending by session_ts"):
        BarTable.from_price_series(backwards, "SPY")
    duplicate = _series(("2026-07-01T04:00:00Z", "1"), ("2026-07-01T04:00:00Z", "2"))
    with pytest.raises(BarTableError, match="strictly ascending by session_ts"):
        BarTable.from_price_series(duplicate, "SPY")


@pytest.mark.parametrize(
    "session_ts",
    [
        "2026-07-02T20:00:00Z",  # end-of-bar stamp (16:00 NY) — would be a 16h backward error
        "2026-07-02T00:00:00Z",  # UTC midnight
        "2026-01-02T04:00:00Z",  # stale-DST (EDT offset in winter)
    ],
)
def test_from_price_series_fails_closed_on_a_non_opening_boundary_stamp(session_ts: str) -> None:
    with pytest.raises(BarTableError, match="00:00 session-open boundary"):
        BarTable.from_price_series(_series((session_ts, "1")), "SPY")


def test_table_rejects_known_at_that_does_not_ascend() -> None:
    first = _utc("2026-07-01T04:00:00Z")
    second = _utc("2026-07-02T04:00:00Z")
    late = Bar(session_ts=first, close=Decimal(1), known_at=first + timedelta(days=3))
    on_time = Bar(session_ts=second, close=Decimal(1), known_at=session_known_at(second))
    with pytest.raises(BarTableError, match="strictly ascending by known_at"):
        BarTable(symbol="SPY", bars=(late, on_time))
    with pytest.raises(BarTableError, match="symbol must be non-empty"):
        BarTable(symbol="", bars=())


# --- 4. split / slice -------------------------------------------------------------------------


@pytest.mark.parametrize("embargo", [timedelta(0), timedelta(days=3)])
def test_split_embargo_prevents_window_straddle(embargo: timedelta) -> None:
    table = _table(20)
    # Mid-session boundary: bar 10's session opened before `at`, but its close is not public
    # until after `at` — it straddles, so it must land on NEITHER side.
    straddler = table.bars[10]
    at = straddler.session_ts + timedelta(hours=8)
    assert straddler.session_ts < at < straddler.known_at

    train, evaluation = table.split(at, embargo=embargo)

    assert len(train) > 0
    assert len(evaluation) > 0
    assert max(b.known_at for b in train.bars) < min(b.session_ts for b in evaluation.bars)
    assert max(b.known_at for b in train.bars) < at - embargo
    assert {b.session_ts for b in train.bars}.isdisjoint(b.session_ts for b in evaluation.bars)
    assert straddler not in train.bars
    assert straddler not in evaluation.bars
    # Eval is everything from `at` on; nothing after the boundary leaks into train.
    assert evaluation.bars == table.bars[11:]
    assert train.symbol == evaluation.symbol == "SPY"


def test_split_embargo_drops_more_bars_as_it_grows() -> None:
    table = _table(20)
    at = table.bars[10].session_ts

    no_gap, _ = table.split(at, embargo=timedelta(0))
    gapped, _ = table.split(at, embargo=timedelta(days=3))

    assert no_gap.bars == table.bars[:10]
    assert len(gapped) < len(no_gap)
    assert gapped.bars == no_gap.bars[: len(gapped)]


@pytest.mark.parametrize("embargo", [timedelta(0), timedelta(days=3)])
def test_split_train_boundary_is_strict_at_known_at_equal_to_cutoff(embargo: timedelta) -> None:
    # Pins `known_at < at - embargo` (strict): a close that becomes public EXACTLY at the cutoff is
    # not train-side knowledge; one microsecond later it is. `<=` must fail this test.
    table = _table(20)
    edge = table.bars[8]
    at = edge.known_at + embargo

    train, _ = table.split(at, embargo=embargo)
    assert edge not in train.bars
    assert train.bars == table.bars[:8]

    later, _ = table.split(at + timedelta(microseconds=1), embargo=embargo)
    assert edge in later.bars
    assert later.bars == table.bars[:9]


def test_split_rejects_negative_embargo_and_naive_boundary() -> None:
    table = _table(5)
    at = table.bars[2].session_ts
    with pytest.raises(BarTableError, match="non-negative"):
        table.split(at, embargo=timedelta(seconds=-1))
    with pytest.raises(BarTableError, match="timezone-aware"):
        table.split(at.replace(tzinfo=None), embargo=timedelta(0))


def test_slice_known_by_uses_known_at_not_session_ts() -> None:
    table = _table(5)
    bar = table.bars[3]

    # The session has opened (and is even past 16:00 NY) but the close is not yet public.
    assert len(table.slice_known_by(bar.session_ts + timedelta(hours=16))) == 3
    assert len(table.slice_known_by(bar.known_at - timedelta(microseconds=1))) == 3
    assert len(table.slice_known_by(bar.known_at)) == 4
    with pytest.raises(BarTableError, match="timezone-aware"):
        table.slice_known_by(bar.known_at.replace(tzinfo=None))


# --- the frozen artifact: consumed through the task-rl-02 reader, known_at validated ----------


def test_frozen_artifact_loads_with_its_stored_known_at() -> None:
    dataset = rl_dataset_from_json(_FROZEN.read_bytes())

    table = BarTable.from_dataset(dataset)

    assert table.symbol == "SPY"
    assert len(table) == len(dataset.bars) > 1_000
    # Stored values are consumed as-is: closes bit-for-bit, known_at exactly as frozen.
    assert table.closes() == tuple(bar.close for bar in dataset.bars)
    assert [b.known_at for b in table.bars] == [b.known_at for b in dataset.bars]
    assert [b.session_ts for b in table.bars] == [b.session_ts for b in dataset.bars]
    assert all(b.known_at.tzinfo is UTC and b.session_ts.tzinfo is UTC for b in table.bars)


@pytest.mark.parametrize("drift", [timedelta(minutes=-1), timedelta(minutes=1), timedelta(hours=1)])
def test_from_dataset_fails_loudly_on_known_at_drift(drift: timedelta) -> None:
    # An RlDataset whose stored known_at disagrees with session_known_at — e.g. a file frozen under
    # an older rule, or a rule edited without regenerating the file — must not become a BarTable.
    good = [
        RlBar(session_ts=ts, known_at=session_known_at(ts), close=100.0 + i)
        for i, ts in enumerate(_utc(s) for s in _weekday_sessions("2026-06-01T04:00:00Z", 4))
    ]
    good[2] = replace(good[2], known_at=good[2].known_at + drift)
    when = _utc("2026-06-10T00:00:00Z")
    dataset = RlDataset(
        symbol="SPY",
        bars=tuple(good),
        adjustment="all",
        source_adapter="Stub",
        start=when,
        end=when,
        generated_at=when,
    )

    with pytest.raises(BarTableError, match=r"bar 2 .*knowledge-time drift"):
        BarTable.from_dataset(dataset)


def test_closes_are_floats_index_aligned_with_bars() -> None:
    table = _table(3)
    assert table.closes() == (100.0, 101.0, 102.0)
    assert len(table) == 3


def test_pure_modules_never_call_price_series_price_at() -> None:
    # domain/prices.py: price_at returns the EARLIEST point when `when` precedes all points — a
    # silent left-edge forward-look. The PIT modules read series positionally.
    package = Path(bars_module.__file__).parent
    for name in ("bars.py", "costs.py", "rewards.py"):
        assert ".price_at(" not in (package / name).read_text(encoding="utf-8")
