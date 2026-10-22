"""BuildRlDataset + the PIT knowledge-time rule (task-rl-02). Stub port, FixedClock, no network.

The load-bearing assumption (docs/RL-SLEEVE-PROGRAM.md §4.2): Alpaca stamps a daily bar at the
session's OPENING boundary (00:00 America/New_York) while the stored value is that session's
CLOSE. These tests pin `known_at` to 16:00 New York + 15 min under BOTH DST regimes and across both
2026 DST switch weekends, and prove a non-opening-boundary stamp fails closed.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from tests.conftest import FixedClock
from trading.application import rl_dataset as rl_dataset_module
from trading.application.rl_dataset import (
    BuildRlDataset,
    RlBar,
    RlDataset,
    RlDatasetError,
    rl_dataset_to_json,
    session_known_at,
)
from trading.domain.prices import PriceSeries


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _naive(text: str) -> datetime:
    """A deliberately NAIVE datetime (the invalid input under test)."""
    return _utc(text).replace(tzinfo=None)


class _StubPrices:
    """HistoricalPricesPort stub returning fixed points; records each request."""

    def __init__(self, series: dict[str, tuple[tuple[datetime, Decimal], ...]]) -> None:
        self._series = series
        self.calls: list[tuple[list[str], datetime, datetime]] = []

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        self.calls.append((list(symbols), start, end))
        return PriceSeries(series={s: self._series[s] for s in symbols if s in self._series})


def _points(*pairs: tuple[str, str]) -> tuple[tuple[datetime, Decimal], ...]:
    return tuple((_utc(ts), Decimal(close)) for ts, close in pairs)


# --- the knowledge-time rule ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("session_ts", "expected_known_at"),
    [
        # The two cases pinned by the program doc (§4.8 Test D) and task-rl-03.
        ("2026-01-02T05:00:00Z", "2026-01-02T21:15:00Z"),  # EST: 00:00-05:00 -> 16:15-05:00
        ("2026-07-02T04:00:00Z", "2026-07-02T20:15:00Z"),  # EDT: 00:00-04:00 -> 16:15-04:00
        # Spring-forward weekend (DST starts Sun 2026-03-08 02:00 local).
        ("2026-03-06T05:00:00Z", "2026-03-06T21:15:00Z"),  # Fri, still EST
        ("2026-03-09T04:00:00Z", "2026-03-09T20:15:00Z"),  # Mon, now EDT
        # Fall-back weekend (DST ends Sun 2026-11-01 02:00 local).
        ("2026-10-30T04:00:00Z", "2026-10-30T20:15:00Z"),  # Fri, still EDT
        ("2026-11-02T05:00:00Z", "2026-11-02T21:15:00Z"),  # Mon, now EST
    ],
)
def test_known_at_is_after_the_session_close_under_both_dst_regimes(
    session_ts: str, expected_known_at: str
) -> None:
    known_at = session_known_at(_utc(session_ts))

    assert known_at == _utc(expected_known_at)
    assert known_at.tzinfo is UTC
    # On every trading day the zone-aware rule equals the program doc's fixed 16h15m offset.
    assert known_at - _utc(session_ts) == timedelta(hours=16, minutes=15)


def test_known_at_accepts_a_non_utc_aware_session_ts() -> None:
    # Same instant as 2026-07-02T04:00Z, expressed at -04:00.
    edt = timezone(timedelta(hours=-4))
    assert session_known_at(datetime(2026, 7, 2, 0, 0, tzinfo=edt)) == _utc("2026-07-02T20:15:00Z")


@pytest.mark.parametrize(
    "session_ts",
    [
        "2026-01-02T21:00:00Z",  # an END-of-session stamp (16:00 EST) — the feared silent switch
        "2026-01-02T00:00:00Z",  # UTC midnight = 19:00 New York the previous evening
        "2026-07-02T05:00:00Z",  # EST offset applied in summer = 01:00 EDT (stale DST handling)
        "2026-01-02T04:00:00Z",  # EDT offset applied in winter = 23:00 EST the previous day
        "2026-01-02T05:00:01Z",  # a second past the boundary
    ],
)
def test_session_ts_not_at_new_york_midnight_fails_closed(session_ts: str) -> None:
    with pytest.raises(RlDatasetError, match="00:00 session-open boundary"):
        session_known_at(_utc(session_ts))


def test_known_at_rejects_naive_session_ts() -> None:
    with pytest.raises(RlDatasetError, match="timezone-aware"):
        session_known_at(_naive("2026-01-02T05:00:00Z"))


# --- the records ------------------------------------------------------------------------------


def test_bar_rejects_naive_datetimes_and_known_at_not_after_session_ts() -> None:
    aware = _utc("2026-01-02T05:00:00Z")
    naive = _naive("2026-01-02T21:15:00Z")
    with pytest.raises(RlDatasetError, match="timezone-aware"):
        RlBar(session_ts=aware, known_at=naive, close=1.0)
    with pytest.raises(RlDatasetError, match="must be after session_ts"):
        RlBar(session_ts=aware, known_at=aware, close=1.0)


@pytest.mark.parametrize("close", [0.0, -1.0, float("nan"), float("inf")])
def test_bar_rejects_non_positive_or_non_finite_close(close: float) -> None:
    with pytest.raises(RlDatasetError, match="finite and positive"):
        RlBar(
            session_ts=_utc("2026-01-02T05:00:00Z"),
            known_at=_utc("2026-01-02T21:15:00Z"),
            close=close,
        )


def _dataset(bars: tuple[RlBar, ...], *, adjustment: str = "raw") -> RlDataset:
    when = _utc("2026-01-10T00:00:00Z")
    return RlDataset(
        symbol="SPY",
        bars=bars,
        adjustment=adjustment,
        source_adapter="Stub",
        start=when,
        end=when,
        generated_at=when,
    )


def _bar(session_ts: str, close: float = 1.0) -> RlBar:
    ts = _utc(session_ts)
    return RlBar(session_ts=ts, known_at=session_known_at(ts), close=close)


def test_dataset_rejects_non_ascending_bars_and_unknown_adjustment() -> None:
    a, b = _bar("2026-01-02T05:00:00Z"), _bar("2026-01-05T05:00:00Z")
    with pytest.raises(RlDatasetError, match="strictly ascending"):
        _dataset((b, a))
    with pytest.raises(RlDatasetError, match="strictly ascending"):
        _dataset((a, a))
    with pytest.raises(RlDatasetError, match="adjustment"):
        _dataset((a,), adjustment="split")


# --- the use case -----------------------------------------------------------------------------

# Thursday 2026-07-02 19:00Z = 15:00 EDT: the 07-02 session is still open.
_INTRADAY = datetime(2026, 7, 2, 19, 0, tzinfo=UTC)


def _spy(*pairs: tuple[str, str]) -> _StubPrices:
    return _StubPrices({"SPY": _points(*pairs)})


async def test_execute_stamps_every_bar_with_its_known_at() -> None:
    prices = _spy(
        ("2026-06-30T04:00:00Z", "600.10"),
        ("2026-07-01T04:00:00Z", "601.25"),
    )
    builder = BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="all")

    dataset = await builder.execute(symbol=" spy ", lookback_days=10)

    assert dataset.symbol == "SPY"
    assert dataset.adjustment == "all"
    assert dataset.source_adapter == "_StubPrices"
    assert [(b.session_ts, b.known_at, b.close) for b in dataset.bars] == [
        (_utc("2026-06-30T04:00:00Z"), _utc("2026-06-30T20:15:00Z"), 600.10),
        (_utc("2026-07-01T04:00:00Z"), _utc("2026-07-01T20:15:00Z"), 601.25),
    ]
    assert all(b.known_at <= dataset.generated_at for b in dataset.bars)


async def test_execute_drops_the_developing_session_bar() -> None:
    prices = _spy(
        ("2026-07-01T04:00:00Z", "601.25"),
        ("2026-07-02T04:00:00Z", "999.99"),  # in-progress session: "close" is not final
    )
    builder = BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="raw")

    dataset = await builder.execute(symbol="SPY", lookback_days=10)

    assert [b.session_ts for b in dataset.bars] == [_utc("2026-07-01T04:00:00Z")]


async def test_execute_keeps_a_bar_exactly_at_its_known_at() -> None:
    prices = _spy(("2026-07-02T04:00:00Z", "602.00"))
    at_known = FixedClock(_utc("2026-07-02T20:15:00Z"))
    just_before = FixedClock(_utc("2026-07-02T20:14:59Z"))

    kept = await BuildRlDataset(prices=prices, clock=at_known, adjustment="raw").execute(
        symbol="SPY", lookback_days=5
    )
    assert len(kept.bars) == 1
    with pytest.raises(RlDatasetError, match="no CLOSED bars"):
        await BuildRlDataset(prices=prices, clock=just_before, adjustment="raw").execute(
            symbol="SPY", lookback_days=5
        )


async def test_execute_requests_window_from_new_york_midnight_to_now() -> None:
    prices = _spy(("2026-07-01T04:00:00Z", "601.25"))
    builder = BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="raw")

    dataset = await builder.execute(symbol="SPY", lookback_days=30)

    assert prices.calls == [(["SPY"], _utc("2026-06-02T04:00:00Z"), _INTRADAY)]
    assert (dataset.start, dataset.end, dataset.generated_at) == (
        _utc("2026-06-02T04:00:00Z"),
        _INTRADAY,
        _INTRADAY,
    )


async def test_execute_fails_closed_on_an_end_of_bar_stamp() -> None:
    prices = _spy(("2026-07-01T20:00:00Z", "601.25"))  # stamped at the close, not the open
    builder = BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="raw")
    with pytest.raises(RlDatasetError, match="00:00 session-open boundary"):
        await builder.execute(symbol="SPY", lookback_days=10)


async def test_execute_raises_on_missing_symbol_or_duplicate_sessions() -> None:
    clock = FixedClock(_INTRADAY)
    with pytest.raises(RlDatasetError, match="no bars returned"):
        await BuildRlDataset(prices=_spy(), clock=clock, adjustment="raw").execute(
            symbol="QQQ", lookback_days=10
        )
    dupes = _spy(("2026-07-01T04:00:00Z", "601.25"), ("2026-07-01T04:00:00Z", "601.30"))
    with pytest.raises(RlDatasetError, match="strictly ascending"):
        await BuildRlDataset(prices=dupes, clock=clock, adjustment="raw").execute(
            symbol="SPY", lookback_days=10
        )


async def test_execute_validates_its_inputs() -> None:
    prices = _spy(("2026-07-01T04:00:00Z", "601.25"))
    with pytest.raises(ValueError, match="adjustment"):
        BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="split")
    builder = BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="raw")
    with pytest.raises(RlDatasetError, match="lookback_days"):
        await builder.execute(symbol="SPY", lookback_days=0)
    with pytest.raises(RlDatasetError, match="symbol"):
        await builder.execute(symbol="  ", lookback_days=10)
    naive_clock = FixedClock(_naive("2026-07-02T19:00:00Z"))
    with pytest.raises(RlDatasetError, match="timezone-aware"):
        await BuildRlDataset(prices=prices, clock=naive_clock, adjustment="raw").execute(
            symbol="SPY", lookback_days=10
        )


# --- serialization ----------------------------------------------------------------------------


async def test_json_is_deterministic_and_round_trips() -> None:
    prices = _spy(
        ("2026-06-30T04:00:00Z", "600.10"),
        ("2026-07-01T04:00:00Z", "601.25"),
    )
    builder = BuildRlDataset(prices=prices, clock=FixedClock(_INTRADAY), adjustment="all")
    first = rl_dataset_to_json(await builder.execute(symbol="SPY", lookback_days=10))
    second = rl_dataset_to_json(await builder.execute(symbol="SPY", lookback_days=10))

    assert first == second
    assert first.endswith(b"\n")
    doc = json.loads(first)
    assert doc["header"] == {
        "schema_version": 2,
        "symbol": "SPY",
        "adjustment": "all",
        "source_adapter": "_StubPrices",
        "start": "2026-06-22T04:00:00Z",
        "end": "2026-07-02T19:00:00Z",
        "n_bars": 2,
        "generated_at": "2026-07-02T19:00:00Z",
        "known_at_rule": rl_dataset_module.KNOWN_AT_RULE,
        "price_basis": rl_dataset_module.PRICE_BASIS["all"],
    }
    assert doc["bars"] == [
        {"session_ts": "2026-06-30T04:00:00Z", "known_at": "2026-06-30T20:15:00Z", "close": 600.1},
        {
            "session_ts": "2026-07-01T04:00:00Z",
            "known_at": "2026-07-01T20:15:00Z",
            "close": 601.25,
        },
    ]


def test_json_of_an_empty_dataset_is_valid() -> None:
    assert json.loads(rl_dataset_to_json(_dataset(())))["bars"] == []


def test_never_calls_price_series_price_at() -> None:
    # PriceSeries.price_at returns the EARLIEST point for a time before the series starts — a
    # silent left-edge forward-look (domain/prices.py). This task reads series[symbol] by position.
    src = Path(rl_dataset_module.__file__).parents[1]
    for module in (src / "application" / "rl_dataset.py", src / "jobs" / "rl_dataset.py"):
        assert ".price_at(" not in module.read_text(encoding="utf-8")


def test_every_adjustment_mode_has_a_price_basis() -> None:
    # The header must always say what the close levels mean (never implicit).
    assert set(rl_dataset_module.PRICE_BASIS) == set(rl_dataset_module.ALLOWED_ADJUSTMENTS)
