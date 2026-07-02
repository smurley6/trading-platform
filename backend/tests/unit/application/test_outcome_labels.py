"""Outcome labeler unit tests (Task 03, docs/RESEARCH-PLAN.md §2-§3).

Fixture daily bars + FixedClock, no real I/O. Covers the acceptance criteria:
  * hand-computed abnormal returns match to 1e-9;
  * an after-close disclosure resolves to the NEXT session's bar — and an intra-session
    disclosure on the SAME date resolves to that SAME next-day entry (never same-day);
  * SYNTHETIC input -> raises;
  * unresolvable statuses (no bars / insufficient forward bars) carry a reason;
  * benchmark fallback to SPY ("market") for an unmapped-sector ticker;
  * a rerun replaces rather than duplicates (port-level).
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from tests.conftest import FixedClock
from trading.application.data_integrity import SyntheticDataError
from trading.application.outcome_labels import HORIZONS, LabelOutcomes
from trading.application.ports.read_model import PortfolioSnapshotRecord, SignalRecord
from trading.domain.events import EventType
from trading.domain.positions import HeldPosition
from trading.domain.prices import PriceSeries
from trading.domain.provenance import Provenance
from trading.domain.sectors import Sector
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.memory import InMemoryEventStore
from trading.infrastructure.persistence.memory_outcome_labels import InMemoryOutcomeLabels

# A late-2025 clock so 2025 disclosures are old enough for the 20-day horizon to resolve against
# the fixture bars below.
_NOW = datetime(2025, 12, 1, 12, 0, tzinfo=UTC)


class _FakePrices:
    """A `HistoricalPricesPort` returning a fixed PriceSeries, ignoring the requested window."""

    def __init__(self, series: PriceSeries) -> None:
        self._series = series

    async def price_series(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> PriceSeries:
        # Return only the requested symbols (mirrors the real adapter: unknown symbols absent).
        wanted = {s.strip().upper() for s in symbols}
        return PriceSeries(
            series={sym: pts for sym, pts in self._series.series.items() if sym in wanted}
        )


class _FakeSectorMap:
    def __init__(self, mapping: dict[str, Sector]) -> None:
        self._mapping = mapping

    def sector_for(self, ticker: str) -> Sector | None:
        return self._mapping.get(ticker.upper())


def _bars(*closes: tuple[str, str]) -> tuple[tuple[datetime, Decimal], ...]:
    """Build a daily bar series from (ISO-date, close) pairs."""
    return tuple((datetime.fromisoformat(f"{d}T00:00:00+00:00"), Decimal(c)) for d, c in closes)


def _signal(
    *,
    signal_id: str,
    ticker: str | None,
    disclosed_at: datetime,
    provenance: Provenance = Provenance.BACKFILLED,
) -> Signal:
    return Signal(
        id=signal_id,
        created_at=disclosed_at,
        source_type=SignalSource.DISCLOSED_TRADE,
        ticker=ticker,
        politician_id=None,
        sentiment=0.5,
        politician_weight=1.0,
        source_reliability=0.75,
        raw_payload={"source_uid": signal_id},
        provenance=provenance,
    )


def _labeler(
    prices: PriceSeries, sectors: dict[str, Sector]
) -> tuple[LabelOutcomes, InMemoryOutcomeLabels, InMemoryEventStore, FixedClock]:
    clock = FixedClock(_NOW)
    store = InMemoryEventStore(clock)
    labels = InMemoryOutcomeLabels()
    labeler = LabelOutcomes(
        read_model=_UnusedReadModel(),
        prices=_FakePrices(prices),
        sector_map=_FakeSectorMap(sectors),
        labels_store=labels,
        event_store=store,
        clock=clock,
    )
    return labeler, labels, store, clock


class _UnusedReadModel:
    """A ReadModelPort stand-in; the unit tests always pass an explicit `signals` batch."""

    async def held_positions(self) -> Sequence[HeldPosition]:  # pragma: no cover - unused
        return []

    async def order_statuses(self, *, sleeve_id: str | None = None) -> dict[str, str]:
        return {}  # pragma: no cover - unused

    async def signals_in_window(
        self, *, now: datetime, window_days: int
    ) -> Sequence[Signal]:  # pragma: no cover - unused
        return []

    async def recent_signals(self, *, limit: int) -> Sequence[SignalRecord]:  # pragma: no cover
        return []

    async def all_signals(self) -> Sequence[Signal]:  # pragma: no cover - unused
        return []

    async def latest_portfolio_snapshots(
        self,
    ) -> Sequence[PortfolioSnapshotRecord]:  # pragma: no cover
        return []


async def test_hand_computed_abnormal_returns_match_to_1e_9() -> None:
    # Ticker AAPL (tech -> XLK proxy). Disclosure 2025-01-01 -> entry = first bar after
    # 2025-01-01 = 2025-01-02 (index 0 here since the series starts on the 2nd).
    ticker_bars = _bars(
        ("2025-01-02", "100"),  # entry (index 0)
        ("2025-01-03", "101"),  # h=1
        ("2025-01-04", "102"),
        ("2025-01-05", "103"),
        ("2025-01-06", "104"),
        ("2025-01-07", "105"),  # h=5
    )
    xlk_bars = _bars(
        ("2025-01-02", "50"),  # benchmark entry (date >= 2025-01-02)
        ("2025-01-03", "50.5"),  # h=1
        ("2025-01-04", "51"),
        ("2025-01-05", "51.5"),
        ("2025-01-06", "52"),
        ("2025-01-07", "52.5"),  # h=5
    )
    prices = PriceSeries(series={"AAPL": ticker_bars, "XLK": xlk_bars})
    labeler, labels, _store, _clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    signal = _signal(
        signal_id="s1", ticker="AAPL", disclosed_at=datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
    )

    run_id = str(uuid4())
    report = await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])

    # Reach into the store to assert exact values.
    stored = labels._by_run[run_id]
    by_h = {label.horizon_days: label for label in stored}

    # h=1: ticker 101/100-1 = 0.01 ; bench 50.5/50-1 = 0.01 ; abnormal = 0.0
    assert by_h[1].status == "ok"
    assert by_h[1].benchmark == "sector:XLK"
    assert by_h[1].abnormal_return == pytest.approx(0.0, abs=1e-9)
    # h=5: ticker 105/100-1 = 0.05 ; bench 52.5/50-1 = 0.05 ; abnormal = 0.0
    assert by_h[5].abnormal_return == pytest.approx(0.0, abs=1e-9)
    # h=20: not enough forward bars -> unresolvable with reason.
    assert by_h[20].status == "unresolvable"
    assert by_h[20].reason == "insufficient_forward_bars"

    assert report.signal_count == 1
    assert report.labels == len(HORIZONS)
    assert report.ok == 2
    assert report.unresolvable == 1
    assert report.committed is True


async def test_nonzero_abnormal_return_hand_computed() -> None:
    # Ticker up 10% at h=1, benchmark up 2% -> abnormal = 0.10 - 0.02 = 0.08.
    ticker_bars = _bars(("2025-03-04", "200"), ("2025-03-05", "220"))
    xlk_bars = _bars(("2025-03-04", "100"), ("2025-03-05", "102"))
    prices = PriceSeries(series={"AAPL": ticker_bars, "XLK": xlk_bars})
    labeler, labels, _store, _clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    signal = _signal(
        signal_id="s1", ticker="AAPL", disclosed_at=datetime(2025, 3, 3, 15, 0, tzinfo=UTC)
    )

    run_id = str(uuid4())
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    stored = {label.horizon_days: label for label in labels._by_run[run_id]}
    assert stored[1].abnormal_return == pytest.approx(0.08, abs=1e-9)


async def test_after_close_and_intra_session_resolve_to_same_next_day_entry() -> None:
    # Two disclosures on the SAME calendar date (2025-06-10): one intra-session (14:00 UTC), one
    # after-close (21:30 UTC). BOTH must resolve to the SAME next-session entry (2025-06-11), and
    # NEVER to the same-day 2025-06-10 bar. The abnormal returns are therefore identical.
    ticker_bars = _bars(
        ("2025-06-10", "999"),  # same-day bar — must NOT be the entry
        ("2025-06-11", "100"),  # entry for both disclosures
        ("2025-06-12", "110"),  # h=1
    )
    spy_bars = _bars(
        ("2025-06-10", "500"),
        ("2025-06-11", "200"),
        ("2025-06-12", "202"),
    )
    prices = PriceSeries(series={"NVDA": ticker_bars, "SPY": spy_bars})
    labeler, labels, _store, _clock = _labeler(prices, {})  # unmapped -> SPY/market
    intra = _signal(
        signal_id="intra", ticker="NVDA", disclosed_at=datetime(2025, 6, 10, 14, 0, tzinfo=UTC)
    )
    after_close = _signal(
        signal_id="after", ticker="NVDA", disclosed_at=datetime(2025, 6, 10, 21, 30, tzinfo=UTC)
    )

    run_id = str(uuid4())
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[intra, after_close])
    by_signal_h = {(label.signal_id, label.horizon_days): label for label in labels._by_run[run_id]}

    # Entry = 100 (2025-06-11), NOT 999 (same-day). h=1 ticker 110/100-1=0.10; SPY 202/200-1=0.01.
    for sig_id in ("intra", "after"):
        label = by_signal_h[(sig_id, 1)]
        assert label.status == "ok"
        assert label.benchmark == "market"
        assert label.abnormal_return == pytest.approx(0.10 - 0.01, abs=1e-9)
    # Identical entry -> identical abnormal return for both disclosures on the same date.
    assert by_signal_h[("intra", 1)].abnormal_return == by_signal_h[("after", 1)].abnormal_return


async def test_synthetic_input_raises() -> None:
    prices = PriceSeries(series={"AAPL": _bars(("2025-01-02", "100"), ("2025-01-03", "101"))})
    labeler, _labels, _store, _clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    synthetic = _signal(
        signal_id="fake",
        ticker="AAPL",
        disclosed_at=datetime(2025, 1, 1, tzinfo=UTC),
        provenance=Provenance.SYNTHETIC,
    )
    with pytest.raises(SyntheticDataError):
        await labeler.execute(label_run_id=str(uuid4()), dry_run=False, signals=[synthetic])


async def test_no_bars_is_unresolvable_with_reason() -> None:
    # The ticker has no bars at all (delisted / no data) -> every horizon unresolvable "no_bars".
    prices = PriceSeries(series={"SPY": _bars(("2025-01-02", "100"), ("2025-01-03", "101"))})
    labeler, labels, _store, _clock = _labeler(prices, {})
    signal = _signal(
        signal_id="s1", ticker="DELISTED", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC)
    )
    run_id = str(uuid4())
    report = await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    stored = labels._by_run[run_id]
    assert len(stored) == len(HORIZONS)
    assert all(label.status == "unresolvable" for label in stored)
    assert all(label.reason == "no_bars" for label in stored)
    assert report.ok == 0


async def test_too_recent_signal_is_unresolvable() -> None:
    # Disclosure AFTER the last bar -> no forward session -> insufficient_forward_bars.
    prices = PriceSeries(
        series={
            "AAPL": _bars(("2025-01-02", "100"), ("2025-01-03", "101")),
            "XLK": _bars(("2025-01-02", "50"), ("2025-01-03", "50.5")),
        }
    )
    labeler, labels, _store, _clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    signal = _signal(signal_id="s1", ticker="AAPL", disclosed_at=datetime(2025, 6, 1, tzinfo=UTC))
    run_id = str(uuid4())
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    stored = labels._by_run[run_id]
    assert all(label.status == "unresolvable" for label in stored)
    assert all(label.reason == "insufficient_forward_bars" for label in stored)


async def test_benchmark_fallback_to_spy_for_unmapped_sector() -> None:
    # Unmapped-sector ticker (not in the sector map) -> SPY, benchmark="market".
    ticker_bars = _bars(("2025-01-02", "100"), ("2025-01-03", "110"))
    spy_bars = _bars(("2025-01-02", "400"), ("2025-01-03", "404"))
    prices = PriceSeries(series={"RANDO": ticker_bars, "SPY": spy_bars})
    labeler, labels, _store, _clock = _labeler(prices, {})
    signal = _signal(signal_id="s1", ticker="RANDO", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC))
    run_id = str(uuid4())
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    label = {la.horizon_days: la for la in labels._by_run[run_id]}[1]
    assert label.benchmark == "market"
    # ticker 110/100-1 = 0.10 ; SPY 404/400-1 = 0.01 ; abnormal = 0.09.
    assert label.abnormal_return == pytest.approx(0.09, abs=1e-9)


async def test_sector_etf_without_bars_falls_back_to_spy() -> None:
    # Ticker mapped to DEFENSE (ITA), but ITA has no bars -> fall back to SPY/market.
    ticker_bars = _bars(("2025-01-02", "100"), ("2025-01-03", "105"))
    spy_bars = _bars(("2025-01-02", "300"), ("2025-01-03", "303"))
    prices = PriceSeries(series={"LMT": ticker_bars, "SPY": spy_bars})  # no ITA
    labeler, labels, _store, _clock = _labeler(prices, {"LMT": Sector.DEFENSE})
    signal = _signal(signal_id="s1", ticker="LMT", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC))
    run_id = str(uuid4())
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    label = {la.horizon_days: la for la in labels._by_run[run_id]}[1]
    assert label.benchmark == "market"


async def test_signals_without_ticker_are_skipped() -> None:
    prices = PriceSeries(series={"SPY": _bars(("2025-01-02", "100"), ("2025-01-03", "101"))})
    labeler, labels, _store, _clock = _labeler(prices, {})
    no_ticker = _signal(signal_id="s1", ticker="   ", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC))
    run_id = str(uuid4())
    report = await labeler.execute(label_run_id=run_id, dry_run=False, signals=[no_ticker])
    assert report.signal_count == 0
    assert report.labels == 0
    assert labels._by_run[run_id] == []


async def test_signals_with_none_ticker_are_skipped() -> None:
    """Task 04b: an unresolved-entity backfilled signal (ticker=None) must be SKIPPED (not
    crash on `.strip()`), same as the existing empty-string ticker case above."""
    prices = PriceSeries(series={"SPY": _bars(("2025-01-02", "100"), ("2025-01-03", "101"))})
    labeler, labels, _store, _clock = _labeler(prices, {})
    none_ticker = _signal(
        signal_id="s1", ticker=None, disclosed_at=datetime(2025, 1, 1, tzinfo=UTC)
    )
    run_id = str(uuid4())
    report = await labeler.execute(label_run_id=run_id, dry_run=False, signals=[none_ticker])
    assert report.signal_count == 0
    assert report.labels == 0
    assert labels._by_run[run_id] == []


async def test_dry_run_writes_nothing_and_appends_no_audit() -> None:
    prices = PriceSeries(
        series={
            "AAPL": _bars(("2025-01-02", "100"), ("2025-01-03", "101")),
            "XLK": _bars(("2025-01-02", "50"), ("2025-01-03", "50.5")),
        }
    )
    labeler, labels, store, _clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    signal = _signal(signal_id="s1", ticker="AAPL", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC))
    run_id = str(uuid4())
    report = await labeler.execute(label_run_id=run_id, dry_run=True, signals=[signal])
    assert report.committed is False
    assert await labels.count_for_run(run_id) == 0
    events = await store.read_all()
    assert not [e for e in events if e.event.event_type == EventType.USER_ACTION]


async def test_confirm_appends_single_user_action_audit_event() -> None:
    prices = PriceSeries(
        series={
            "AAPL": _bars(("2025-01-02", "100"), ("2025-01-03", "101")),
            "XLK": _bars(("2025-01-02", "50"), ("2025-01-03", "50.5")),
        }
    )
    labeler, _labels, store, clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    signal = _signal(signal_id="s1", ticker="AAPL", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC))
    run_id = str(uuid4())
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    actions = [
        e.event for e in await store.read_all() if e.event.event_type == EventType.USER_ACTION
    ]
    assert len(actions) == 1
    payload = actions[0].payload
    assert payload["action"] == "label_outcomes"
    assert payload["label_run_id"] == run_id
    assert payload["horizons"] == [1, 5, 20]
    assert payload["provenance_filter"] == ["backfilled", "real"]
    assert payload["signal_count"] == 1
    # Audit event occurred_at is the run wall-clock (clock.now()), not a disclosure timestamp.
    assert actions[0].occurred_at == clock.now()


async def test_rerun_replaces_rather_than_duplicates() -> None:
    # Port-level rebuild semantics: re-running the SAME run id overwrites its rows.
    prices = PriceSeries(
        series={
            "AAPL": _bars(("2025-01-02", "100"), ("2025-01-03", "101")),
            "XLK": _bars(("2025-01-02", "50"), ("2025-01-03", "50.5")),
        }
    )
    labeler, labels, _store, _clock = _labeler(prices, {"AAPL": Sector.TECHNOLOGY})
    signal = _signal(signal_id="s1", ticker="AAPL", disclosed_at=datetime(2025, 1, 1, tzinfo=UTC))
    run_id = str(uuid4())

    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    count_1 = await labels.count_for_run(run_id)
    await labeler.execute(label_run_id=run_id, dry_run=False, signals=[signal])
    count_2 = await labels.count_for_run(run_id)

    assert count_1 == count_2 == len(HORIZONS)
