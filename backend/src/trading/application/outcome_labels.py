"""LabelOutcomes use case (Task 03, docs/RESEARCH-PLAN.md §2-§3).

Computes forward event-study abnormal returns for ticker-resolved signals and writes them to a
REBUILDABLE research projection (:class:`~trading.application.ports.outcome_labels.
OutcomeLabelsPort`), never the immutable event log. Labels are DERIVED, not truth.

Correctness invariants enforced here (RESEARCH-PLAN §2):

  * **Anti-lookahead session resolution.** Bars are DAILY closes. A signal's entry bar is the
    FIRST bar whose calendar DATE is STRICTLY AFTER the disclosure timestamp's UTC date — never
    the disclosure's own session, so an intra-day OR after-close disclosure both resolve to the
    NEXT trading session. The horizon-``h`` return is ``close(entry+h) / close(entry) - 1`` by
    trading-day (bar-index) offset within that symbol's series.
  * **Provenance filter {REAL, BACKFILLED}; SYNTHETIC refused.** The use case filters its read
    of the projection to REAL + BACKFILLED, and additionally REFUSES any explicit SYNTHETIC
    input via :func:`~trading.application.data_integrity.ensure_no_synthetic` — synthetic data
    never enters the labeler/learner.
  * **Sector-relative abnormal return, SPY fallback.** ``abnormal = ticker_return -
    benchmark_return`` over the same entry-date + bar-offset window. The benchmark is the sector
    ETF proxy (:data:`SECTOR_ETF`) when the sector map resolves the ticker, else SPY
    (``benchmark="market"``); the ETF-proxy decision is documented in MEMORY.md.
  * **Unresolvable, never dropped.** A signal too recent (or with no/insufficient bars, or a
    missing benchmark window) yields an ``unresolvable`` label WITH a queryable reason.

The use case receives ports (read model, historical prices, sector map, outcome-labels store,
event store, clock) — never concrete adapters; the composition root wires them.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trading.application.data_integrity import ensure_no_synthetic
from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import EventStorePort
from trading.application.ports.historical import HistoricalPricesPort
from trading.application.ports.outcome_labels import OutcomeLabel, OutcomeLabelsPort
from trading.application.ports.read_model import ReadModelPort
from trading.application.ports.reference_data import SectorMapPort
from trading.domain.events import DomainEvent, EventType
from trading.domain.prices import PriceSeries
from trading.domain.provenance import Provenance
from trading.domain.sectors import Sector
from trading.domain.signals import Signal

logger = logging.getLogger(__name__)

_USER_ACTION_STREAM = "user-actions"

# Forward return windows in TRADING days (bar-index offsets), fixed by the spec. One label row
# per (signal, horizon). A signal old enough for 1/5 but not 20 gets "ok" rows for 1/5 and an
# "unresolvable" row for 20.
HORIZONS: tuple[int, ...] = (1, 5, 20)

# The provenance tiers the labeler admits: REAL + BACKFILLED (backfilled data is real,
# historically reconstructed data permitted into the research/labeler path). SYNTHETIC is refused.
LABELABLE_PROVENANCE: frozenset[Provenance] = frozenset({Provenance.REAL, Provenance.BACKFILLED})

# The market-wide benchmark proxy: SPY. Used when the ticker's sector is unknown (the COMMON case
# for backfilled Form 4 — the sector map covers ~60 tickers) and as the ultimate fallback when a
# sector ETF lacks aligned bars. A label against SPY is stamped benchmark="market".
MARKET_BENCHMARK = "SPY"

# Sector -> a liquid sector-ETF price proxy, so "sector return" has an actual price series to
# measure against (the sector map yields a Sector, not a tradeable symbol). Documented in
# MEMORY.md as the ETF-proxy decision. A label against one of these is stamped
# benchmark="sector:<ETF>" so the Task-05 diagnostic can split sector-relative vs market-relative
# results on the "sector:"/"market" prefix.
SECTOR_ETF: dict[Sector, str] = {
    Sector.TECHNOLOGY: "XLK",
    Sector.DEFENSE: "ITA",
    Sector.HEALTHCARE: "XLV",
    Sector.ENERGY: "XLE",
    Sector.FINANCIALS: "XLF",
    Sector.INDUSTRIALS: "XLI",
    Sector.CONSUMER: "XLY",
    Sector.COMMUNICATIONS: "XLC",
    Sector.MATERIALS: "XLB",
    Sector.UTILITIES: "XLU",
    Sector.REAL_ESTATE: "XLRE",
}

# A generous calendar buffer past the last disclosure so the daily-bar window covers ~2x the max
# horizon in TRADING days despite weekends/holidays (20 trading days ~ 28 calendar; 2x + slack).
_FORWARD_BUFFER_DAYS = 2 * max(HORIZONS) * 2 + 7
# A small buffer before the earliest disclosure so its entry bar (the NEXT session) is in-window.
_BACKWARD_BUFFER_DAYS = 7


class LabelingError(Exception):
    """A labeler run could not proceed (e.g. no historical price source, empty benchmark set)."""


@dataclass(frozen=True, slots=True)
class LabelReport:
    """Per-run outcome: what was labeled and (unless dry-run) written."""

    label_run_id: str
    as_of: datetime
    signal_count: int  # ticker-resolved, labelable signals fed to the computation
    labels: int  # total label rows produced (one per signal x horizon)
    ok: int
    unresolvable: int
    committed: bool


def _entry_index(points: Sequence[tuple[datetime, Decimal]], disclosed_at: datetime) -> int | None:
    """Index of the entry bar: the first bar whose UTC DATE is strictly after the disclosure date.

    Anti-lookahead core (RESEARCH-PLAN §2.1): daily bars, entry is the NEXT session after the
    disclosure's own calendar date, so an intra-day or after-close disclosure never resolves to
    its own session. Returns None if no such forward bar exists (the signal is too recent).
    """
    disclosure_date = disclosed_at.astimezone(UTC).date()
    for index, (when, _close) in enumerate(points):
        if when.date() > disclosure_date:
            return index
    return None


def _benchmark_entry_index(
    points: Sequence[tuple[datetime, Decimal]], entry_date: datetime
) -> int | None:
    """Index of the benchmark's entry bar: its first bar with DATE >= the ticker's entry-bar date.

    Aligns the benchmark window to the SAME entry date as the ticker (clarification 2). Returns
    None when the benchmark has no bar on/after the ticker's entry date.
    """
    target = entry_date.date()
    for index, (when, _close) in enumerate(points):
        if when.date() >= target:
            return index
    return None


def _forward_return(
    points: Sequence[tuple[datetime, Decimal]], entry_index: int, horizon: int
) -> float | None:
    """``close(entry+h) / close(entry) - 1`` by bar-index offset, as a float, or None if short.

    Decimal -> float conversion happens here, once, at the return-computation boundary.
    """
    exit_index = entry_index + horizon
    if exit_index >= len(points):
        return None
    entry_close = points[entry_index][1]
    exit_close = points[exit_index][1]
    if entry_close == 0:
        return None
    return float(exit_close) / float(entry_close) - 1.0


class LabelOutcomes:
    """Label ticker-resolved signals with forward abnormal returns (event-study projection)."""

    def __init__(
        self,
        *,
        read_model: ReadModelPort,
        prices: HistoricalPricesPort,
        sector_map: SectorMapPort,
        labels_store: OutcomeLabelsPort,
        event_store: EventStorePort,
        clock: ClockPort,
    ) -> None:
        self._read_model = read_model
        self._prices = prices
        self._sector_map = sector_map
        self._labels_store = labels_store
        self._event_store = event_store
        self._clock = clock

    async def execute(
        self,
        *,
        label_run_id: str,
        dry_run: bool,
        signals: Sequence[Signal] | None = None,
    ) -> LabelReport:
        """Label the labelable signals, writing (unless dry_run) a fresh ``label_run_id`` run.

        Pulls REAL+BACKFILLED ticker-resolved signals from the read model when ``signals`` is not
        supplied; when it IS supplied (tests / explicit callers), that batch is used as-is but
        SYNTHETIC input is refused (:func:`ensure_no_synthetic`) and only ticker-resolved,
        labelable-provenance signals are labeled. On a --confirm run the labels replace this run
        id's rows (delete-then-write) and ONE ``user.action`` audit event is appended.
        """
        as_of = self._clock.now()

        if signals is None:
            candidates = list(await self._read_model.all_signals())
        else:
            # An explicit caller must not smuggle synthetic data into the labeler/learner.
            ensure_no_synthetic(signals)
            candidates = list(signals)

        # Filter to labelable provenance (REAL + BACKFILLED) and ticker-resolved signals only.
        # Signals without a ticker are SKIPPED (not errors) — only "ticker-resolved" signals are
        # labeled (their unresolved-ticker set is itself research data, handled upstream).
        labelable = [
            s
            for s in candidates
            if s.provenance in LABELABLE_PROVENANCE and s.ticker is not None and s.ticker.strip()
        ]
        # Defence in depth: even a read-model pull must never carry SYNTHETIC into the labeler.
        ensure_no_synthetic(labelable)

        price_series = await self._fetch_bars(labelable)

        labels: list[OutcomeLabel] = []
        for signal in labelable:
            labels.extend(self._label_signal(signal, price_series, label_run_id))

        ok = sum(1 for label in labels if label.status == "ok")
        unresolvable = len(labels) - ok

        committed = False
        if not dry_run:
            await self._labels_store.replace_run(label_run_id, labels)
            await self._append_audit(
                label_run_id=label_run_id, as_of=as_of, signal_count=len(labelable)
            )
            committed = True

        return LabelReport(
            label_run_id=label_run_id,
            as_of=as_of,
            signal_count=len(labelable),
            labels=len(labels),
            ok=ok,
            unresolvable=unresolvable,
            committed=committed,
        )

    async def _fetch_bars(self, signals: Sequence[Signal]) -> PriceSeries:
        """One price_series call for {tickers} u {sector ETFs} u {SPY} over the padded window."""
        if not signals:
            return PriceSeries(series={})
        # Callers only ever pass `labelable` signals (already ticker-resolved, see `execute`),
        # but `s.ticker` is typed `str | None` on the shared domain Signal, so narrow here too.
        tickers = {s.ticker.strip().upper() for s in signals if s.ticker and s.ticker.strip()}
        # Every ETF proxy any ticker could map to, plus the ETFs for their resolved sectors, plus
        # SPY (the market fallback and ultimate benchmark). Collecting all proxies keeps this to
        # one fetch.
        symbols: set[str] = set(tickers)
        for ticker in tickers:
            sector = self._sector_map.sector_for(ticker)
            if sector is not None and sector in SECTOR_ETF:
                symbols.add(SECTOR_ETF[sector])
        symbols.add(MARKET_BENCHMARK)

        disclosures = [s.created_at for s in signals]
        start = min(disclosures) - timedelta(days=_BACKWARD_BUFFER_DAYS)
        end = max(disclosures) + timedelta(days=_FORWARD_BUFFER_DAYS)
        return await self._prices.price_series(sorted(symbols), start, end)

    def _label_signal(
        self, signal: Signal, prices: PriceSeries, label_run_id: str
    ) -> list[OutcomeLabel]:
        """One label row per horizon for a single signal (ok or unresolvable-with-reason)."""
        # Only ever called on `labelable` signals, which `execute` has already filtered to
        # ticker-resolved (`s.ticker is not None`) — narrow the shared domain type here too.
        assert signal.ticker is not None
        ticker = signal.ticker.strip().upper()
        benchmark_symbol, benchmark_tag = self._benchmark_for(ticker, prices)

        ticker_points = prices.series.get(ticker) or ()
        if not ticker_points:
            return [
                self._unresolvable(signal, ticker, horizon, benchmark_tag, "no_bars", label_run_id)
                for horizon in HORIZONS
            ]

        entry_index = _entry_index(ticker_points, signal.created_at)
        if entry_index is None:
            return [
                self._unresolvable(
                    signal,
                    ticker,
                    horizon,
                    benchmark_tag,
                    "insufficient_forward_bars",
                    label_run_id,
                )
                for horizon in HORIZONS
            ]

        entry_date = ticker_points[entry_index][0]
        benchmark_points = prices.series.get(benchmark_symbol) or ()
        benchmark_entry = _benchmark_entry_index(benchmark_points, entry_date)

        rows: list[OutcomeLabel] = []
        for horizon in HORIZONS:
            ticker_return = _forward_return(ticker_points, entry_index, horizon)
            if ticker_return is None:
                rows.append(
                    self._unresolvable(
                        signal,
                        ticker,
                        horizon,
                        benchmark_tag,
                        "insufficient_forward_bars",
                        label_run_id,
                    )
                )
                continue
            if benchmark_entry is None:
                rows.append(
                    self._unresolvable(
                        signal,
                        ticker,
                        horizon,
                        benchmark_tag,
                        "benchmark_bars_missing",
                        label_run_id,
                    )
                )
                continue
            benchmark_return = _forward_return(benchmark_points, benchmark_entry, horizon)
            if benchmark_return is None:
                rows.append(
                    self._unresolvable(
                        signal,
                        ticker,
                        horizon,
                        benchmark_tag,
                        "benchmark_bars_missing",
                        label_run_id,
                    )
                )
                continue
            rows.append(
                OutcomeLabel(
                    signal_id=signal.id,
                    ticker=ticker,
                    horizon_days=horizon,
                    abnormal_return=ticker_return - benchmark_return,
                    benchmark=benchmark_tag,
                    status="ok",
                    reason=None,
                    label_run_id=label_run_id,
                )
            )
        return rows

    def _benchmark_for(self, ticker: str, prices: PriceSeries) -> tuple[str, str]:
        """Resolve (benchmark_symbol, benchmark_tag) with the SPY fallback.

        Sector proxy when the sector map resolves the ticker AND that ETF has bars; otherwise SPY
        ("market"). Falling back when the sector ETF lacks bars keeps the label computable rather
        than throwing away a signal for a thin sector proxy.
        """
        sector = self._sector_map.sector_for(ticker)
        if sector is not None and sector in SECTOR_ETF:
            etf = SECTOR_ETF[sector]
            if prices.series.get(etf):
                return etf, f"sector:{etf}"
        return MARKET_BENCHMARK, "market"

    @staticmethod
    def _unresolvable(
        signal: Signal, ticker: str, horizon: int, benchmark_tag: str, reason: str, run_id: str
    ) -> OutcomeLabel:
        return OutcomeLabel(
            signal_id=signal.id,
            ticker=ticker,
            horizon_days=horizon,
            abnormal_return=None,
            benchmark=benchmark_tag,
            status="unresolvable",
            reason=reason,
            label_run_id=run_id,
        )

    async def _append_audit(self, *, label_run_id: str, as_of: datetime, signal_count: int) -> None:
        """Append ONE user.action audit event recording the run parameters (not a PIT signal).

        occurred_at = clock.now(): this is an audit fact about the run, so it is stamped with the
        run's wall-clock (unlike a backfilled signal.received event, whose occurred_at is its
        disclosure timestamp).
        """
        await self._event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": "label_outcomes",
                    "label_run_id": label_run_id,
                    "as_of": as_of.isoformat(),
                    "horizons": list(HORIZONS),
                    "provenance_filter": sorted(p.value for p in LABELABLE_PROVENANCE),
                    "benchmark_policy": "sector_etf_proxy_else_market_spy",
                    "signal_count": signal_count,
                },
                occurred_at=self._clock.now(),
                correlation_id=label_run_id,
            )
        )
