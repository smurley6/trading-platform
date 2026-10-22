"""Point-in-time daily bar table for the RL env (task-rl-03; docs/RL-SLEEVE-PROGRAM.md §4.2).

PURE half of ``trading.rl``: stdlib + first-party pure modules only — no numpy, no gymnasium, no
torch — so the leakage-critical arithmetic runs in the default ``pytest`` with zero optional deps.

**Why ``session_ts`` is not knowledge time.** Alpaca stamps a daily bar ``t`` at the session's
OPENING boundary — 00:00 America/New_York, i.e. ``04:00Z`` under EDT and ``05:00Z`` under EST —
while ``alpaca_history.py:167`` (``entry.get("c")``) stores that session's CLOSE. The close is not
public at ``session_ts``; reading it there looks ~16 hours into the future. Every ``Bar`` therefore
carries an explicit ``known_at``: the moment its close became public.

**The rule** is 16:00 New York + a 15 min settle buffer (mirrors
``alpaca_history._RECENT_DATA_BUFFER``, 16 min). Because ``session_ts`` is local-midnight already
expressed in UTC, that is a FIXED ``session_ts + 16h15m`` offset, correct under BOTH DST regimes
(``2026-01-02T05:00:00Z -> 21:15:00Z`` EST, ``2026-07-02T04:00:00Z -> 20:15:00Z`` EDT): US DST
switches happen at 02:00 local on Sundays, never inside a trading session. Early-close sessions
(13:00 ET: day after Thanksgiving, Christmas Eve) make ``known_at`` ~3 hours LATE — conservative,
therefore safe. The offset must never be shortened.

**One copy of the rule.** This module does NOT re-implement the offset. The rule lives ONCE, in
``trading.application.rl_dataset.session_known_at`` (zone-aware, and fail-closed on any stamp that
is not exactly 00:00 New York). ``BarTable.from_dataset`` CONSUMES the ``known_at`` stored in the
frozen artifact and VALIDATES it against that function, raising ``BarTableError`` on any mismatch;
``BarTable.from_price_series`` (an un-frozen in-memory series) computes ``known_at`` with that same
function. So the frozen file and the code can never silently disagree about knowledge time.

**Closes are ratios only** (§4.2 "Bar adjustment — DECIDED", BINDING): the canonical artifact is
``adjustment=all``, whose LEVELS are back-adjusted with later dividends. ``closes()`` exists to be
consumed as ``close[t] / close[t-1]``-style ratios; no level may drive a feature, reward or cost.

Never calls ``PriceSeries.price_at``: ``domain/prices.py`` returns the EARLIEST point when asked for
a time before the series starts — a silent left-edge forward-look. Series are read positionally.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

from trading.application.rl_dataset import RlDataset, RlDatasetError, session_known_at
from trading.domain.prices import PriceSeries


class BarTableError(ValueError):
    """The bars cannot form a point-in-time-correct table (fails closed, never guesses)."""


def _require_aware(value: datetime, *, what: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise BarTableError(f"{what} must be timezone-aware, got naive {value.isoformat()}")


@dataclass(frozen=True, slots=True)
class Bar:
    """One closed daily bar. ``session_ts`` is Alpaca's ``t`` (the session OPEN boundary) and is
    never knowledge time; ``known_at`` is when ``close`` became public. Both tz-aware."""

    session_ts: datetime
    close: Decimal
    known_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.session_ts, what="session_ts")
        _require_aware(self.known_at, what="known_at")
        if self.known_at <= self.session_ts:
            raise BarTableError(
                f"known_at {self.known_at.isoformat()} must be after session_ts "
                f"{self.session_ts.isoformat()}"
            )
        if not self.close.is_finite() or self.close <= 0:
            raise BarTableError(
                f"close must be finite and positive (closes are used as ratios), got "
                f"{self.close} at {self.session_ts.isoformat()}"
            )


@dataclass(frozen=True, slots=True)
class BarTable:
    """A single-symbol table of closed bars, strictly ascending by ``session_ts`` AND by
    ``known_at`` (a later session can never become public before an earlier one)."""

    symbol: str
    bars: tuple[Bar, ...]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise BarTableError("symbol must be non-empty")
        for previous, current in pairwise(self.bars):
            if current.session_ts <= previous.session_ts:
                raise BarTableError(
                    "bars must be strictly ascending by session_ts: "
                    f"{current.session_ts.isoformat()} follows {previous.session_ts.isoformat()}"
                )
            if current.known_at <= previous.known_at:
                raise BarTableError(
                    "bars must be strictly ascending by known_at: "
                    f"{current.known_at.isoformat()} follows {previous.known_at.isoformat()}"
                )

    @classmethod
    def from_dataset(cls, dataset: RlDataset) -> "BarTable":
        """The canonical constructor: the frozen artifact, read through ``rl_dataset_from_json``.

        Consumes each bar's STORED ``known_at`` and asserts it equals
        ``session_known_at(session_ts)`` — a mismatch (the rule changed, or the file was edited)
        raises ``BarTableError`` naming the bar. ``close`` is converted float -> ``Decimal`` via
        ``repr`` so ``closes()`` returns the stored floats bit-for-bit.
        """
        bars: list[Bar] = []
        for index, bar in enumerate(dataset.bars):
            try:
                expected = session_known_at(bar.session_ts)
            except RlDatasetError as exc:
                raise BarTableError(f"bar {index}: {exc}") from exc
            if bar.known_at != expected:
                raise BarTableError(
                    f"bar {index} ({bar.session_ts.isoformat()}): stored known_at "
                    f"{bar.known_at.isoformat()} != rule {expected.isoformat()} — "
                    "knowledge-time drift between the frozen dataset and the code"
                )
            bars.append(
                Bar(
                    session_ts=bar.session_ts.astimezone(UTC),
                    close=Decimal(repr(bar.close)),
                    known_at=bar.known_at.astimezone(UTC),
                )
            )
        return cls(symbol=dataset.symbol, bars=tuple(bars))

    @classmethod
    def from_price_series(cls, series: PriceSeries, symbol: str) -> "BarTable":
        """Build from an in-memory ``PriceSeries`` (e.g. a test fixture or a fresh port read).

        Reads ``series.series[symbol]`` POSITIONALLY and stamps each point with
        ``session_known_at`` — the same single rule the frozen dataset was built with. Raises
        ``BarTableError`` if the symbol is missing/empty, a stamp is not 00:00 New York, or the
        points are not strictly ascending. The caller supplies CLOSED sessions only (use
        ``slice_known_by(now)`` to drop a developing session).
        """
        points = series.series.get(symbol)
        if not points:
            raise BarTableError(f"no bars for symbol {symbol!r} in the price series")
        bars: list[Bar] = []
        for session_ts, close in points:
            _require_aware(session_ts, what="session_ts")
            try:
                known_at = session_known_at(session_ts)
            except RlDatasetError as exc:
                raise BarTableError(str(exc)) from exc
            bars.append(Bar(session_ts=session_ts.astimezone(UTC), close=close, known_at=known_at))
        return cls(symbol=symbol, bars=tuple(bars))

    def __len__(self) -> int:
        return len(self.bars)

    def closes(self) -> tuple[float, ...]:
        """Closes as floats, index-aligned with ``bars``. Consume ONLY as ratios of closes."""
        return tuple(float(bar.close) for bar in self.bars)

    def slice_known_by(self, when: datetime) -> "BarTable":
        """The bars whose close was public at ``when`` (``known_at <= when``) — by knowledge time,
        never by ``session_ts``."""
        _require_aware(when, what="when")
        return BarTable(
            symbol=self.symbol, bars=tuple(bar for bar in self.bars if bar.known_at <= when)
        )

    def split(self, at: datetime, *, embargo: timedelta) -> tuple["BarTable", "BarTable"]:
        """Split into ``(train, eval)`` with no window straddling the boundary.

        ``eval`` = bars with ``session_ts >= at``. ``train`` = bars whose close was public
        strictly before ``at - embargo`` (``known_at < at - embargo``). A bar whose session opened
        before ``at`` but whose close was not yet public by then belongs to NEITHER side, so
        ``max(train.known_at) < min(eval.session_ts)`` and the two session sets are disjoint.
        """
        _require_aware(at, what="at")
        if embargo < timedelta(0):
            raise BarTableError(f"embargo must be non-negative, got {embargo}")
        cutoff = at - embargo
        train = tuple(bar for bar in self.bars if bar.known_at < cutoff)
        evaluation = tuple(bar for bar in self.bars if bar.session_ts >= at)
        return BarTable(symbol=self.symbol, bars=train), BarTable(
            symbol=self.symbol, bars=evaluation
        )
