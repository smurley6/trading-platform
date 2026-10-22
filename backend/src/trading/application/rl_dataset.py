"""Frozen, point-in-time-correct daily-bar dataset for the offline RL research program (task-rl-02).

``BuildRlDataset`` pulls ONE symbol's daily closes through the ``HistoricalPricesPort`` and turns
them into an ``RlDataset``: an ascending tuple of CLOSED bars, each carrying an EXPLICIT
``known_at`` — the moment that bar's close became public. ``rl_dataset_to_json`` renders it to
deterministic bytes; ``python -m trading.jobs.rl_dataset`` freezes those bytes to disk, and every
later RL milestone reads that file, never the network (docs/RL-SLEEVE-PROGRAM.md §4.2, §4.7) —
always through ``rl_dataset_from_json`` (task-rl-03), the validating inverse of the writer.

**The knowledge-time rule — the load-bearing assumption.** Alpaca stamps a daily bar ``t`` at the
session's OPENING boundary — 00:00 America/New_York, i.e. ``04:00Z`` under EDT and ``05:00Z``
under EST — while the value the adapter stores (``alpaca_history.py``: ``entry["c"]``) is that
session's CLOSE. The close is NOT public at ``session_ts``: treating the bar timestamp as knowledge
time reads ~16 hours into the future. So each bar's knowledge time is::

    known_at = (16:00 America/New_York on session_ts's New-York calendar date) + 15 min

computed in the exchange time zone (``zoneinfo``), then expressed in UTC. On every trading day this
equals the fixed ``session_ts + 16h15m`` of the program doc (US DST switches happen at 02:00 local
on Sundays, never inside a trading session), so it is correct under BOTH DST regimes:
``2026-01-02T05:00:00Z -> 2026-01-02T21:15:00Z`` (EST), ``2026-07-02T04:00:00Z ->
2026-07-02T20:15:00Z`` (EDT). Early-close sessions (13:00 ET) make ``known_at`` ~3h LATE —
conservative, therefore safe; never shorten the offset.

**Fail closed on a stamping change.** The opening-boundary stamping is verified by inference, not
by Alpaca's docs (which only call ``t`` an RFC-3339 timestamp). Rather than blindly adding an
offset, every bar's ``session_ts`` MUST be exactly 00:00 America/New_York; anything else (an
end-of-bar stamp, a UTC-midnight stamp, a stale DST offset) raises ``RlDatasetError`` instead of
silently producing a 16-hour PIT error that every internal-consistency test would still pass.

**Closed bars only.** A bar whose ``known_at`` is after the clock's ``now`` (the developing
session) is dropped — its "close" is not final.

Reads ``series[symbol]`` POSITIONALLY. Never calls ``PriceSeries.price_at``: it returns the
EARLIEST point when asked for a time before the series starts — a silent left-edge forward-look
(``domain/prices.py``).

Adjustment: the mode (``raw`` | ``all``) is fixed when the historical-prices adapter is wired and
is passed here explicitly so it is RECORDED in the dataset, never implicit. Note that ``all``
back-adjusts historical levels using LATER corporate actions (the level series is current-vintage);
returns/ratios are what an investor actually earned, but absolute price LEVELS before a dividend are
not the as-traded prints.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from itertools import pairwise
from typing import Any, Final
from zoneinfo import ZoneInfo

from trading.application.ports.clock import ClockPort
from trading.application.ports.historical import HistoricalPricesPort

# The exchange time zone the session boundary and close are defined in.
MARKET_TZ: Final = ZoneInfo("America/New_York")
# Regular-session close, exchange-local. Alpaca's daily-bar `t` is this session's 00:00 local.
SESSION_CLOSE_LOCAL: Final = time(16, 0)
# Settle buffer after the close before the bar is treated as public. Matches task-rl-03's
# `_SETTLE_BUFFER` (16:00 + 15 min => the pinned 21:15Z / 20:15Z cases).
SETTLE_BUFFER: Final = timedelta(minutes=15)

ALLOWED_ADJUSTMENTS: Final = frozenset({"raw", "all"})

# Bump if the on-disk layout changes; recorded in the file header.
# v2 (2026-10-21): header gains `price_basis` (what the close LEVELS do and do not mean).
DATASET_SCHEMA_VERSION: Final = 2

KNOWN_AT_RULE: Final = (
    "session_ts must be 00:00 America/New_York (Alpaca daily-bar opening boundary); "
    "known_at = 16:00 America/New_York on that date + 15 min, in UTC"
)

# Self-describing price basis per adjustment mode, written to the header next to `known_at_rule`
# so no consumer has to know the adjustment semantics out-of-band (docs/RL-SLEEVE-PROGRAM.md §4.2
# "Bar adjustment": closes may be used ONLY as ratios of closes).
PRICE_BASIS: Final[Mapping[str, str]] = {
    "all": (
        "Alpaca adjustment=all: split- and dividend-adjusted (total-return) closes as served by "
        "Alpaca at generated_at; may reflect later corrections. Ratios of closes are PIT-correct; "
        "absolute levels before later dividends are back-adjusted (current-vintage), NOT as-traded "
        "prices. Use closes only as ratios of closes."
    ),
    "raw": (
        "Alpaca adjustment=raw: unadjusted closes as served by Alpaca at generated_at; may reflect "
        "later corrections. Ex-dividend drops are NOT adjusted, so ratios understate total return. "
        "Use closes only as ratios of closes."
    ),
}


class RlDatasetError(ValueError):
    """The bars cannot be frozen into a PIT-correct dataset (fails closed, never guesses)."""


def _require_aware(value: datetime, *, what: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RlDatasetError(f"{what} must be timezone-aware, got naive {value.isoformat()}")


def session_known_at(session_ts: datetime) -> datetime:
    """The UTC moment a daily bar stamped ``session_ts`` (its session's 00:00 New York) closed
    publicly: 16:00 New York on that date + ``SETTLE_BUFFER``.

    Raises ``RlDatasetError`` if ``session_ts`` is naive or is not exactly 00:00 New York — the
    stamping convention this rule depends on — so a stamping change fails loudly.
    """
    _require_aware(session_ts, what="session_ts")
    local = session_ts.astimezone(MARKET_TZ)
    if local.time() != time(0, 0):
        raise RlDatasetError(
            f"bar session_ts {session_ts.astimezone(UTC).isoformat()} is "
            f"{local.isoformat()} in America/New_York, not the 00:00 session-open boundary "
            "Alpaca daily bars are stamped at; refusing to infer a knowledge time"
        )
    close_local = datetime.combine(local.date(), SESSION_CLOSE_LOCAL, tzinfo=MARKET_TZ)
    return (close_local + SETTLE_BUFFER).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class RlBar:
    """One closed daily bar: Alpaca's opening-boundary stamp, the session close, and the moment
    that close became public. Both timestamps tz-aware UTC; ``known_at`` strictly after
    ``session_ts``."""

    session_ts: datetime
    known_at: datetime
    close: float

    def __post_init__(self) -> None:
        _require_aware(self.session_ts, what="session_ts")
        _require_aware(self.known_at, what="known_at")
        if self.known_at <= self.session_ts:
            raise RlDatasetError(
                f"known_at {self.known_at.isoformat()} must be after session_ts "
                f"{self.session_ts.isoformat()}"
            )
        if not math.isfinite(self.close) or self.close <= 0:
            raise RlDatasetError(
                f"close must be finite and positive, got {self.close} "
                f"at {self.session_ts.isoformat()}"
            )


@dataclass(frozen=True, slots=True)
class RlDataset:
    """A frozen single-symbol daily-close dataset. ``bars`` strictly ascending by ``session_ts``,
    closed bars only. ``adjustment`` is recorded, never implicit. ``start``/``end`` are the
    REQUESTED fetch window; ``generated_at`` is the clock's ``now`` at build time."""

    symbol: str
    bars: tuple[RlBar, ...]
    adjustment: str
    source_adapter: str
    start: datetime
    end: datetime
    generated_at: datetime

    def __post_init__(self) -> None:
        if self.adjustment not in ALLOWED_ADJUSTMENTS:
            raise RlDatasetError(
                f"adjustment must be one of {sorted(ALLOWED_ADJUSTMENTS)}, got {self.adjustment!r}"
            )
        for value, what in (
            (self.start, "start"),
            (self.end, "end"),
            (self.generated_at, "generated_at"),
        ):
            _require_aware(value, what=what)
        for previous, current in pairwise(self.bars):
            if current.session_ts <= previous.session_ts:
                raise RlDatasetError(
                    "bars must be strictly ascending by session_ts: "
                    f"{current.session_ts.isoformat()} follows {previous.session_ts.isoformat()}"
                )


class BuildRlDataset:
    """Fetch one symbol's daily bars over a lookback window and freeze them PIT-correctly."""

    def __init__(self, *, prices: HistoricalPricesPort, clock: ClockPort, adjustment: str) -> None:
        if adjustment not in ALLOWED_ADJUSTMENTS:
            raise ValueError(
                f"adjustment must be one of {sorted(ALLOWED_ADJUSTMENTS)}, got {adjustment!r}"
            )
        self._prices = prices
        self._clock = clock
        self._adjustment = adjustment

    @property
    def adjustment(self) -> str:
        """The adjustment mode the wired historical-prices adapter was built with."""
        return self._adjustment

    async def execute(self, *, symbol: str, lookback_days: int) -> RlDataset:
        wanted = symbol.strip().upper()
        if not wanted:
            raise RlDatasetError("symbol must be non-empty")
        if lookback_days <= 0:
            raise RlDatasetError(f"lookback_days must be positive, got {lookback_days}")

        now = self._clock.now()
        _require_aware(now, what="clock.now()")
        now = now.astimezone(UTC)
        # Window start = 00:00 New York on the lookback date (exactly a bar's opening stamp), so
        # that date's bar is inside the inclusive [start, end] request.
        start_date = (now.astimezone(MARKET_TZ) - timedelta(days=lookback_days)).date()
        start = datetime.combine(start_date, time(0, 0), tzinfo=MARKET_TZ).astimezone(UTC)

        series = await self._prices.price_series([wanted], start, now)
        points = series.series.get(wanted)
        if not points:
            raise RlDatasetError(f"no bars returned for {wanted} over [{start}, {now}]")

        bars: list[RlBar] = []
        for session_ts, close in points:  # positional read; never PriceSeries.price_at
            known_at = session_known_at(session_ts)
            if known_at > now:
                continue  # the developing session: its close is not final yet
            bars.append(
                RlBar(
                    session_ts=session_ts.astimezone(UTC),
                    known_at=known_at,
                    close=float(close),
                )
            )
        if not bars:
            raise RlDatasetError(f"no CLOSED bars for {wanted} as of {now.isoformat()}")

        return RlDataset(
            symbol=wanted,
            bars=tuple(bars),
            adjustment=self._adjustment,
            source_adapter=type(self._prices).__name__,
            start=start,
            end=now,
            generated_at=now,
        )


def _iso_utc(value: datetime) -> str:
    _require_aware(value, what="timestamp")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def rl_dataset_header(dataset: RlDataset) -> dict[str, Any]:
    """The file header. Every field is explicit — notably ``adjustment`` and ``source_adapter``."""
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "symbol": dataset.symbol,
        "adjustment": dataset.adjustment,
        "source_adapter": dataset.source_adapter,
        "start": _iso_utc(dataset.start),
        "end": _iso_utc(dataset.end),
        "n_bars": len(dataset.bars),
        "generated_at": _iso_utc(dataset.generated_at),
        "known_at_rule": KNOWN_AT_RULE,
        "price_basis": PRICE_BASIS[dataset.adjustment],
    }


def _bar_record(bar: RlBar) -> Mapping[str, Any]:
    return {
        "session_ts": _iso_utc(bar.session_ts),
        "known_at": _iso_utc(bar.known_at),
        "close": bar.close,
    }


def rl_dataset_to_json(dataset: RlDataset) -> bytes:
    """Deterministic UTF-8 JSON: a pretty header, then one bar object per line (diff-friendly).

    Same dataset => byte-identical output (fixed key order, ``repr``-exact floats, no NaN/inf,
    trailing newline). Every timestamp is ``...Z`` UTC.
    """
    header = json.dumps(rl_dataset_header(dataset), indent=2, allow_nan=False)
    rows = [
        "    " + json.dumps(_bar_record(bar), separators=(", ", ": "), allow_nan=False)
        for bar in dataset.bars
    ]
    bars_block = ['  "bars": [', ",\n".join(rows), "  ]"] if rows else ['  "bars": []']
    lines = ["{", '  "header": ' + header.replace("\n", "\n  ") + ",", *bars_block, "}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_utc(value: object, *, what: str) -> datetime:
    if not isinstance(value, str):
        raise RlDatasetError(f"{what} must be an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RlDatasetError(f"{what} is not ISO-8601: {value!r}") from exc
    _require_aware(parsed, what=what)
    return parsed.astimezone(UTC)


def _reject_constant(name: str) -> float:
    raise RlDatasetError(f"non-finite JSON constant {name} is not allowed in an RL dataset")


def rl_dataset_from_json(data: bytes) -> RlDataset:
    """Parse bytes written by ``rl_dataset_to_json`` back into an ``RlDataset`` — the ONE reader
    every RL consumer goes through (never ad-hoc ``json.loads`` of the artifact).

    Fails closed (``RlDatasetError``) on anything it cannot vouch for: a different
    ``schema_version``; a header ``known_at_rule`` or ``price_basis`` that is not this module's
    current text; an ``n_bars`` that disagrees with the rows; and — the drift guard — any bar whose
    STORED ``known_at`` is not exactly ``session_known_at(session_ts)``. The stored value is what
    consumers use; recomputing it here only VALIDATES it, so the knowledge-time rule exists once
    (``session_known_at``) and a rule change can never silently disagree with a frozen file.

    Round trip: ``rl_dataset_to_json(rl_dataset_from_json(b)) == b`` for every file this module
    wrote.
    """
    try:
        doc = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RlDatasetError(f"RL dataset is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict) or set(doc) != {"header", "bars"}:
        raise RlDatasetError("RL dataset must be an object with exactly 'header' and 'bars'")
    header, rows = doc["header"], doc["bars"]
    if not isinstance(header, dict) or not isinstance(rows, list):
        raise RlDatasetError("'header' must be an object and 'bars' a list")

    if header.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RlDatasetError(
            f"schema_version {header.get('schema_version')!r} is not the supported "
            f"{DATASET_SCHEMA_VERSION}; regenerate the dataset"
        )
    if header.get("known_at_rule") != KNOWN_AT_RULE:
        raise RlDatasetError(
            "header known_at_rule does not match this code's rule; the file was frozen under a "
            f"different knowledge-time rule: {header.get('known_at_rule')!r}"
        )
    adjustment = header.get("adjustment")
    if not isinstance(adjustment, str) or adjustment not in ALLOWED_ADJUSTMENTS:
        raise RlDatasetError(
            f"header adjustment {adjustment!r} is not one of {ALLOWED_ADJUSTMENTS}"
        )
    if header.get("price_basis") != PRICE_BASIS[adjustment]:
        raise RlDatasetError(f"header price_basis does not match adjustment {adjustment!r}")
    symbol, source_adapter = header.get("symbol"), header.get("source_adapter")
    if not isinstance(symbol, str) or not symbol or not isinstance(source_adapter, str):
        raise RlDatasetError("header symbol and source_adapter must be non-empty strings")
    if header.get("n_bars") != len(rows):
        raise RlDatasetError(f"header n_bars {header.get('n_bars')!r} != {len(rows)} bar rows")

    bars: list[RlBar] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"session_ts", "known_at", "close"}:
            raise RlDatasetError(f"bar {index} must have exactly session_ts, known_at, close")
        session_ts = _parse_utc(row["session_ts"], what=f"bar {index} session_ts")
        known_at = _parse_utc(row["known_at"], what=f"bar {index} known_at")
        expected = session_known_at(session_ts)
        if known_at != expected:
            raise RlDatasetError(
                f"bar {index} ({session_ts.isoformat()}): stored known_at {known_at.isoformat()} "
                f"!= rule {expected.isoformat()} — knowledge-time drift; regenerate the dataset"
            )
        close = row["close"]
        if isinstance(close, bool) or not isinstance(close, int | float):
            raise RlDatasetError(f"bar {index} close must be a number, got {close!r}")
        try:
            close_value = float(close)
        except OverflowError as exc:  # a JSON integer too large for a float
            raise RlDatasetError(f"bar {index} close {close} overflows a float") from exc
        bars.append(RlBar(session_ts=session_ts, known_at=known_at, close=close_value))

    return RlDataset(
        symbol=symbol,
        bars=tuple(bars),
        adjustment=adjustment,
        source_adapter=source_adapter,
        start=_parse_utc(header.get("start"), what="header start"),
        end=_parse_utc(header.get("end"), what="header end"),
        generated_at=_parse_utc(header.get("generated_at"), what="header generated_at"),
    )
