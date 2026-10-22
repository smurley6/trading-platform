"""Episode results and the RL evaluation metrics (task-rl-07; program doc §5.2, §5.4).

PURE half of ``trading.rl``: stdlib + first-party pure modules only — no numpy, no gymnasium — so
everything a report, the demo bundle job (rl-12) or a gate needs to turn a replay into numbers runs
with zero optional deps. ``rl/scoring.py`` (heavy: it drives the env) PRODUCES the
:class:`EpisodeResult` records consumed here.

**Reuse, not rebuild (§5.4).** Sharpe, Sortino, total return, max drawdown and CAGR are the
existing ``trading.domain.metrics`` functions, unchanged: ``sharpe_ratio`` (rf = 0, POPULATION
stdev, daily simple returns annualized by ``sqrt(252)``), ``sortino_ratio``, ``total_return_pct``,
``max_drawdown_pct`` and ``cagr_pct`` (365.25-day calendar span between the first and last marks).
What is new is only what the env makes measurable: turnover, exposure, action entropy, the
break-even cost and the random-policy null band.

**Everything comes from the env's own accounting** (its info dicts): equity marks, positions, bar
INDICES and fill events. No price level appears anywhere in an :class:`EpisodeResult` (§4.2
returns-only rule). The one exception is :func:`buy_and_hold_reference`, a frictionless benchmark
computed from close RATIOS over the segment, which never feeds a decision.

**Metric definitions** (one :class:`EpisodeResult` = one replay of one policy, one seed, one
:class:`Segment`; ``n = steps`` decisions; equity marks ``E_0 = 1.0 .. E_n``):

- ``daily returns`` ``r_i = E_i / E_{i-1} - 1`` (one per step, net of costs — costs live inside the
  equity); ``sharpe`` / ``sortino`` annualize them by ``sqrt(252)``.
- ``turnover`` = ``sum |delta position|`` over fills (env ``info["turnover"]``), in units of equity;
  ``annualized_turnover = turnover * 252 / n``.
- ``exposure_fraction`` = mean position HELD over each step's market move (``info["position"]``
  before the step, the position that earns ``C_t -> C_{t+1}``). Buy-and-hold scores
  ``(n - 1 - lag) / n``, NOT 1.0: the one-session fill lag is honest exposure lost.
- ``action_entropy`` = Shannon entropy, in NATS, of the empirical distribution of EFFECTIVE actions
  (no-ops count as HOLD — what was actually done). 0.0 for a constant policy; ``ln 3`` max.
- ``break_even_cost_bps`` — see :func:`break_even_cost_bps`.
- ``longest_holding_streak`` (task-rl-10) — the longest run of consecutive steps held long;
  ``round_trip_win_rate`` — share of closed round trips that made money net of costs;
  ``position_correlation`` — Pearson correlation of two held-position paths (``corr(bh)``).
"""

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Final

from trading.domain import metrics as domain_metrics
from trading.domain.metrics import DEFAULT_ANNUALIZATION, EquityPoint
from trading.domain.rl_features import MAX_LOOKBACK
from trading.rl.bars import BarTable
from trading.rl.baselines import HOLD
from trading.rl.costs import BPS, DEFAULT_COST, CostModel

# §5.2: the random baseline is a distribution — 30 seeds per segment.
NULL_BAND_SEEDS: Final = 30
# IQM = mean of the middle 50% (Agarwal et al. 2021, rliable): trim 25% from each end.
_IQM_TRIM: Final = 0.25
# Bisection iterations: far past float resolution on [0, 1).
_BISECTION_STEPS: Final = 200


class ScoringError(ValueError):
    """A result, segment or set of results is inconsistent (fails closed, never guesses)."""


def _require_int(value: object, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoringError(f"{what} must be an int, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Segment:
    """One fixed replay window: the decision index ``start`` (bar index of the first decision),
    ``length`` decisions, the env's ``execution_lag`` and ``cost``. Replaying the same segment
    for every policy and seed is what makes their numbers comparable (rl-10's folds)."""

    start: int
    length: int
    execution_lag: int = 0
    cost: CostModel = DEFAULT_COST

    def __post_init__(self) -> None:
        _require_int(self.start, what="start")
        _require_int(self.length, what="length")
        _require_int(self.execution_lag, what="execution_lag")
        if self.start < MAX_LOOKBACK:
            raise ScoringError(f"start must be >= MAX_LOOKBACK ({MAX_LOOKBACK}), got {self.start}")
        if self.length < 1:
            raise ScoringError(f"length must be >= 1, got {self.length}")
        if not 0 <= self.execution_lag < self.length:
            raise ScoringError(
                f"execution_lag must be in [0, length), got {self.execution_lag} for {self.length}"
            )
        if not isinstance(self.cost, CostModel):
            raise ScoringError(f"cost must be a CostModel, got {self.cost!r}")

    @property
    def end(self) -> int:
        """Bar index of the last close the segment reveals (``start + length``)."""
        return self.start + self.length


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One env fill event (``info["fill"]``): indices and position fractions, never a price."""

    decision_index: int
    fill_index: int
    side: str
    position_before: float
    position_after: float
    cost_fraction: float

    @property
    def traded(self) -> float:
        """Traded notional as a fraction of equity, ``|position_after - position_before|``."""
        return abs(self.position_after - self.position_before)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One policy replayed over one :class:`Segment` with one seed, as the env accounted it.

    Mark ``i`` (``0 .. steps``) is the state at decision bar ``cursors[i]``: ``equity[i]`` (marked
    at ``marked_at[i] = bars[cursors[i]].known_at``) and ``positions[i]`` (held after fills at
    closes ``<= cursors[i]``). ``actions`` / ``requested_actions`` hold one entry per step.
    ``n_trades`` / ``turnover`` / ``cost_paid`` are the env's episode-cumulative counters;
    ``ruined`` is the env's ``terminated`` (equity hit ``ruin_fraction``; the episode stopped
    early).
    """

    segment: Segment
    seed: int
    cursors: tuple[int, ...]
    marked_at: tuple[datetime, ...]
    equity: tuple[float, ...]
    positions: tuple[float, ...]
    actions: tuple[int, ...]
    requested_actions: tuple[int, ...]
    n_noops: int
    n_trades: int
    turnover: float
    cost_paid: float
    fills: tuple[FillRecord, ...]
    ruined: bool

    def __post_init__(self) -> None:
        marks = len(self.equity)
        if marks < 1:
            raise ScoringError("an episode needs at least the starting equity mark")
        for name, size in (
            ("cursors", len(self.cursors)),
            ("marked_at", len(self.marked_at)),
            ("positions", len(self.positions)),
        ):
            if size != marks:
                raise ScoringError(f"{name} has {size} entries, equity has {marks}")
        for name, size in (
            ("actions", len(self.actions)),
            ("requested_actions", len(self.requested_actions)),
        ):
            if size != marks - 1:
                raise ScoringError(f"{name} has {size} entries for {marks - 1} steps")
        if self.n_trades != len(self.fills):
            raise ScoringError(f"n_trades {self.n_trades} != {len(self.fills)} fill records")
        if self.cursors[0] != self.segment.start:
            raise ScoringError(f"first cursor {self.cursors[0]} != start {self.segment.start}")
        if self.steps > self.segment.length or (
            not self.ruined and self.steps != self.segment.length
        ):
            raise ScoringError(
                f"{self.steps} steps for a {self.segment.length}-step segment "
                f"(ruined={self.ruined})"
            )

    @property
    def steps(self) -> int:
        return len(self.equity) - 1

    @property
    def final_equity(self) -> float:
        return self.equity[-1]


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    """The §5.4 row for one :class:`EpisodeResult` (see the module docstring for definitions)."""

    steps: int
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    turnover: float
    annualized_turnover: float
    exposure_fraction: float
    action_entropy: float
    n_trades: int
    cost_paid: float
    break_even_cost_bps: float | None
    ruined: bool


def daily_returns(result: EpisodeResult) -> tuple[float, ...]:
    """Net-of-cost simple return per step, ``E_i / E_{i-1} - 1`` (equity never hits 0 before the
    env's ruin stop, which is ``> 0`` by default; a non-positive mark contributes 0.0 as in
    ``domain.metrics.period_returns``)."""
    return tuple(
        (now / prev - 1.0) if prev > 0 else 0.0
        for prev, now in zip(result.equity, result.equity[1:], strict=False)
    )


def exposure_fraction(result: EpisodeResult) -> float:
    """Mean position held over each step's market move (positions BEFORE each step)."""
    if result.steps == 0:
        return 0.0
    return math.fsum(result.positions[:-1]) / result.steps


def action_entropy(actions: Sequence[int]) -> float:
    """Shannon entropy in nats of the empirical action distribution (0.0 when empty/constant)."""
    if not actions:
        return 0.0
    total = len(actions)
    entropy = 0.0
    for count in Counter(actions).values():
        share = count / total
        entropy -= share * math.log(share)
    return entropy


def annualized_turnover(
    result: EpisodeResult, *, annualization: float = DEFAULT_ANNUALIZATION
) -> float:
    """``turnover * annualization / steps``: equity-units traded per year at this pace."""
    if result.steps == 0:
        return 0.0
    return result.turnover * annualization / result.steps


def frictionless_growth(result: EpisodeResult) -> float:
    """The final equity the SAME position path earns at zero cost:
    ``E_n / prod(1 - cost_fraction_k)`` over the fills (every cost is a multiplicative haircut of
    the equity marked at its fill close — env step (b) — so dividing them out is exact up to float
    rounding)."""
    growth = result.final_equity
    for fill in result.fills:
        growth /= 1.0 - fill.cost_fraction
    return growth


def break_even_one_way_bps(growth: float, traded: Sequence[float]) -> float:
    """The one-way cost ``c`` (bps) at which a position path with frictionless final equity
    ``growth`` and per-fill traded notionals ``traded`` (fractions of equity) ends at exactly 1.0:

        ln(growth) + sum_k ln(1 - c * BPS * traded_k) == 0

    i.e. where its edge over always-flat (final equity exactly 1.0 at EVERY cost) nets to zero.
    The left side strictly decreases in ``c``, so the root is unique; solved by bisection (with a
    single size of trade it is the closed form ``(1 - growth ** (-1 / n)) / (BPS * traded)``).
    Returns 0.0 when there is no edge to lose (``growth <= 1``, which includes "never traded").
    """
    if not math.isfinite(growth) or growth <= 0:
        raise ScoringError(f"growth must be finite and positive, got {growth}")
    sizes = [float(t) for t in traded]
    if any(not math.isfinite(t) or t < 0 for t in sizes):
        raise ScoringError(f"traded notionals must be finite and >= 0, got {sizes}")
    target = math.log(growth)
    if target <= 0:
        return 0.0
    largest = max(sizes, default=0.0)
    if largest == 0:
        raise ScoringError(f"growth {growth} != 1 with nothing traded is not a valid path")

    def net_log_growth(c: float) -> float:
        return target + math.fsum(math.log1p(-c * t) for t in sizes)

    lo, hi = 0.0, 1.0 / largest  # net_log_growth(lo) > 0; -> -inf as c -> hi
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2.0
        if mid in (lo, hi):
            break
        if net_log_growth(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo / BPS


def break_even_cost_bps(result: EpisodeResult) -> float | None:
    """**The one-way cost, in bps of traded notional, at which this policy's edge over
    always-flat nets to zero** on this segment and seed.

    Precisely: always-flat ends at equity exactly 1.0 at every cost, so the break-even is the
    one-way ``CostModel.one_way_bps`` at which the replay's final equity would be exactly 1.0 —
    :func:`break_even_one_way_bps` of :func:`frictionless_growth` and the fills' traded notionals.
    This is exact for the env because a policy's position path does not depend on the cost: the
    observation (closes, ``held``, the entry close), the closes ``history`` and a seeded RNG are
    all cost-free, and each fill's cost is a multiplicative haircut ``(1 - c * |delta p|)`` of the
    equity (pinned against a real env replay at the solved cost by the tests).

    - ``0.0``: no edge to lose — the frictionless path already ends at or below 1.0.
    - ``None``: the replay was RUINED (stopped early), so its path is not cost-invariant (a lower
      cost could have kept it alive) and no break-even is defined.

    Exactness additionally assumes the replay at the solved cost does not RUIN: a higher
    cost lowers intermediate equity and could cross the ruin floor first (practically
    unreachable on SPY daily data, but stated for honesty).
    """
    if result.ruined:
        return None
    return break_even_one_way_bps(frictionless_growth(result), [f.traded for f in result.fills])


def _decimals(values: Sequence[float]) -> list[Decimal]:
    # Decimal(float) is the float's EXACT binary value: no rounding enters the domain functions.
    return [Decimal(v) for v in values]


def summarize(result: EpisodeResult) -> EpisodeScore:
    """The §5.4 metrics of one replay, reusing ``trading.domain.metrics`` wherever it applies."""
    returns = daily_returns(result)
    equity = _decimals(result.equity)
    curve = [EquityPoint(when, value) for when, value in zip(result.marked_at, equity, strict=True)]
    return EpisodeScore(
        steps=result.steps,
        final_equity=result.final_equity,
        total_return_pct=domain_metrics.total_return_pct(equity[0], equity[-1]),
        cagr_pct=domain_metrics.cagr_pct(curve),
        sharpe=domain_metrics.sharpe_ratio(returns),
        sortino=domain_metrics.sortino_ratio(returns),
        max_drawdown_pct=domain_metrics.max_drawdown_pct(equity),
        turnover=result.turnover,
        annualized_turnover=annualized_turnover(result),
        exposure_fraction=exposure_fraction(result),
        action_entropy=action_entropy(result.actions),
        n_trades=result.n_trades,
        cost_paid=result.cost_paid,
        break_even_cost_bps=break_even_cost_bps(result),
        ruined=result.ruined,
    )


def buy_and_hold_reference(bars: BarTable, segment: Segment) -> tuple[float, ...]:
    """The FRICTIONLESS, lag-free buy-and-hold curve over the segment's marks: ``C_i / C_start``
    for ``i = start .. start + length`` (aligned with an un-ruined :attr:`EpisodeResult.equity`).

    A benchmark, never a policy input: it "buys" at ``C_start``, the close already public at the
    first decision — the fill the env's timing contract forbids. The gap to the env's
    :func:`~trading.rl.baselines.buy_and_hold` replay (filled at ``C_{start + 1 + lag}``, paying
    costs) is the execution drag (§5.2). Equal to ``domain.metrics.buy_and_hold_curve`` on the same
    timeline (a test pins it); computed POSITIONALLY from close ratios because ``trading.rl`` never
    calls ``PriceSeries.price_at`` (its left-edge lookup returns the earliest point — rl-06 ban).
    """
    if segment.end > len(bars) - 1:
        raise ScoringError(
            f"segment ends at bar {segment.end}, beyond the last bar {len(bars) - 1}"
        )
    closes = bars.closes()
    base = closes[segment.start]
    return tuple(closes[i] / base for i in range(segment.start, segment.end + 1))


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean after trimming ``floor(0.25 * n)`` values from each end of the
    sorted sample (``scipy.stats.trim_mean(x, 0.25)``, as rliable reports it)."""
    if not values:
        raise ScoringError("IQM of an empty sample")
    ordered = sorted(values)
    cut = int(_IQM_TRIM * len(ordered))
    return math.fsum(ordered[cut : len(ordered) - cut]) / (len(ordered) - 2 * cut)


@dataclass(frozen=True, slots=True)
class NullBand:
    """The random-policy null distribution over ONE segment (§1 item 3, §5.2): per-mark equity
    percentiles across seeds (aligned with ``cursors``) plus every seed's final equity and its IQM.

    Percentiles use ``statistics.quantiles(method="inclusive")`` (linear interpolation, numpy's
    default). A ruined seed's curve is held at its last (ruin) equity — ruin is absorbing."""

    segment: Segment
    seeds: tuple[int, ...]
    cursors: tuple[int, ...]
    p05: tuple[float, ...]
    p25: tuple[float, ...]
    p50: tuple[float, ...]
    p75: tuple[float, ...]
    p95: tuple[float, ...]
    final_equity: tuple[float, ...]
    final_equity_iqm: float
    turnover: tuple[float, ...]


def null_band(results: Sequence[EpisodeResult]) -> NullBand:
    """Collapse >= 2 replays of the SAME segment (distinct seeds) into a :class:`NullBand`."""
    if len(results) < 2:
        raise ScoringError(f"a null band needs >= 2 seeds, got {len(results)}")
    segment = results[0].segment
    seeds = tuple(r.seed for r in results)
    if any(r.segment != segment for r in results):
        raise ScoringError("every result in a null band must replay the same segment")
    if len(set(seeds)) != len(seeds):
        raise ScoringError(f"null-band seeds must be distinct, got {seeds}")
    cursors = tuple(range(segment.start, segment.end + 1))
    marks = len(cursors)
    curves = []
    for result in results:
        if result.cursors != cursors[: len(result.cursors)]:
            raise ScoringError(
                f"seed {result.seed} marks bars {result.cursors[:3]}..., not {cursors[:3]}..."
            )
        curves.append(result.equity + (result.equity[-1],) * (marks - len(result.equity)))
    bands: list[list[float]] = [[] for _ in range(5)]
    for i in range(marks):
        cuts = statistics.quantiles([curve[i] for curve in curves], n=20, method="inclusive")
        for band, cut in zip(bands, (cuts[0], cuts[4], cuts[9], cuts[14], cuts[18]), strict=True):
            band.append(cut)
    finals = tuple(curve[-1] for curve in curves)
    return NullBand(
        segment=segment,
        seeds=seeds,
        cursors=cursors,
        p05=tuple(bands[0]),
        p25=tuple(bands[1]),
        p50=tuple(bands[2]),
        p75=tuple(bands[3]),
        p95=tuple(bands[4]),
        final_equity=finals,
        final_equity_iqm=iqm(finals),
        turnover=tuple(r.turnover for r in results),
    )


def matched_trade_probability(results: Sequence[EpisodeResult]) -> float:
    """The ``trade_probability`` that TURNOVER-MATCHES :func:`~trading.rl.baselines.random_policy`
    to a policy (rl-10, T2 gate): its accepted orders per decision, pooled over ``results`` —
    ``sum(non-HOLD effective actions) / sum(steps)``. Counts ACCEPTED orders (not fills), because
    that is what the random policy's propensity controls; orders still pending when an episode
    ends are discarded unfilled by the env for both policies alike."""
    steps = sum(r.steps for r in results)
    if steps == 0:
        raise ScoringError("cannot match a trade probability over zero steps")
    accepted = sum(1 for r in results for a in r.actions if a != HOLD)
    return accepted / steps


def longest_holding_streak(result: EpisodeResult) -> int:
    """The longest run of consecutive steps HELD long (positions before each step, the position
    that earns that step's move — the :func:`exposure_fraction` convention), in steps. Program doc
    §5.4's "longest holding streak" (task-rl-10): ``steps - 1 - lag`` for buy-and-hold, 0 for
    always-flat."""
    best = run = 0
    for position in result.positions[:-1]:
        run = run + 1 if position > 0 else 0
        best = max(best, run)
    return best


def round_trip_win_rate(result: EpisodeResult) -> float | None:
    """Share of CLOSED round trips (flat -> long fill, then long -> flat fill) that ended with more
    equity than they started with, net of both legs' costs: the equity marked at the exit fill's
    close vs the equity marked one bar before the entry fill (unchanged while flat, before the
    entry cost). ``None`` when no round trip closed. A position still open at the end is not
    counted (as ``domain.metrics.round_trips`` leaves open lots out)."""
    mark = {cursor: i for i, cursor in enumerate(result.cursors)}
    trips = wins = 0
    entry_equity: float | None = None
    for fill in result.fills:
        index = mark.get(fill.fill_index)
        if index is None or index == 0:
            raise ScoringError(f"fill at bar {fill.fill_index} has no preceding equity mark")
        if fill.position_before == 0 and fill.position_after > 0:
            entry_equity = result.equity[index - 1]
        elif fill.position_after == 0 and fill.position_before > 0 and entry_equity is not None:
            trips += 1
            wins += result.equity[index] > entry_equity
            entry_equity = None
    return None if trips == 0 else wins / trips


def position_correlation(result: EpisodeResult, other: EpisodeResult) -> float | None:
    """Pearson correlation of two replays' HELD position paths (positions before each step) over
    their common steps — task-rl-10's ``corr(agent, buy-and-hold)``. ``None`` when fewer than two
    common steps or either path is constant (no defined correlation)."""
    if result.segment.start != other.segment.start:
        raise ScoringError("position paths must come from the same segment start")
    n = min(result.steps, other.steps)
    if n < 2:
        return None
    try:
        return statistics.correlation(result.positions[:n], other.positions[:n])
    except statistics.StatisticsError:
        return None
