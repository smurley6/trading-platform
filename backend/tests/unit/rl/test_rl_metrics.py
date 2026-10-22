"""Pure RL evaluation metrics (task-rl-07; program doc §5.2, §5.4) on hand-built results.

No gymnasium: every :class:`EpisodeResult` here is constructed by hand so each metric is checked
against an arithmetic the test states. ``test_scoring.py`` checks the same metrics on real env
replays (incl. the break-even cost against a replay AT that cost).
"""

import dataclasses
import math
import statistics
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tests.unit.rl.conftest import bar_table, synthetic_random_walk
from trading.domain.metrics import buy_and_hold_curve, sharpe_ratio
from trading.domain.prices import PriceSeries
from trading.rl.baselines import BUY, HOLD, SELL
from trading.rl.costs import CostModel
from trading.rl.metrics import (
    NULL_BAND_SEEDS,
    EpisodeResult,
    FillRecord,
    ScoringError,
    Segment,
    action_entropy,
    annualized_turnover,
    break_even_cost_bps,
    break_even_one_way_bps,
    buy_and_hold_reference,
    daily_returns,
    exposure_fraction,
    frictionless_growth,
    iqm,
    matched_trade_probability,
    null_band,
    summarize,
)

_T0 = datetime(2024, 1, 2, 21, 15, tzinfo=UTC)


def _fill(at: int, before: float, after: float, cost_bps: float) -> FillRecord:
    return FillRecord(
        decision_index=at - 1,
        fill_index=at,
        side="buy" if after > before else "sell",
        position_before=before,
        position_after=after,
        cost_fraction=cost_bps * 1e-4 * abs(after - before),
    )


def _result(
    equity: list[float],
    positions: list[float],
    actions: list[int],
    *,
    fills: tuple[FillRecord, ...] = (),
    start: int = 100,
    length: int | None = None,
    seed: int = 0,
    cost_bps: float = 0.0,
    ruined: bool = False,
) -> EpisodeResult:
    steps = len(equity) - 1
    return EpisodeResult(
        segment=Segment(start, steps if length is None else length, cost=CostModel(cost_bps)),
        seed=seed,
        cursors=tuple(range(start, start + steps + 1)),
        marked_at=tuple(_T0 + timedelta(days=i) for i in range(steps + 1)),
        equity=tuple(equity),
        positions=tuple(positions),
        actions=tuple(actions),
        requested_actions=tuple(actions),
        n_noops=0,
        n_trades=len(fills),
        turnover=math.fsum(f.traded for f in fills),
        cost_paid=0.0,
        fills=fills,
        ruined=ruined,
    )


# --- Segment / EpisodeResult validation ------------------------------------------------------


def test_segment_validates_and_reports_its_end() -> None:
    assert Segment(49, 63).end == 112
    for bad in [
        {"start": 48, "length": 10},
        {"start": 49, "length": 0},
        {"start": 49, "length": 10, "execution_lag": 10},
        {"start": 49, "length": 10, "execution_lag": -1},
        {"start": 49.0, "length": 10},
        {"start": 49, "length": True},
        {"start": 49, "length": 10, "cost": 1.0},
    ]:
        with pytest.raises(ScoringError):
            Segment(**bad)  # type: ignore[arg-type]


def test_episode_result_rejects_inconsistent_records() -> None:
    good = _result([1.0, 1.1, 1.2], [0.0, 1.0, 1.0], [BUY, HOLD])
    assert good.steps == 2 and good.final_equity == 1.2
    with pytest.raises(ScoringError, match="positions"):
        _result([1.0, 1.1], [0.0], [HOLD])
    with pytest.raises(ScoringError, match="actions"):
        _result([1.0, 1.1], [0.0, 0.0], [HOLD, HOLD])
    with pytest.raises(ScoringError, match="steps"):  # short of the segment, yet not ruined
        _result([1.0, 1.1], [0.0, 0.0], [HOLD], length=5)
    with pytest.raises(ScoringError, match="n_trades"):
        dataclasses.replace(good, n_trades=1)
    with pytest.raises(ScoringError, match="first cursor"):
        dataclasses.replace(good, cursors=(101, 102, 103))
    with pytest.raises(ScoringError, match="starting equity"):
        _result([], [], [], length=1)
    # A ruined episode may stop early.
    assert _result([1.0, 0.1], [1.0, 1.0], [HOLD], length=5, ruined=True).steps == 1


# --- per-episode metrics ---------------------------------------------------------------------


def test_daily_returns_exposure_and_turnover_by_hand() -> None:
    # Flat, BUY (fills at mark 2), held 2 steps, SELL (fills at mark 4), flat.
    result = _result(
        [1.0, 1.0, 1.0, 1.1, 1.21, 1.21],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [BUY, HOLD, SELL, HOLD, HOLD],
        fills=(_fill(102, 0.0, 1.0, 0.0), _fill(104, 1.0, 0.0, 0.0)),
    )
    assert daily_returns(result) == pytest.approx((0.0, 0.0, 0.1, 0.1, 0.0))
    # Positions held over the 5 steps: 0, 0, 1, 1, 0 -> 2 / 5.
    assert exposure_fraction(result) == 0.4
    assert result.turnover == 2.0
    assert annualized_turnover(result) == pytest.approx(2.0 * 252 / 5)


def test_exposure_is_one_for_always_long_marks_and_zero_for_flat() -> None:
    long = _result([1.0, 1.01, 1.02], [1.0, 1.0, 1.0], [HOLD, HOLD])
    flat = _result([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [HOLD, HOLD])
    assert exposure_fraction(long) == 1.0
    assert exposure_fraction(flat) == 0.0


def test_action_entropy_in_nats() -> None:
    assert action_entropy([]) == 0.0
    assert action_entropy([HOLD] * 40) == 0.0
    assert action_entropy([HOLD, BUY] * 10) == pytest.approx(math.log(2))
    assert action_entropy([HOLD, BUY, SELL] * 7) == pytest.approx(math.log(3))
    p = [0.5, 0.25, 0.25]
    assert action_entropy([HOLD, HOLD, BUY, SELL]) == pytest.approx(
        -sum(x * math.log(x) for x in p)
    )


def test_summarize_reuses_the_domain_metrics() -> None:
    equity = [1.0, 1.02, 0.99, 1.03, 1.01]
    result = _result(
        equity, [0.0, 1.0, 1.0, 1.0, 1.0], [BUY, HOLD, HOLD, HOLD], fills=(_fill(101, 0, 1, 0),)
    )
    score = summarize(result)
    returns = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity))]
    assert score.sharpe == pytest.approx(sharpe_ratio(returns))
    assert score.sharpe == pytest.approx(
        statistics.fmean(returns) / statistics.pstdev(returns) * math.sqrt(252)
    )
    assert score.total_return_pct == pytest.approx(1.0)
    # Peak 1.03 -> trough 1.01 is smaller than 1.02 -> 0.99.
    assert score.max_drawdown_pct == pytest.approx((1.02 - 0.99) / 1.02 * 100)
    assert score.steps == 4 and score.n_trades == 1 and score.final_equity == 1.01
    assert score.exposure_fraction == 0.75
    # Marks one calendar day apart: 4 days of span, compounded over 365.25 days.
    assert score.cagr_pct == pytest.approx((1.01 ** (365.25 / 4) - 1) * 100)


# --- break-even cost -------------------------------------------------------------------------


def test_break_even_matches_the_closed_form_for_equal_trades() -> None:
    for growth, n in [(1.10, 1), (1.10, 2), (1.02, 40), (3.5, 1), (1.0001, 7)]:
        expected = (1.0 - growth ** (-1.0 / n)) / 1e-4
        assert break_even_one_way_bps(growth, [1.0] * n) == pytest.approx(expected, rel=1e-12)


def test_break_even_nets_the_edge_to_exactly_zero_with_unequal_trades() -> None:
    traded = [1.0, 0.5, 0.25, 1.0]
    growth = 1.07
    c = break_even_one_way_bps(growth, traded) * 1e-4
    final = growth * math.prod(1.0 - c * t for t in traded)
    assert final == pytest.approx(1.0, abs=1e-12)
    # Monotone: more trading at the same frictionless growth breaks even sooner.
    assert break_even_one_way_bps(growth, [*traded, 1.0]) < break_even_one_way_bps(growth, traded)


def test_break_even_is_zero_without_an_edge() -> None:
    assert break_even_one_way_bps(1.0, []) == 0.0  # always-flat
    assert break_even_one_way_bps(1.0, [1.0, 1.0]) == 0.0  # round trip that earned nothing
    assert break_even_one_way_bps(0.9, [1.0]) == 0.0  # lost money even frictionless


def test_break_even_rejects_impossible_inputs() -> None:
    with pytest.raises(ScoringError, match="nothing traded"):
        break_even_one_way_bps(1.2, [])
    for bad_growth in [0.0, -1.0, float("nan"), float("inf")]:
        with pytest.raises(ScoringError, match="growth"):
            break_even_one_way_bps(bad_growth, [1.0])
    with pytest.raises(ScoringError, match="traded"):
        break_even_one_way_bps(1.2, [-1.0])


def test_frictionless_growth_divides_out_each_fill_cost() -> None:
    # BUY at 5 bps, market +10%, SELL at 5 bps: final = (1 - c) * 1.1 * (1 - c).
    c = 5e-4
    final = (1 - c) * 1.1 * (1 - c)
    result = _result(
        [1.0, 1.0 - c, (1.0 - c) * 1.1 * (1 - c)],
        [0.0, 1.0, 0.0],
        [BUY, SELL],
        fills=(_fill(101, 0, 1, 5.0), _fill(102, 1, 0, 5.0)),
        cost_bps=5.0,
    )
    assert result.final_equity == pytest.approx(final)
    assert frictionless_growth(result) == pytest.approx(1.1, rel=1e-14)
    expected = (1.0 - 1.1 ** (-1.0 / 2)) / 1e-4
    assert break_even_cost_bps(result) == pytest.approx(expected, rel=1e-12)


def test_break_even_is_undefined_for_a_ruined_episode() -> None:
    ruined = _result([1.0, 0.1], [1.0, 1.0], [HOLD], length=5, ruined=True)
    assert break_even_cost_bps(ruined) is None
    assert summarize(ruined).break_even_cost_bps is None


# --- buy-and-hold reference ------------------------------------------------------------------


def test_buy_and_hold_reference_equals_the_domain_benchmark() -> None:
    closes = synthetic_random_walk(300, seed=3)
    table = bar_table(closes)
    segment = Segment(120, 100)
    reference = buy_and_hold_reference(table, segment)
    assert len(reference) == segment.length + 1
    assert reference[0] == 1.0
    assert reference[-1] == pytest.approx(closes[220] / closes[120], rel=1e-15)
    # Parity with domain.metrics.buy_and_hold_curve on the SAME timeline (exact timestamps, so its
    # price lookup never takes the left-edge path).
    series = PriceSeries(series={"SPY": tuple((b.session_ts, b.close) for b in table.bars)})
    timeline = [table.bars[i].session_ts for i in range(segment.start, segment.end + 1)]
    domain_curve = buy_and_hold_curve(["SPY"], series, timeline, Decimal(1))
    assert [float(p.equity) for p in domain_curve] == pytest.approx(reference, rel=1e-12)


def test_buy_and_hold_reference_rejects_a_segment_past_the_data() -> None:
    table = bar_table(synthetic_random_walk(100, seed=1))
    with pytest.raises(ScoringError, match="beyond"):
        buy_and_hold_reference(table, Segment(60, 40))


# --- null band, IQM, turnover matching -------------------------------------------------------


def test_iqm_trims_a_quarter_from_each_end() -> None:
    assert iqm([5.0]) == 5.0
    assert iqm([1.0, 2.0, 3.0, 4.0]) == 2.5  # trims 1 each side
    assert iqm([1.0, 2.0, 3.0, 100.0, -50.0, 4.0, 5.0, 6.0]) == pytest.approx(3.5)  # mean(2..5)
    values = [float(v) for v in range(30)]  # trims 7 each side -> mean(7..22)
    assert iqm(values) == pytest.approx(statistics.fmean(range(7, 23)))
    with pytest.raises(ScoringError):
        iqm([])


def test_null_band_percentiles_per_mark() -> None:
    finals = [1.0 + 0.01 * k for k in range(NULL_BAND_SEEDS)]
    results = [
        _result([1.0, (1.0 + f) / 2, f], [0.0, 0.0, 0.0], [HOLD, HOLD], seed=k)
        for k, f in enumerate(finals)
    ]
    band = null_band(results)
    assert band.seeds == tuple(range(NULL_BAND_SEEDS))
    assert band.cursors == (100, 101, 102)
    assert band.p50[0] == band.p05[0] == band.p95[0] == 1.0
    cuts = statistics.quantiles(finals, n=20, method="inclusive")
    assert (band.p05[-1], band.p25[-1], band.p50[-1], band.p75[-1], band.p95[-1]) == (
        cuts[0],
        cuts[4],
        cuts[9],
        cuts[14],
        cuts[18],
    )
    assert band.p25[-1] == pytest.approx(1.0725)  # numpy-linear 25th percentile of 1.00..1.29
    assert band.final_equity == tuple(finals)
    assert band.final_equity_iqm == pytest.approx(iqm(finals))
    assert band.p05[-1] <= band.p25[-1] <= band.p50[-1] <= band.p75[-1] <= band.p95[-1]


def test_null_band_holds_a_ruined_seed_at_its_ruin_equity() -> None:
    alive = _result([1.0, 1.1, 1.2], [1.0, 1.0, 1.0], [HOLD, HOLD], seed=0)
    ruined = _result([1.0, 0.1], [1.0, 1.0], [HOLD], length=2, seed=1, ruined=True)
    band = null_band([alive, ruined])
    assert band.final_equity == (1.2, 0.1)
    assert band.p50 == pytest.approx((1.0, 0.6, 0.65))


def test_null_band_rejects_mismatched_inputs() -> None:
    a = _result([1.0, 1.1], [0.0, 0.0], [HOLD], seed=0)
    with pytest.raises(ScoringError, match=">= 2"):
        null_band([a])
    with pytest.raises(ScoringError, match="distinct"):
        null_band([a, a])
    other = _result([1.0, 1.1], [0.0, 0.0], [HOLD], seed=1, start=101)
    with pytest.raises(ScoringError, match="same segment"):
        null_band([a, other])


def test_matched_trade_probability_counts_accepted_orders_per_decision() -> None:
    a = _result([1.0] * 5, [0.0] * 5, [BUY, HOLD, SELL, HOLD])
    b = _result([1.0] * 5, [0.0] * 5, [HOLD, HOLD, HOLD, BUY], seed=1)
    assert matched_trade_probability([a]) == 0.5
    assert matched_trade_probability([a, b]) == 3 / 8
    with pytest.raises(ScoringError):
        matched_trade_probability([])
