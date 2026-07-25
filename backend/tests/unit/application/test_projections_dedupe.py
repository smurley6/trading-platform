"""Read-side dedupe of duplicated `signal.received` events (review F1).

The daily cron re-appends its whole lookback every run, so the append-only log can hold up
to ~21 copies of one signal. `signals_from_events` collapses copies (first occurrence per
id), which must make the reconstructed window — and therefore the conviction score,
decayed weighted sum, corroboration, and per-signal contributions — identical whether the
log holds 1 copy or 21. It also repairs the audit invariant: with duplicates counted, the
weighted sum inflated while `contributions` collapsed by id, so contributions no longer
summed to the score.
"""

from datetime import UTC, datetime, timedelta

import pytest

from tests.conftest import FixedClock
from trading.application.projections import signals_from_events
from trading.application.signal_recording import record_signals
from trading.domain.conviction import ConvictionParams, ConvictionThresholds, compute_conviction
from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource
from trading.infrastructure.persistence.memory import InMemoryEventStore

_WINDOW_DAYS = 21
_NOW = datetime(2026, 6, 12, 14, 30, tzinfo=UTC)


def _signal(sid: str, source: SignalSource, age_days: int) -> Signal:
    return Signal(
        id=sid,
        created_at=_NOW - timedelta(days=age_days),
        source_type=source,
        ticker="AAPL",
        politician_id="pol-1",
        sentiment=0.8,
        politician_weight=1.2,
        source_reliability=0.9,
        raw_payload={"source": "test-fixture"},
        provenance=Provenance.REAL,
        pit_confidence="exact",
    )


# Two distinct source types on AAPL at different ages: exercises BOTH decay (the copies
# share a timestamp, so a naive per-copy sum would not attenuate) AND corroboration.
_SIGNALS = (
    _signal("sig-a", SignalSource.STATEMENT, age_days=1),
    _signal("sig-b", SignalSource.DISCLOSED_TRADE, age_days=5),
)
# Decay ON is the exact failure mode: identical duplicate timestamps do not attenuate.
_PARAMS = ConvictionParams(decay_half_life_days=7.0)
_THRESHOLDS = ConvictionThresholds()


async def _score_with_copies(n_copies: int) -> tuple[float, float, dict[str, float], float]:
    """Build a log holding `n_copies` of each signal, then reconstruct + score AAPL."""
    clock = FixedClock(_NOW)
    store = InMemoryEventStore(clock)
    for _ in range(n_copies):
        # Default (skip_existing_streams=False) so duplicates DO land — this is the corrupted
        # log the cron produced before the fix, exactly what the read side must now heal.
        await record_signals(
            event_store=store,
            clock=clock,
            signals=_SIGNALS,
            correlation_id=f"corr-{_}",
        )
    window = signals_from_events(await store.read_all(), now=_NOW, window_days=_WINDOW_DAYS)
    result = compute_conviction("AAPL", window, _WINDOW_DAYS, _THRESHOLDS, _PARAMS)
    return (
        result.score,
        result.decayed_weighted_sum,
        dict(result.contributions),
        sum(result.contributions.values()),
    )


@pytest.mark.parametrize("n_copies", [1, 3, 21])
async def test_conviction_is_invariant_to_duplicate_copies(n_copies: int) -> None:
    baseline = await _score_with_copies(1)
    score, decayed, contributions, _contrib_sum = await _score_with_copies(n_copies)

    base_score, base_decayed, base_contributions, _ = baseline
    assert score == base_score
    assert decayed == base_decayed
    assert contributions == base_contributions
    # Only the two distinct ids ever appear — copies never add a third contribution.
    assert set(contributions) == {"sig-a", "sig-b"}


@pytest.mark.parametrize("n_copies", [1, 3, 21])
async def test_contributions_sum_to_the_decayed_weighted_sum(n_copies: int) -> None:
    # The audit invariant (spec section 12): contributions sum to the decayed weighted sum,
    # which duplicate-counting broke (sum inflated, dict collapsed by id).
    _, decayed, _, contrib_sum = await _score_with_copies(n_copies)
    assert contrib_sum == pytest.approx(decayed)


async def test_dedupe_keeps_the_earliest_recorded_copy() -> None:
    # First occurrence in recording order is the earliest append — PIT-correct, never a later
    # copy. All copies are payload-identical, so the kept signal round-trips faithfully.
    clock = FixedClock(_NOW)
    store = InMemoryEventStore(clock)
    for _ in range(5):
        await record_signals(
            event_store=store,
            clock=clock,
            signals=(_SIGNALS[0],),
            correlation_id="corr",
        )
    window = signals_from_events(await store.read_all())
    assert len(window) == 1
    assert window[0].id == "sig-a"
    assert window[0].created_at == _SIGNALS[0].created_at
