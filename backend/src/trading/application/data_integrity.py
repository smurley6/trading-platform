"""Data-integrity guard: keep SYNTHETIC/BACKFILLED data out of real decisions.

The platform may act on a real broker ONLY with REAL data. This module centralises that rule —
one :class:`SyntheticDataError` and the helpers the enforcement points share — so the policy
lives in exactly one place instead of being re-implemented at each gate.

There are two DIFFERENT execution walls for the two non-REAL provenance tiers (see
``docs/RESEARCH-PLAN.md`` §2-§3), and they are NOT interchangeable:

- SYNTHETIC — fully quarantined. :func:`ensure_real_signals` ABORTS the whole evaluation
  (raises :class:`SyntheticDataError`) if even one SYNTHETIC signal is present, because fake
  data anywhere in the batch means the batch cannot be trusted.
- BACKFILLED — real, historically reconstructed data that must never itself produce a decision
  or order, but MUST NOT abort a live evaluation (a live cycle can legitimately mix live-real
  signals with in-window backfilled ones; aborting the whole batch would halt live trading
  until the backfilled signals age out of the corroboration window — a functional regression).
  :func:`tradeable_signals` FILTERS BACKFILLED (and SYNTHETIC) out of the strategy's input
  instead of raising, so REAL signals in the same batch still evaluate normally.

Synthetic data stays permitted in tests and labelled backtests, which never reach a real
broker. Backfilled data stays permitted in projections and (future) labeler/learner paths — it
is simply never fed to a strategy. The gates fire only when ``require_real_data`` is set, which
the composition root turns on automatically whenever a sleeve routes to a real Alpaca account
(see ``main``). The default is fail-closed: an un-stamped Signal is SYNTHETIC, so forgetting to
mark a new source REAL blocks it rather than silently trusting it.
"""

from collections.abc import Iterable, Sequence

from trading.domain.provenance import Provenance
from trading.domain.signals import Signal


class SyntheticDataError(Exception):
    """Raised when SYNTHETIC (fake / placeholder / seeded) data reaches a real decision path."""


def non_real_signals(signals: Iterable[Signal]) -> list[Signal]:
    """Return the signals that are not REAL (everything the guard would reject)."""
    return [s for s in signals if s.provenance is not Provenance.REAL]


def tradeable_signals(signals: Iterable[Signal]) -> list[Signal]:
    """Drop BACKFILLED signals — the execution wall for the evaluate/place path.

    BACKFILLED signals are real data but must NEVER produce a decision or order intent (the
    provenance wall, RESEARCH-PLAN §2): by definition they are ingested long after their true
    disclosure timestamp, so acting on one now would not be a live trade at all. Unlike
    :func:`ensure_real_signals`, this FILTERS rather than aborts: a live evaluation batch that
    happens to include in-window backfilled signals still trades its other signals instead of
    halting entirely.

    This does NOT also filter SYNTHETIC: whether SYNTHETIC signals may reach the strategy is
    governed entirely by ``require_real_data`` + :func:`ensure_real_signals` (fail-open in tests
    and backtests, fail-closed once a real broker is wired), a policy this filter must not
    duplicate or override.
    """
    return [s for s in signals if s.provenance is not Provenance.BACKFILLED]


def ensure_real_signals(signals: Sequence[Signal]) -> None:
    """Raise :class:`SyntheticDataError` if ANY signal is SYNTHETIC.

    The last line of defence on the real-decision path: even if a synthetic signal slipped past
    the ingestion gate, no real order can be driven by it. This check is specific to SYNTHETIC —
    it does NOT fire on BACKFILLED signals, which use the separate filter-not-abort wall in
    :func:`tradeable_signals` so they can coexist with REAL signals in the same live batch
    without halting evaluation.
    """
    fake = [s for s in signals if s.provenance is Provenance.SYNTHETIC]
    if fake:
        sample = ", ".join(sorted(s.id for s in fake)[:10])
        raise SyntheticDataError(
            f"refusing to act on {len(fake)} SYNTHETIC signal(s) in a real decision path "
            f"(require_real_data is on); offending ids: {sample}"
        )


def ensure_no_synthetic(signals: Iterable[Signal]) -> None:
    """Raise :class:`SyntheticDataError` if ANY signal is SYNTHETIC — the labeler's guard.

    Distinct from :func:`ensure_real_signals` in intent, though it shares the SYNTHETIC-only
    rule: the labeler admits BOTH REAL and BACKFILLED provenance (backfilled data is real,
    historically reconstructed data that IS permitted into the research/labeler path — see
    ``docs/RESEARCH-PLAN.md`` §2), but SYNTHETIC (fabricated/placeholder) data must never
    produce a stored outcome label. The outcome labeler applies this to its input so a caller
    that hands it explicit SYNTHETIC signals is refused (RESEARCH-PLAN §2: "SYNTHETIC never
    trains"). Unlike :func:`tradeable_signals` this ABORTS rather than filters, because a
    label computed from fake data is meaningless research output, not a live-trade risk.
    """
    fake = [s for s in signals if s.provenance is Provenance.SYNTHETIC]
    if fake:
        sample = ", ".join(sorted(s.id for s in fake)[:10])
        raise SyntheticDataError(
            f"refusing to label {len(fake)} SYNTHETIC signal(s): synthetic data never enters "
            f"the labeler/learner (RESEARCH-PLAN §2); offending ids: {sample}"
        )
