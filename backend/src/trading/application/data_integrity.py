"""Data-integrity guard: keep SYNTHETIC data out of real decisions (anti-placeholder).

The platform may act on a real broker ONLY with REAL data. This module centralises that rule —
one :class:`SyntheticDataError` and the helpers the enforcement points share — so the policy
lives in exactly one place instead of being re-implemented at each gate.

Synthetic data stays permitted in tests and labelled backtests, which never reach a real
broker. The gates fire only when ``require_real_data`` is set, which the composition root turns
on automatically whenever a sleeve routes to a real Alpaca account (see ``main``). The default
is fail-closed: an un-stamped Signal is SYNTHETIC, so forgetting to mark a new source REAL
blocks it rather than silently trusting it.
"""

from collections.abc import Iterable, Sequence

from trading.domain.provenance import Provenance
from trading.domain.signals import Signal


class SyntheticDataError(Exception):
    """Raised when SYNTHETIC (fake / placeholder / seeded) data reaches a real decision path."""


def non_real_signals(signals: Iterable[Signal]) -> list[Signal]:
    """Return the signals that are not REAL (everything the guard would reject)."""
    return [s for s in signals if s.provenance is not Provenance.REAL]


def ensure_real_signals(signals: Sequence[Signal]) -> None:
    """Raise :class:`SyntheticDataError` if ANY signal is not REAL.

    The last line of defence on the real-decision path: even if a synthetic signal slipped past
    the ingestion gate, no real order can be driven by it.
    """
    fake = non_real_signals(signals)
    if fake:
        sample = ", ".join(sorted(s.id for s in fake)[:10])
        raise SyntheticDataError(
            f"refusing to act on {len(fake)} non-REAL signal(s) in a real decision path "
            f"(require_real_data is on); offending ids: {sample}"
        )
