"""Data provenance: is a piece of decision-driving data REAL or SYNTHETIC?

The platform must never place a real (paper/live) order on fabricated inputs (the
anti-placeholder guarantee). Every :class:`~trading.domain.signals.Signal` and
:class:`~trading.domain.politicians.Politician` carries a :class:`Provenance`, and the default
is ``SYNTHETIC`` (fail-closed) — anything that forgets to declare itself real is treated as
untrusted. ONLY genuine live data adapters stamp ``REAL``.

The data-integrity guard (:mod:`trading.application.data_integrity`) refuses to act on
``SYNTHETIC`` data anywhere in a real decision path. Synthetic data remains permitted ONLY in
tests and clearly-labelled backtests, which never reach a real broker.
"""

from enum import StrEnum


class Provenance(StrEnum):
    """Origin trust level of decision-driving data.

    REAL — sourced from a genuine, live external feed (real market/news/disclosure data tied
    to real-world entities). SYNTHETIC — fabricated, seeded, placeholder, or test data. The
    default everywhere is SYNTHETIC so that an un-stamped value is never mistaken for real.
    """

    REAL = "real"
    SYNTHETIC = "synthetic"
