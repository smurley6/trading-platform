"""Data provenance: is a piece of decision-driving data REAL, BACKFILLED, or SYNTHETIC?

The platform must never place a real (paper/live) order on fabricated inputs (the
anti-placeholder guarantee). Every :class:`~trading.domain.signals.Signal` and
:class:`~trading.domain.politicians.Politician` carries a :class:`Provenance`, and the default
is ``SYNTHETIC`` (fail-closed) — anything that forgets to declare itself real is treated as
untrusted. ONLY genuine live data adapters stamp ``REAL``.

The data-integrity guard (:mod:`trading.application.data_integrity`) refuses to act on
``SYNTHETIC`` data anywhere in a real decision path. Synthetic data remains permitted ONLY in
tests and clearly-labelled backtests, which never reach a real broker.

``BACKFILLED`` is a THIRD tier (see ``docs/RESEARCH-PLAN.md`` §2-§3): real historical events
reconstructed after the fact (e.g. a Form 4 filing ingested months after its acceptance
datetime). Unlike SYNTHETIC, backfilled data is real and is permitted into projections, the
labeler, and the learner — it is simply never allowed to drive a live decision, because a
backfilled signal by definition arrives too late to be actionable. The execution wall for this
tier is a FILTER (drop from the evaluation input), not an ABORT — see
:mod:`trading.application.data_integrity`.
"""

from enum import StrEnum
from typing import Literal

# Per-signal point-in-time confidence, inherited from the source tier table (RESEARCH-PLAN §3):
# "exact" — disclosure/acceptance timestamp is authoritative (e.g. EDGAR ACCEPTANCE-DATETIME).
# "approximate" — timestamp has undocumented lag (e.g. scrape-only sources).
# "excluded" — no trustworthy point-in-time timestamp exists at all.
PitConfidence = Literal["exact", "approximate", "excluded"]


class Provenance(StrEnum):
    """Origin trust level of decision-driving data.

    REAL — sourced from a genuine, live external feed (real market/news/disclosure data tied
    to real-world entities). BACKFILLED — real, historically reconstructed data with a true
    point-in-time disclosure timestamp; never drives a live decision, but is real data for
    research purposes. SYNTHETIC — fabricated, seeded, placeholder, or test data. The default
    everywhere is SYNTHETIC so that an un-stamped value is never mistaken for real.
    """

    REAL = "real"
    BACKFILLED = "backfilled"
    SYNTHETIC = "synthetic"
