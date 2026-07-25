"""Data-coverage report: where is the REAL data, and where are the holes?

The platform must operate on REAL data only for execution (anti-placeholder, see
:mod:`trading.application.data_integrity`). When synthetic data is purged, whole sources can
go empty until real ingestion fills them back in — that is acceptable, but it must be VISIBLE.
This module turns a set of recorded :class:`~trading.domain.signals.Signal` into a
:class:`CoverageReport`: per-source REAL/BACKFILLED/SYNTHETIC counts, the freshest REAL
timestamp per source, and — crucially — the explicit set of sources with ZERO real signals
(the "holes").

REAL, BACKFILLED, and SYNTHETIC are reported as three SEPARATE counts (task-08). BACKFILLED is
real, historically reconstructed data that belongs in projections/labeler/learner
(``docs/RESEARCH-PLAN.md`` §2) and must survive a purge — it must never be absorbed into the
synthetic count, which would make it indistinguishable from purge-eligible fake data.

It is pure application logic: it imports only the domain and computes deterministically over
an in-memory ``Sequence[Signal]``. It performs no I/O and reaches no infrastructure, so it
stays inside the hexagonal layer contract and is trivially testable. The purge job and any
interface endpoint share this single computation rather than re-deriving counts ad hoc.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from trading.domain.provenance import Provenance
from trading.domain.signals import Signal, SignalSource


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """REAL/BACKFILLED/SYNTHETIC coverage for one :class:`SignalSource`.

    ``last_real_at`` is the freshest ``created_at`` among that source's REAL signals, or
    ``None`` when the source has no real signals at all (a hole).
    """

    source_type: SignalSource
    real_count: int
    backfilled_count: int
    synthetic_count: int
    last_real_at: datetime | None


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Whole-system view of real/backfilled/synthetic coverage and the holes.

    ``per_source`` has exactly one entry per :class:`SignalSource` enum member, ordered by the
    enum's declaration order, so a source with no signals at all is still reported (as a hole)
    rather than silently omitted. ``empty_sources`` is the subset of sources with zero REAL
    signals — the holes the user wants surfaced once synthetic data is gone. ``total_backfilled``
    is reported separately from ``total_synthetic`` (task-08): BACKFILLED is real data that must
    survive a purge, so folding it into the synthetic count would misreport it as purge-eligible.
    """

    per_source: tuple[SourceCoverage, ...]
    total_real: int
    total_backfilled: int
    total_synthetic: int
    empty_sources: tuple[SignalSource, ...]


def coverage_report(signals: Sequence[Signal]) -> CoverageReport:
    """Compute a :class:`CoverageReport` over every :class:`SignalSource`.

    Iterates ALL enum members (not just the sources present in ``signals``) so holes are
    explicit: a source with no real data appears in ``per_source`` with ``real_count == 0``
    and ``last_real_at is None``, and is listed in ``empty_sources``. Deterministic: output
    depends only on the input signals, and source order follows the enum declaration order.
    BACKFILLED and SYNTHETIC are counted separately (task-08) — BACKFILLED is neither REAL nor
    purge-eligible, so it gets its own bucket instead of falling into ``synthetic_count``.
    """
    real_counts: dict[SignalSource, int] = dict.fromkeys(SignalSource, 0)
    backfilled_counts: dict[SignalSource, int] = dict.fromkeys(SignalSource, 0)
    synthetic_counts: dict[SignalSource, int] = dict.fromkeys(SignalSource, 0)
    last_real: dict[SignalSource, datetime | None] = dict.fromkeys(SignalSource)

    for signal in signals:
        source = signal.source_type
        if signal.provenance is Provenance.REAL:
            real_counts[source] += 1
            current = last_real[source]
            if current is None or signal.created_at > current:
                last_real[source] = signal.created_at
        elif signal.provenance is Provenance.BACKFILLED:
            backfilled_counts[source] += 1
        else:
            synthetic_counts[source] += 1

    per_source = tuple(
        SourceCoverage(
            source_type=source,
            real_count=real_counts[source],
            backfilled_count=backfilled_counts[source],
            synthetic_count=synthetic_counts[source],
            last_real_at=last_real[source],
        )
        for source in SignalSource
    )
    empty_sources = tuple(cov.source_type for cov in per_source if cov.real_count == 0)
    total_real = sum(cov.real_count for cov in per_source)
    total_backfilled = sum(cov.backfilled_count for cov in per_source)
    total_synthetic = sum(cov.synthetic_count for cov in per_source)
    return CoverageReport(
        per_source=per_source,
        total_real=total_real,
        total_backfilled=total_backfilled,
        total_synthetic=total_synthetic,
        empty_sources=empty_sources,
    )
