"""Data-coverage report: where is the REAL data, and where are the holes?

The platform must operate on REAL data only (anti-placeholder, see
:mod:`trading.application.data_integrity`). When synthetic data is purged, whole sources can
go empty until real ingestion fills them back in — that is acceptable, but it must be VISIBLE.
This module turns a set of recorded :class:`~trading.domain.signals.Signal` into a
:class:`CoverageReport`: per-source REAL/SYNTHETIC counts, the freshest REAL timestamp per
source, and — crucially — the explicit set of sources with ZERO real signals (the "holes").

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
    """REAL/SYNTHETIC coverage for one :class:`SignalSource`.

    ``last_real_at`` is the freshest ``created_at`` among that source's REAL signals, or
    ``None`` when the source has no real signals at all (a hole).
    """

    source_type: SignalSource
    real_count: int
    synthetic_count: int
    last_real_at: datetime | None


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Whole-system view of real vs synthetic coverage and the holes.

    ``per_source`` has exactly one entry per :class:`SignalSource` enum member, ordered by the
    enum's declaration order, so a source with no signals at all is still reported (as a hole)
    rather than silently omitted. ``empty_sources`` is the subset of sources with zero REAL
    signals — the holes the user wants surfaced once synthetic data is gone.
    """

    per_source: tuple[SourceCoverage, ...]
    total_real: int
    total_synthetic: int
    empty_sources: tuple[SignalSource, ...]


def coverage_report(signals: Sequence[Signal]) -> CoverageReport:
    """Compute a :class:`CoverageReport` over every :class:`SignalSource`.

    Iterates ALL enum members (not just the sources present in ``signals``) so holes are
    explicit: a source with no real data appears in ``per_source`` with ``real_count == 0``
    and ``last_real_at is None``, and is listed in ``empty_sources``. Deterministic: output
    depends only on the input signals, and source order follows the enum declaration order.
    """
    real_counts: dict[SignalSource, int] = dict.fromkeys(SignalSource, 0)
    synthetic_counts: dict[SignalSource, int] = dict.fromkeys(SignalSource, 0)
    # Dict literal (not dict.fromkeys) so the value type widens to `datetime | None`; the
    # comprehension form keeps mypy's invariant assignment check happy.
    last_real: dict[SignalSource, datetime | None] = {s: None for s in SignalSource}  # noqa: C420

    for signal in signals:
        source = signal.source_type
        if signal.provenance is Provenance.REAL:
            real_counts[source] += 1
            current = last_real[source]
            if current is None or signal.created_at > current:
                last_real[source] = signal.created_at
        else:
            synthetic_counts[source] += 1

    per_source = tuple(
        SourceCoverage(
            source_type=source,
            real_count=real_counts[source],
            synthetic_count=synthetic_counts[source],
            last_real_at=last_real[source],
        )
        for source in SignalSource
    )
    empty_sources = tuple(cov.source_type for cov in per_source if cov.real_count == 0)
    total_real = sum(cov.real_count for cov in per_source)
    total_synthetic = sum(cov.synthetic_count for cov in per_source)
    return CoverageReport(
        per_source=per_source,
        total_real=total_real,
        total_synthetic=total_synthetic,
        empty_sources=empty_sources,
    )
