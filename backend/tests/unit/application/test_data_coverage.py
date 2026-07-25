"""Coverage report: per-source REAL/SYNTHETIC counts, last_real_at, and the holes.

This is how the user "knows where the holes are" once synthetic data is purged: every
SignalSource is reported, even one with zero signals, and the empty (no-real-data) sources are
called out explicitly.
"""

from datetime import timedelta

from tests.conftest import FixedClock, SignalFactory
from trading.application.data_coverage import CoverageReport, SourceCoverage, coverage_report
from trading.domain.provenance import Provenance
from trading.domain.signals import SignalSource


def _by_source(report: CoverageReport) -> dict[SignalSource, SourceCoverage]:
    # Small helper: index per_source by its SignalSource for assertions.
    return {cov.source_type: cov for cov in report.per_source}


def test_every_source_is_reported_even_when_empty() -> None:
    # No signals at all: every enum member appears as a hole, all counts zero.
    report = coverage_report([])

    assert report.total_real == 0
    assert report.total_synthetic == 0
    assert {cov.source_type for cov in report.per_source} == set(SignalSource)
    assert report.empty_sources == tuple(SignalSource)
    for cov in report.per_source:
        assert cov.real_count == 0
        assert cov.synthetic_count == 0
        assert cov.last_real_at is None


def test_per_source_ordering_follows_enum(make_signal: SignalFactory) -> None:
    report = coverage_report([make_signal(provenance=Provenance.REAL)])
    assert tuple(cov.source_type for cov in report.per_source) == tuple(SignalSource)


def test_counts_real_and_synthetic_per_source(make_signal: SignalFactory) -> None:
    signals = [
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL),
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL),
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.SYNTHETIC),
        make_signal(source_type=SignalSource.CONTRACT, provenance=Provenance.SYNTHETIC),
    ]

    report = coverage_report(signals)
    by_source = _by_source(report)

    assert report.total_real == 2
    assert report.total_synthetic == 2
    statement = by_source[SignalSource.STATEMENT]
    assert statement.real_count == 2
    assert statement.synthetic_count == 1
    contract = by_source[SignalSource.CONTRACT]
    assert contract.real_count == 0
    assert contract.synthetic_count == 1


def test_last_real_at_is_the_freshest_real_signal(
    clock: FixedClock, make_signal: SignalFactory
) -> None:
    older = make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL)
    newer = make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL)
    # Make `newer` genuinely later than `older` despite the fixed clock.
    object.__setattr__(newer, "created_at", clock.now() + timedelta(days=2))

    report = coverage_report([older, newer])
    statement = _by_source(report)[SignalSource.STATEMENT]

    assert statement.last_real_at == newer.created_at


def test_synthetic_only_source_has_no_last_real_at_and_is_a_hole(
    make_signal: SignalFactory,
) -> None:
    # A source with only synthetic data is still a hole: no real => last_real_at None.
    report = coverage_report(
        [make_signal(source_type=SignalSource.DISCLOSED_TRADE, provenance=Provenance.SYNTHETIC)]
    )
    disclosed = _by_source(report)[SignalSource.DISCLOSED_TRADE]

    assert disclosed.real_count == 0
    assert disclosed.synthetic_count == 1
    assert disclosed.last_real_at is None
    assert SignalSource.DISCLOSED_TRADE in report.empty_sources


def test_all_synthetic_makes_every_source_a_hole(make_signal: SignalFactory) -> None:
    signals = [
        make_signal(source_type=source, provenance=Provenance.SYNTHETIC) for source in SignalSource
    ]

    report = coverage_report(signals)

    assert report.total_real == 0
    assert report.total_synthetic == len(tuple(SignalSource))
    assert report.empty_sources == tuple(SignalSource)


def test_mixed_holes_are_only_the_sources_without_real(make_signal: SignalFactory) -> None:
    # STATEMENT has real; CONTRACT only synthetic; DISCLOSED_TRADE absent entirely.
    signals = [
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL),
        make_signal(source_type=SignalSource.CONTRACT, provenance=Provenance.SYNTHETIC),
    ]

    report = coverage_report(signals)

    assert SignalSource.STATEMENT not in report.empty_sources
    assert SignalSource.CONTRACT in report.empty_sources
    assert SignalSource.DISCLOSED_TRADE in report.empty_sources


def test_backfilled_is_counted_separately_from_synthetic(make_signal: SignalFactory) -> None:
    # task-08: BACKFILLED must NEVER be absorbed into total_synthetic — it is real,
    # historically reconstructed data that must survive a purge, unlike SYNTHETIC.
    signals = [
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.REAL),
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.BACKFILLED),
        make_signal(source_type=SignalSource.STATEMENT, provenance=Provenance.BACKFILLED),
        make_signal(source_type=SignalSource.CONTRACT, provenance=Provenance.SYNTHETIC),
    ]

    report = coverage_report(signals)
    by_source = _by_source(report)

    assert report.total_real == 1
    assert report.total_backfilled == 2
    assert report.total_synthetic == 1
    statement = by_source[SignalSource.STATEMENT]
    assert statement.real_count == 1
    assert statement.backfilled_count == 2
    assert statement.synthetic_count == 0


def test_backfilled_only_source_has_no_last_real_at_and_is_a_hole(
    make_signal: SignalFactory,
) -> None:
    # A source with only backfilled data is still a "hole" for REAL purposes (no live data yet),
    # even though the backfilled data itself is real and must not be purged.
    report = coverage_report(
        [make_signal(source_type=SignalSource.DISCLOSED_TRADE, provenance=Provenance.BACKFILLED)]
    )
    disclosed = _by_source(report)[SignalSource.DISCLOSED_TRADE]

    assert disclosed.real_count == 0
    assert disclosed.backfilled_count == 1
    assert disclosed.synthetic_count == 0
    assert disclosed.last_real_at is None
    assert SignalSource.DISCLOSED_TRADE in report.empty_sources
