"""Label-outcomes job CLI, driven against a fake labeler (never a real DB or network).

`build_dependencies` is monkeypatched to return a lightweight deps carrying a fake LabelOutcomes
and a `real_historical_prices` flag. Asserts the dry-run default reports without committing, that
`--confirm` requires the REAL historical adapter (refuses the synthetic provider), that a
confirmed run reports success, and that main() maps outcomes to exit codes.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from trading.application.outcome_labels import LabelReport
from trading.config import Settings
from trading.jobs import label_outcomes

_AS_OF = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
# Risk caps are required for Settings() to construct; a real (offline) Settings gives log_level.
_SETTINGS = Settings(
    _env_file=None,
    max_trade_pct_of_sleeve="10",
    max_position_concentration_pct="25",
    max_drawdown_pct="20",
)


class _FakeLabeler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.exclude_tickers_calls: list[frozenset[str] | None] = []

    async def execute(
        self,
        *,
        label_run_id: str,
        dry_run: bool,
        signals: object | None = None,
        exclude_tickers: frozenset[str] | None = None,
    ) -> LabelReport:
        self.calls.append((label_run_id, dry_run))
        self.exclude_tickers_calls.append(exclude_tickers)
        excluded = len(exclude_tickers) if exclude_tickers else 0
        return LabelReport(
            label_run_id=label_run_id,
            as_of=_AS_OF,
            signal_count=3,
            labels=9,
            ok=7,
            unresolvable=2,
            committed=not dry_run,
            excluded=excluded,
        )


@dataclass
class _FakeDeps:
    """The AppDeps surface the label-outcomes job reads."""

    label_outcomes: _FakeLabeler
    real_historical_prices: bool
    aclosers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def _patch(
    monkeypatch: pytest.MonkeyPatch, *, labeler: _FakeLabeler, real_prices: bool
) -> _FakeDeps:
    deps = _FakeDeps(label_outcomes=labeler, real_historical_prices=real_prices)
    monkeypatch.setattr(label_outcomes, "build_dependencies", lambda _settings: deps)
    monkeypatch.setattr(label_outcomes, "Settings", lambda: _SETTINGS)
    monkeypatch.setattr(label_outcomes, "configure_logging", lambda _level: None)
    return deps


async def test_dry_run_default_reports_without_committing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)  # synthetic prices are fine for dry-run

    rc = await label_outcomes.run_labeling(confirm=False)

    assert rc == 0
    assert labeler.calls and labeler.calls[0][1] is True  # dry_run=True
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "ok:            7" in out
    assert "unresolvable:  2" in out


async def test_confirm_requires_real_historical_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)  # synthetic provider

    rc = await label_outcomes.run_labeling(confirm=True)

    assert rc == 1
    assert labeler.calls == []  # never computed labels against fake bars
    assert "REAL historical-bars adapter" in capsys.readouterr().out


async def test_confirm_with_real_adapter_writes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=True)

    rc = await label_outcomes.run_labeling(confirm=True)

    assert rc == 0
    assert labeler.calls and labeler.calls[0][1] is False  # dry_run=False
    out = capsys.readouterr().out
    assert "CONFIRM" in out
    assert "stored 9 outcome label(s)" in out


def test_main_dry_run_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)
    assert label_outcomes.main([]) == 0


def test_main_confirm_without_real_adapter_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)
    assert label_outcomes.main(["--confirm"]) == 1


def test_main_unknown_flag_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # argparse exits 2 on a usage error; the entrypoint normalizes it to 1.
    assert label_outcomes.main(["--bogus"]) == 1


# --- --exclude-tickers (Task 04c: excludes the 44 leaked sec_form4 NAN signals) ------------------


def test_parse_exclude_tickers_absent_is_none() -> None:
    # Empty/absent flag = unchanged behavior: None, not an empty frozenset.
    assert label_outcomes._parse_exclude_tickers([]) is None


def test_parse_exclude_tickers_flattens_comma_joined_and_repeated_and_upper_normalizes() -> None:
    # Comma-joined in one occurrence, repeated flag, mixed case -> one upper-cased frozenset.
    result = label_outcomes._parse_exclude_tickers(["nan,xyz", "abc"])
    assert result == frozenset({"NAN", "XYZ", "ABC"})


def test_main_threads_exclude_tickers_to_labeler(monkeypatch: pytest.MonkeyPatch) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)
    assert label_outcomes.main(["--exclude-tickers", "nan"]) == 0
    assert labeler.exclude_tickers_calls == [frozenset({"NAN"})]


def test_main_without_exclude_tickers_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)
    assert label_outcomes.main([]) == 0
    assert labeler.exclude_tickers_calls == [None]


async def test_dry_run_report_shows_excluded_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labeler = _FakeLabeler()
    _patch(monkeypatch, labeler=labeler, real_prices=False)

    rc = await label_outcomes.run_labeling(confirm=False, exclude_tickers=frozenset({"NAN"}))

    assert rc == 0
    out = capsys.readouterr().out
    assert "exclude_tickers: NAN" in out
    assert "excluded:      1" in out
