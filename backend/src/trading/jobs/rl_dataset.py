"""Freeze one symbol's daily bars into the offline RL dataset file (task-rl-02).

Fetches daily closes through the composition root's historical-prices port, stamps every bar with
an explicit point-in-time ``known_at`` (16:00 America/New_York close + 15 min — Alpaca stamps a
daily bar at the session's OPENING boundary, so its timestamp is NOT knowledge time; see
``trading.application.rl_dataset``), and writes a deterministic JSON file every later RL milestone
reads instead of the network (docs/RL-SLEEVE-PROGRAM.md §4.2, §4.7). Torch-free: this module is a
source of the "Heavy ML deps never enter the app import graph" import-linter contract.

USAGE::

    python -m trading.jobs.rl_dataset --symbol SPY --years 8 --out data/rl/spy_daily.json
    python -m trading.jobs.rl_dataset --symbol SPY --years 8 --adjustment all \\
        --out data/rl/spy_daily.json --confirm

  * DRY-RUN (default): fetch + validate the bars and report the bar count, first/last session date,
    adjustment, source adapter and the sha256 the file WOULD have — WITHOUT writing anything.
    Always allowed; may run against whatever historical-prices provider is wired (the synthetic
    provider's daily grid is not session-stamped, so it fails the session-boundary check closed).
  * ``--confirm``: write the file (atomically, creating parent directories) and append ONE
    ``user.action`` audit event (``action="rl.dataset"``, with the file's sha256). Requires the
    REAL historical-bars adapter (ENABLE_HISTORICAL_DATA + resolvable Alpaca data credentials) —
    the job refuses to freeze bars from the synthetic price provider (same precedent as the
    labeler refusing to store synthetic-derived labels).
  * ``--adjustment raw|all`` (default ``raw``): Alpaca's corporate-action adjustment, ALWAYS sent
    explicitly and recorded in the file header — the mode is never implicit.
  * Run after 17:00 ET so the final bar is settled (16-min end-clamp + SIP delay); a session whose
    ``known_at`` has not passed is dropped, never frozen half-formed.
  * The ``user.action`` audit event is durable only when DATABASE_URL is configured; without it the
    composition root wires an in-memory event store and the event is discarded at exit (record the
    printed sha256 instead).

No HTTP endpoint is added (the API surface stays stable; documented in MEMORY.md). Composition
mirrors ``trading.jobs.backfill`` / ``trading.jobs.cycle``: build dependencies via the composition
root, run, and ALWAYS tear down async resources. Exit 0 clean / 1 on failure.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from trading.application.rl_dataset import (
    ALLOWED_ADJUSTMENTS,
    RlDataset,
    RlDatasetError,
    rl_dataset_to_json,
)
from trading.config import Settings
from trading.domain.events import DomainEvent, EventType
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging

logger = logging.getLogger(__name__)

_USER_ACTION_STREAM = "user-actions"
_DAYS_PER_YEAR = 365.25


def _emit(text: str) -> None:
    """Write a line of the operator-facing summary to stdout (this is a CLI command)."""
    print(text)  # noqa: T201 — summary output is the whole point of this command


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trading.jobs.rl_dataset",
        description="Freeze one symbol's daily bars (PIT known_at per bar) for offline RL.",
    )
    parser.add_argument("--symbol", default="SPY", help="ticker to freeze (default SPY)")
    parser.add_argument(
        "--years", type=int, default=8, help="lookback in years from now (default 8)"
    )
    parser.add_argument(
        "--adjustment",
        choices=sorted(ALLOWED_ADJUSTMENTS),
        default="raw",
        help="Alpaca corporate-action adjustment, sent explicitly and recorded (default raw)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output JSON path (written only with --confirm)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="write the file (default is a dry run); requires the REAL historical-bars adapter",
    )
    args = parser.parse_args(list(argv))
    if args.years <= 0:
        parser.error("--years must be positive")
    return args


def _lookback_days(years: int) -> int:
    return round(years * _DAYS_PER_YEAR)


def _format_summary(dataset: RlDataset, *, sha256: str) -> str:
    first, last = dataset.bars[0], dataset.bars[-1]
    lines = [
        f"  symbol:          {dataset.symbol}",
        f"  adjustment:      {dataset.adjustment}",
        f"  source_adapter:  {dataset.source_adapter}",
        f"  window:          {dataset.start.isoformat()} .. {dataset.end.isoformat()}",
        f"  bars:            {len(dataset.bars)}",
        f"  first session:   {first.session_ts.date().isoformat()}  "
        f"(known_at {first.known_at.isoformat()})",
        f"  last session:    {last.session_ts.date().isoformat()}  "
        f"(known_at {last.known_at.isoformat()})",
        f"  sha256:          {sha256}",
    ]
    return "\n".join(lines)


def _write_atomically(path: Path, data: bytes) -> None:
    """Write via a sibling temp file + ``os.replace`` so a crash never leaves a torn dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


async def run_dataset(
    *, confirm: bool, symbol: str, years: int, out: Path, adjustment: str = "raw"
) -> int:
    """Build dependencies, build the dataset, write it (``--confirm``) or dry-run, tear down.

    Guard: ``--confirm`` requires the REAL historical-bars adapter
    (``deps.real_historical_prices``). A dry run may use whatever provider is configured. Teardown
    always runs via ``finally``.
    """
    settings = Settings()
    configure_logging(settings.log_level)

    deps = build_dependencies(settings, historical_adjustment=adjustment)
    try:
        if deps.rl_dataset is None:  # defensive: the composition root always wires this
            raise RuntimeError("RL dataset builder is not configured")

        if confirm and not deps.real_historical_prices:
            _emit(
                "error: --confirm requires the REAL historical-bars adapter (set "
                "ENABLE_HISTORICAL_DATA=true with resolvable Alpaca data credentials). Refusing "
                "to freeze bars from the synthetic price provider — a dry run is always allowed."
            )
            return 1

        if deps.rl_dataset.adjustment != adjustment:  # wiring drift would make the header lie
            _emit(
                f"error: requested --adjustment {adjustment} but the wired builder records "
                f"{deps.rl_dataset.adjustment}; refusing to write an implicit adjustment mode."
            )
            return 1

        try:
            dataset = await deps.rl_dataset.execute(
                symbol=symbol, lookback_days=_lookback_days(years)
            )
        except RlDatasetError as exc:
            _emit(f"error: {exc}")
            return 1

        data = rl_dataset_to_json(dataset)
        digest = hashlib.sha256(data).hexdigest()

        _emit("=== RL dataset ===")
        _emit(f"mode: {'CONFIRM (writes file)' if confirm else 'DRY-RUN (no changes)'}")
        _emit(f"out:  {out}")
        _emit("")
        _emit(_format_summary(dataset, sha256=digest))

        if not confirm:
            _emit("")
            _emit("DRY-RUN: nothing was written. Re-run with --confirm to write the file.")
            return 0

        _write_atomically(out, data)
        await deps.event_store.append(
            DomainEvent(
                stream_id=_USER_ACTION_STREAM,
                event_type=EventType.USER_ACTION,
                payload={
                    "action": "rl.dataset",
                    "symbol": dataset.symbol,
                    "years": years,
                    "adjustment": dataset.adjustment,
                    "n_bars": len(dataset.bars),
                    "sha256": digest,
                },
                occurred_at=deps.clock.now(),
                correlation_id=str(uuid4()),
            )
        )
        _emit("")
        _emit(f"wrote {len(dataset.bars)} bar(s) to {out} (sha256 {digest}).")
        return 0
    finally:
        await _teardown(deps)


async def _teardown(deps: AppDeps) -> None:
    """Release every async resource the composition root registered (mirrors app shutdown).

    Each acloser is awaited independently so one failing close cannot strand the others (e.g. the
    Postgres pool is still disposed even if an HTTP client close raised).
    """
    for aclose in deps.aclosers:
        try:
            await aclose()
        except Exception:
            logger.exception("error closing a resource during job teardown")


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous entrypoint: parse args, run the freeze, return a process exit code.

    Returns 0 on success (including a clean dry run) and 1 on any failure — a disallowed
    --confirm, a dataset that fails PIT validation, or an unhandled exception (logged).
    """
    raw = sys.argv[1:] if argv is None else list(argv)
    try:
        args = _parse_args(raw)
    except SystemExit as exc:  # argparse exits 2 on a usage error; normalize to 1
        return 1 if exc.code not in (0, None) else 0
    try:
        return asyncio.run(
            run_dataset(
                confirm=bool(args.confirm),
                symbol=str(args.symbol),
                years=int(args.years),
                out=Path(args.out),
                adjustment=str(args.adjustment),
            )
        )
    except Exception:
        logger.exception("rl-dataset job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
