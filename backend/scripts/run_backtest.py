#!/usr/bin/env python
"""CLI to run a real-data backtest and print rich metrics (FR-BT-1/2).

A thin composition-root *script* (not library code): it is the one place allowed to reach
into ``trading.config`` and ``trading.infrastructure`` to wire a concrete
``HistoricalPricesPort``, build the strategy, run the backtest, and print a table. It lives
outside the ``trading`` package, so the hexagonal import-linter contract (rooted at
``trading``) does not apply to it.

Wiring mirrors ``trading.main.run_backtest``: REAL Alpaca daily bars are used when
``ENABLE_HISTORICAL_DATA`` is set and data credentials resolve; otherwise it falls back to
the deterministic synthetic price path so the script *always* runs with zero network calls.

Usage::

    python scripts/run_backtest.py --strategy political_signal \
        --symbols AAPL,MSFT,NVDA --days 180 --cash 100000

    # other knobs
    python scripts/run_backtest.py --symbols LMT --days 90 --seed 42 \
        --burst-interval-days 5 --window-days 21
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Allow running as a bare script (``python scripts/run_backtest.py``) by putting the package
# source on the path; a normal editable install makes this a no-op.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import httpx  # noqa: E402

from trading.application.backtest import Backtest, BacktestResult  # noqa: E402
from trading.application.backtest_seeding import (  # noqa: E402
    SeededScenario,
    build_seeded_request,
)
from trading.application.ports.historical import HistoricalPricesPort  # noqa: E402
from trading.application.risk import RiskLimits  # noqa: E402
from trading.config import Settings  # noqa: E402
from trading.domain.conviction import ConvictionThresholds  # noqa: E402
from trading.infrastructure.backtest_env import InMemoryBacktestEnvFactory  # noqa: E402
from trading.infrastructure.market_data.alpaca_history import AlpacaHistoricalData  # noqa: E402
from trading.infrastructure.market_data.historical import (  # noqa: E402
    SyntheticHistoricalPrices,
)

# Importing the strategies package runs each plugin's @register side effect, so the registry
# is populated before --strategy choices are read.
from trading.strategies import registry as strategy_registry  # noqa: E402
from trading.strategies.base import StrategyConfig  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_backtest",
        description="Run a real-data (or synthetic-fallback) backtest and print metrics.",
    )
    parser.add_argument(
        "--strategy",
        default="political_signal",
        choices=strategy_registry.available(),
        help="Registered strategy id to backtest (default: political_signal).",
    )
    parser.add_argument(
        "--symbols",
        default="AAPL,MSFT,NVDA",
        help="Comma-separated tickers (default: AAPL,MSFT,NVDA).",
    )
    parser.add_argument(
        "--days", type=int, default=180, help="Lookback window in days (default: 180)."
    )
    parser.add_argument(
        "--cash",
        type=Decimal,
        default=Decimal("100000"),
        help="Starting cash (default: 100000).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Corroboration window in days (default: settings.corroboration_window_days).",
    )
    parser.add_argument(
        "--seed", type=int, default=1729, help="Seed for the signal scenario (default: 1729)."
    )
    parser.add_argument(
        "--burst-interval-days",
        type=int,
        default=7,
        help="Days between seeded signal bursts (default: 7).",
    )
    return parser.parse_args(argv)


def _build_historical_prices(
    settings: Settings,
) -> tuple[HistoricalPricesPort, httpx.AsyncClient | None]:
    """Wire REAL Alpaca bars when enabled + creds resolve, else the synthetic fallback.

    Returns the port and (when real) the HTTP client to close afterward. Mirrors
    ``trading.main``'s gating so the script behaves exactly like the running service.
    """
    data_key_id = settings.data_key_id()
    data_secret_key = settings.data_secret_key()
    if settings.enable_historical_data and data_key_id and data_secret_key:
        client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
        port = AlpacaHistoricalData(
            client=client,
            key_id=data_key_id,
            secret_key=data_secret_key,
            base_url=settings.alpaca_data_base_url,
        )
        return port, client
    return SyntheticHistoricalPrices(), None


async def _run(args: argparse.Namespace) -> BacktestResult:
    settings = Settings()
    tickers = [s for s in args.symbols.split(",") if s.strip()]
    window_days = args.window_days or settings.corroboration_window_days
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=args.days)

    strategy = strategy_registry.create(
        args.strategy,
        StrategyConfig(
            thresholds=ConvictionThresholds(
                trade=settings.trade_threshold, watch=settings.watch_threshold
            ),
            window_days=window_days,
        ),
    )
    risk_limits = RiskLimits(
        max_trade_pct_of_sleeve=settings.max_trade_pct_of_sleeve,
        max_position_concentration_pct=settings.max_position_concentration_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
    )
    historical_prices, client = _build_historical_prices(settings)
    try:
        seeded = await build_seeded_request(
            historical_prices=historical_prices,
            strategy=strategy,
            tickers=tickers,
            start=start,
            end=end,
            risk_limits=risk_limits,
            starting_cash=args.cash,
            window_days=window_days,
            scenario=SeededScenario(seed=args.seed, burst_interval_days=args.burst_interval_days),
        )
        source = "REAL Alpaca bars" if client is not None else "synthetic fallback"
        _emit(
            f"Backtest: strategy={args.strategy} symbols={','.join(seeded.symbols)} "
            f"prices={source} signals={seeded.signal_count} "
            f"range={start.date()}..{end.date()}"
        )
        backtest = Backtest(env_factory=InMemoryBacktestEnvFactory())
        return await backtest.run(seeded.request)
    finally:
        if client is not None:
            await client.aclose()


def _fmt_money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _print_table(result: BacktestResult) -> None:
    rows: list[tuple[str, str]] = [
        ("Starting equity", _fmt_money(result.starting_equity)),
        ("Ending equity", _fmt_money(result.ending_equity)),
        ("Total return", _fmt_pct(result.total_return_pct)),
        ("Annualized (CAGR)", _fmt_pct(result.annualized_return_pct)),
        ("Max drawdown", f"{result.max_drawdown_pct:.2f}%"),
        ("Volatility (ann.)", f"{result.return_volatility:.4f}"),
        ("Sharpe ratio", f"{result.sharpe_ratio:.3f}"),
        ("Sortino ratio", f"{result.sortino_ratio:.3f}"),
        ("Buy & hold return", _fmt_pct(result.benchmark_total_return_pct)),
        (
            "vs. benchmark",
            _fmt_pct(result.total_return_pct - result.benchmark_total_return_pct),
        ),
        ("Decisions", str(result.decision_count)),
        ("Fills", str(len(result.trades))),
    ]
    stats = result.trade_stats
    if stats is not None:
        pf = "n/a" if stats.profit_factor is None else f"{stats.profit_factor:.2f}"
        rows.extend(
            [
                ("Round-trips", str(stats.round_trips)),
                ("Win rate", f"{stats.win_rate * 100:.1f}%"),
                ("Wins / losses", f"{stats.wins} / {stats.losses}"),
                ("Average win", _fmt_money(stats.average_win)),
                ("Average loss", _fmt_money(stats.average_loss)),
                ("Profit factor", pf),
                ("Largest win", _fmt_money(stats.largest_win)),
                ("Largest loss", _fmt_money(stats.largest_loss)),
            ]
        )
    width = max(len(label) for label, _ in rows)
    _emit("-" * (width + 22))
    for label, value in rows:
        _emit(f"{label.ljust(width)} : {value}")
    _emit("-" * (width + 22))


def _emit(line: str) -> None:
    # The single sanctioned output sink for this CLI (ruff T20 forbids bare print()).
    sys.stdout.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = asyncio.run(_run(args))
    _print_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
