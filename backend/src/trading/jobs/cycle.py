"""One-shot political-cycle job: ``python -m trading.jobs.cycle``.

Runs exactly ONE end-to-end turn of the political-signal sleeve (ingest -> evaluate -> place
orders) and exits, so the full trading cycle can run on a schedule (e.g. a GitHub Actions cron)
instead of needing an always-on server. It is the headless counterpart to the FastAPI
``/ingest`` route and the in-app scheduled ingest job: it reuses the SAME composition root
(``trading.main.build_dependencies``) and the SAME :class:`PoliticalCycle`, so there is no
re-implementation of ingestion or evaluation here.

State lives in the configured event store. Point ``DATABASE_URL`` at Postgres and each
scheduled run sees the prior runs' events (corroboration window, cursors, positions); with no
``DATABASE_URL`` the run is in-memory and stateless (dev/tests).

Resource lifecycle mirrors the FastAPI shutdown (``trading.interface.app`` lifespan): the
job ALWAYS runs ``deps.aclosers`` on teardown, which closes the shared data-source HTTP client,
any Alpaca clients, and disposes the Postgres connection pool (the composition root appends
``AsyncEngine.dispose`` to the aclosers).

Exit codes: 0 on a clean run (including the no-sources no-op); non-zero on any unhandled
failure (the exception is logged before exit).
"""

import asyncio
import logging
import sys

from trading.config import Settings
from trading.interface.app import AppDeps
from trading.main import build_dependencies
from trading.observability import configure_logging

logger = logging.getLogger(__name__)


async def _teardown(deps: AppDeps) -> None:
    """Release every async resource the composition root registered (mirrors app shutdown).

    Each acloser is awaited independently so one failing close cannot strand the others (e.g.
    the Postgres pool is still disposed even if an HTTP client close raised).
    """
    for aclose in deps.aclosers:
        try:
            await aclose()
        except Exception:
            logger.exception("error closing a resource during job teardown")


async def run_cycle() -> None:
    """Build dependencies, run one political cycle, and tear down all async resources.

    No-ops cleanly (no error) when no data sources are configured: in that case the
    composition root leaves ``deps.political_cycle`` as ``None`` and there is nothing to run.
    Any exception from the cycle propagates to :func:`main`, which maps it to a non-zero exit;
    teardown still runs via the ``finally`` block.
    """
    settings = Settings()
    configure_logging(settings.log_level)
    deps = build_dependencies(settings)
    try:
        cycle = deps.political_cycle
        if cycle is None:
            logger.info("political cycle skipped: no data sources configured (nothing to ingest)")
            return
        result = await cycle.execute()
        # The structured summary goes in the message itself: the platform's JsonFormatter only
        # renders a fixed set of `extra` fields (correlation id + request metrics), so cycle
        # counts would be silently dropped if passed via `extra`. The correlation id IS rendered,
        # so the evaluation id is threaded through for log correlation.
        failed = list(result.ingestion.failed_sources)
        logger.info(
            "political cycle complete: signals_ingested=%d failed_sources=%s decisions=%d "
            "orders_placed=%d ingestion_correlation_id=%s",
            len(result.ingestion.signals),
            failed,
            len(result.evaluation.decisions),
            len(result.evaluation.orders),
            result.ingestion.correlation_id,
            extra={"correlation_id": result.evaluation.correlation_id},
        )
        # Close the async-fill loop from the cron itself (review F14): with the poller/stream off
        # in the documented cron deployment, no fill of any cron-placed order was ever recorded and
        # the same drift was re-appended every run. Order-sync ingests the PRIOR run's fills (this
        # run's orders fill at the open, after the pre-open cron exits) and reconcile then diffs the
        # updated projection against the one broker book, deduping unchanged drift. Best-effort: a
        # failure here must not fail the ingest/evaluate cycle that already committed.
        await _sync_and_reconcile(deps)
    finally:
        await _teardown(deps)


async def _sync_and_reconcile(deps: AppDeps) -> None:
    """Run every sleeve's order-sync then the pool reconcile, isolating any failure.

    Each step is independent so one broker hiccup cannot strand the others or fail the cron whose
    ingest/evaluate work is already durably recorded.
    """
    for sync in deps.order_syncs:
        try:
            sync_report = await sync.execute()
            logger.info(
                "cron order-sync: sleeve=%s fills=%d status_changes=%d checked=%d failed=%d",
                sync_report.sleeve_id,
                sync_report.fills_emitted,
                sync_report.status_changes_emitted,
                sync_report.checked_orders,
                sync_report.failed_orders,
            )
        except Exception:
            logger.exception("cron order-sync failed")
    for reconciler in deps.reconcilers:
        try:
            recon_report = await reconciler.execute()
            logger.info(
                "cron reconcile: sleeve=%s drift_count=%d checked_positions=%d checked_orders=%d",
                recon_report.sleeve_id,
                len(recon_report.drifts),
                recon_report.checked_positions,
                recon_report.checked_orders,
            )
        except Exception:
            logger.exception("cron reconcile failed")


def main() -> int:
    """Synchronous entrypoint: run one cycle and return a process exit code.

    Returns 0 on success and 1 on any unhandled failure (logged before returning). Wired to
    ``sys.exit`` under ``__main__`` so ``python -m trading.jobs.cycle`` exits with this code.
    """
    try:
        asyncio.run(run_cycle())
    except Exception:
        logger.exception("political cycle job failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
