# Always-on real-time data scraper (local, launchd)

Runs the backend 24/7 on your Mac so signals are scraped **in real time** instead of once a
day. The serverless GitHub Actions cron can't do this (5-minute floor, no persistent process,
no WebSocket); a long-running local process can.

## How it works

```
launchd  ──keeps alive──▶  uvicorn @ 127.0.0.1:8000   (python -m trading.main)
                                │
                                ├─ APScheduler ticks every INGEST_INTERVAL_SECONDS
                                │     └─ political cycle: poll every source → append NEW signals → Neon
                                │           (CYCLE_TRADE_MODE=manual → places ZERO orders)
                                │
                                └─ /ws/events ──push──▶ dashboard refreshes the instant a signal lands
```

Every piece already existed in the app — the scheduler ([`_scheduled_jobs`](../../backend/src/trading/interface/app.py)),
the live WebSocket fan-out (`/ws/events` via `BroadcastingEventStore`), and the frontend
subscriber (`useEventStream` in `App.tsx`). launchd just keeps the process alive so the
scheduler keeps ticking.

## Install / start

```bash
ops/realtime/install.sh
# or tune the cadence / mode:
INGEST_INTERVAL_SECONDS=60 CYCLE_TRADE_MODE=manual ops/realtime/install.sh
```

Starts immediately, restarts on crash, and relaunches at every login. Idempotent — re-run to
change settings.

## The real-time constraint (important)

The political sources are **not push feeds** — they publish on their own schedule, often with
legal delays:

| Source | Realistic freshness | Notes |
|---|---|---|
| **SEC Form 4** (EDGAR `getcurrent` Atom) | **minutes–hours** | Insider trades; ~2 business-day statutory lag. The fastest signal. |
| Federal Register (Public Inspection) | ~1 business day early | Best FDA / agency-action catalyst. |
| House PTRs | up to ~45 days | STOCK Act lag; Clerk republishes intraday. |
| USASpending | days–weeks | Contract awards report on lag. |
| Senate eFD | up to ~45 days | Endpoint fights automation; poll gently. |

So "real time" here = **poll frequently + push to the UI the instant a new signal is
ingested**. Polling faster lowers *detection* latency; it can't outrun when the source
actually publishes. Default poll cadence is 120s (kind to upstream rate limits; the data
doesn't change faster than that for most sources).

## Manual vs auto

`CYCLE_TRADE_MODE=manual` (default) → the loop **collects + classifies + builds proposals but
places no orders**. You approve trades in the Proposals UI. Set `CYCLE_TRADE_MODE=auto` only if
you want the always-on loop to place paper orders itself.

## View the dashboard

The scraper runs headless. To watch signals stream in:

```bash
cd frontend && npm run dev    # → http://localhost:5173 (Vite proxies /api + /ws to :8000)
```

The header shows a green **LIVE** pill and a **+N** badge each time new signals arrive.

## Operate

```bash
tail -f ~/Library/Logs/trading-realtime/stdout.log          # watch cycles ("political cycle complete: signals_ingested=…")
launchctl print gui/$(id -u)/com.stevenmurley.trading.realtime | grep -E 'state|pid'
ops/realtime/uninstall.sh                                   # stop + remove (keeps logs)
```

## Coexistence with the GitHub cron

Harmless to run both. The daily Actions cron stays as a backstop (runs even when your Mac is
off); the source cursors + event-store dedup mean a signal is never double-counted if both
the cron and the local loop see it.
