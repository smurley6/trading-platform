# Deploy: FREE always-on trading cycle (GitHub Actions + managed Postgres)

This guide stands up the trading platform with **zero hosting cost**: a scheduled
[GitHub Actions](https://docs.github.com/actions) workflow runs the trading cycle every
20 minutes, and a free managed Postgres holds all state between runs. No server to keep
alive, no credit card for the always-on part.

The workflow lives at [`.github/workflows/cycle.yml`](../.github/workflows/cycle.yml). Each
run does: **install -> migrate the DB -> `python -m trading.jobs.cycle`** (one
ingest -> evaluate -> place-orders pass), then exits.

> [!IMPORTANT]
> **TimescaleDB prerequisite.** Migration `0002` creates a TimescaleDB hypertable
> (`CREATE EXTENSION timescaledb` + `create_hypertable(...)` in
> `backend/alembic/versions/0002_read_models_and_timescale.py`). **Stock Neon and Supabase
> Postgres do NOT ship the TimescaleDB extension, so `alembic upgrade head` will FAIL on
> them.** Pick one:
> - **Use a Postgres that bundles TimescaleDB** — the simplest is
>   [Timescale Cloud](https://www.timescale.com/cloud) (has a free trial) or a
>   [Tiger Cloud](https://www.tigerdata.com) free instance. Neon's
>   [`timescaledb` availability](https://neon.tech/docs/extensions/timescaledb) is limited to
>   the apache-licensed subset and may not include `create_hypertable`; verify before relying on it.
> - **Or** ask the backend owner to make migration `0002` degrade gracefully on plain Postgres
>   (skip the extension/hypertable when unavailable, keeping `portfolio_snapshots` a plain table).
>   That is application code and is out of scope for this deploy doc.
>
> The steps below work with **any** Postgres for the URL/secrets mechanics; just make sure the
> instance you choose can run migration `0002`.

---

## 1. Create a free managed Postgres

Using **Neon** (example; Supabase is analogous):

1. Sign up at [neon.tech](https://neon.tech) and create a project (the free tier is enough).
2. Open the project's **Connection Details** and copy the connection string. It looks like:

   ```
   postgresql://USER:PASSWORD@ep-xxxx-xxxx.us-east-2.aws.neon.tech/dbname?sslmode=require
   ```

3. **Convert it to the async form this app needs.** SQLAlchemy + asyncpg requires the
   `postgresql+asyncpg://` scheme, and **asyncpg does not understand the `?sslmode=...` query
   param** (that's a libpq/psycopg option). So:
   - Change the scheme `postgresql://` -> `postgresql+asyncpg://`.
   - **Remove** `?sslmode=require` (and any other libpq params) from the URL.
   - Neon requires TLS, but asyncpg negotiates SSL automatically against Neon's endpoint, so no
     extra param is needed. (If your provider needs an explicit toggle, asyncpg uses `?ssl=true`,
     not `sslmode`.)

   Result (this is the value for the `DATABASE_URL` secret):

   ```
   postgresql+asyncpg://USER:PASSWORD@ep-xxxx-xxxx.us-east-2.aws.neon.tech/dbname
   ```

   > Neon offers a "pooled" host (`...-pooler...`) and a "direct" host. Use the **direct** host
   > for `DATABASE_URL` — migrations and the async driver behave best against the direct endpoint.

The same URL is used by both the migration step (`alembic upgrade head` reads `DATABASE_URL`)
and the trading cycle.

---

## 2. Add the GitHub Actions repository secrets

In your repo: **Settings -> Secrets and variables -> Actions -> New repository secret**. Add a
secret for each row below. Optional rows have a default baked into the workflow — add a secret
only to override it.

> Names are EXACT (they map 1:1 to the app's `Settings`). A `.env`-style value is just the value;
> do not wrap in quotes in the GitHub UI.

### Database (required)

| Secret | Example | Required? |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://user:pass@ep-x.aws.neon.tech/neondb` | **Yes** |

### Risk caps (REQUIRED — the app will NOT start without all three)

These have **no defaults** in the app on purpose. If any is missing the process crashes at
startup (`pydantic` validation error) and the run fails red.

| Secret | Example | Required? |
| --- | --- | --- |
| `MAX_TRADE_PCT_OF_SLEEVE` | `10` | **Yes** |
| `MAX_POSITION_CONCENTRATION_PCT` | `25` | **Yes** |
| `MAX_DRAWDOWN_PCT` | `20` | **Yes** |

### Alpaca PAPER trading (required to actually place orders)

Get keys from the [Alpaca paper dashboard](https://app.alpaca.markets/paper/dashboard/overview)
(**Paper Trading -> API Keys -> Generate**). These are a real paper account — orders are real, the
money is not.

| Secret | Example | Required? |
| --- | --- | --- |
| `ALPACA_PAPER_KEY_ID` | `PKXXXXXXXXXXXXXXXXXX` | **Yes** (for paper trading) |
| `ALPACA_PAPER_SECRET_KEY` | `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` | **Yes** (for paper trading) |
| `ENABLE_ALPACA_PAPER` | `true` | Optional (defaults to `true` in the workflow) |
| `ALPACA_PAPER_BASE_URL` | `https://paper-api.alpaca.markets` | Optional (default shown) |
| `ALPACA_PAPER_SLEEVE_ID` | `sleeve-political` | Optional (default shown) |

> The Alpaca **paper key pair doubles as a data key**, so you usually do NOT need separate
> `ALPACA_DATA_*` secrets — the app falls back to the paper key for market data.

### Market data (optional — live quotes / live P&L marks)

| Secret | Example | Required? |
| --- | --- | --- |
| `ENABLE_LIVE_QUOTES` | `false` | Optional (default `false`) |
| `ENABLE_HISTORICAL_DATA` | `false` | Optional (default `false`) |
| `ALPACA_DATA_KEY_ID` | `PKXXXXXXXXXXXXXXXXXX` | Optional (falls back to paper/live key) |
| `ALPACA_DATA_SECRET_KEY` | `xxxxxxxx...` | Optional (falls back to paper/live key) |
| `ALPACA_DATA_BASE_URL` | `https://data.alpaca.markets` | Optional (default shown) |

### Data sources / feature flags (the political pipeline)

| Secret | Example | Required? |
| --- | --- | --- |
| `USASPENDING_ENABLED` | `true` | Optional (default `true`) — **FREE, no key, REAL signals** |
| `USASPENDING_LOOKBACK_DAYS` | `30` | Optional (default `30`) |
| `ENABLE_CONGRESS_ROSTER` | `true` | Optional (default `true`) — FREE roster so signals attach to REAL members |
| `NEWS_FEED_URLS` | `["https://host/politics.rss"]` | Optional (default `[]`) — see note below |
| `REQUIRE_REAL_DATA` | `true` | Optional (default `true`; auto-engages with a real Alpaca account anyway) |

> **`NEWS_FEED_URLS` is a JSON array string.** Paste the JSON literal as the secret value,
> e.g. `["https://www.example.gov/news.rss","https://other/feed.xml"]`. An empty list is `[]`
> (the default — news ingestion simply stays off). Do not add quotes around the whole thing in
> the GitHub UI; the value IS the JSON text.

### Conviction thresholds (optional — defaults match the spec)

| Secret | Example | Required? |
| --- | --- | --- |
| `TRADE_THRESHOLD` | `70` | Optional (default `70`) |
| `WATCH_THRESHOLD` | `40` | Optional (default `40`) |
| `CORROBORATION_WINDOW_DAYS` | `21` | Optional (default `21`) |

### Authentication (REQUIRED once a real broker is wired and the URL is reachable)

The mutating endpoints (`POST /proposals/execute`, `/kill-switch`, `/cycle/mode`,
`/sleeves/{id}/mode`, `/signals`, `/ingest`, `/reconcile`, `/orders/*`) and the sensitive reads
(`/capital`, `/events`, `/positions/live`, `/ws/events`) are guarded by `API_KEY`.

| Secret | Example | Required? |
| --- | --- | --- |
| `API_KEY` | a long random string | **Yes** once a real Alpaca account is wired and the app is internet-reachable |
| `ALLOW_UNAUTHENTICATED` | `true` | Optional (default `false`) — escape hatch only |

- **Unset (the default):** those endpoints are **OPEN**. Fine for the local in-memory dev app,
  but on a public URL it exposes the kill switch, the cycle-mode arm/disarm, and manual order
  execution to anyone. To fail closed, **the app REFUSES TO START when a real Alpaca broker
  (paper or live) is wired and `API_KEY` is unset** — set `API_KEY`, or (only for a deployment
  reachable solely on a private network) set `ALLOW_UNAUTHENTICATED=true` to accept the risk
  deliberately.
- **Set:** callers must present it via `Authorization: Bearer <key>` **or** the `X-API-Key`
  header; a mismatch is `401`. Genuinely public reads (`/sleeves`, `/conviction`, `/signals/feed`,
  `/proposals`, health) stay open. **Rotate** the key by generating a new value and updating the
  secret (the OWNER's action; the app picks it up on the next deploy).
- Store it **only** as a secret. Never commit a real key.

> **`MODE_DEFAULT` cannot be `live`.** Boot is always paper — setting `MODE_DEFAULT=live` fails
> config validation at startup (it would bypass the readiness gate). Enable live per sleeve via
> `POST /sleeves/{id}/mode` after `ALLOW_LIVE` and the readiness preflight pass.

> Any of the "optional, literal default" flags above (`ENABLE_ALPACA_PAPER`, `USASPENDING_ENABLED`,
> `ENABLE_CONGRESS_ROSTER`, etc.) can be left as the workflow's literal default **or** overridden
> by adding a same-named secret. The workflow reads `${{ secrets.NAME || 'default' }}`.

---

## 3. The schedule (and how to run it manually)

The workflow triggers two ways:

- **Cron:** `*/20 * * * *` — **every 20 minutes**, 24/7. Cron is in **UTC**. GitHub may **delay**
  scheduled runs under load (sometimes by minutes, occasionally skipping a tick at peak times),
  so treat 20 minutes as a best-effort cadence, not a hard SLA.
- **Manual:** GitHub repo -> **Actions** tab -> **Trading cycle** workflow (left sidebar) ->
  **Run workflow** button -> **Run workflow**. This dispatches an immediate one-off run — handy
  for the first smoke test and for forcing a cycle on demand.

A `concurrency` group ensures runs never overlap: if a cron tick arrives while a previous run is
still going, it queues rather than double-placing orders.

**Cost:** GitHub Actions is free for public repos, and private repos include a free monthly
Actions-minutes allowance. A 20-minute cron of a short (~1-2 min) job stays comfortably within
the free allowance for typical use. (If you ever hit the cap, widen the cron, e.g. `*/30`.)

---

## 4. What each run does

Each cycle (`python -m trading.jobs.cycle`) performs one pass, then exits:

1. **Ingest real signals** — pulls USASpending federal-contract awards (FREE, no key, REAL
   `CONTRACT` signals) and, if `NEWS_FEED_URLS` is set, news signals; with `ENABLE_CONGRESS_ROSTER`
   on, signals attach to REAL congressional members so they survive the anti-placeholder guard.
2. **Evaluate conviction** — scores signals (corroboration across sources/time) against the
   `TRADE_THRESHOLD` / `WATCH_THRESHOLD`.
3. **Place paper orders** — for anything clearing the trade threshold, submits orders to your
   Alpaca **paper** account (when `ENABLE_ALPACA_PAPER` is on).
4. **Sync fills + reconcile** — the real Alpaca venue fills **asynchronously** after submission
   (an order placed pre-open fills at the open, after this run has already exited), so the cron
   now runs `SyncOrders` + a pool `Reconcile` at the **end of every cycle** (review F14). Each run
   records the **prior** run's fills into the event log (cumulative-increment model, so partials
   that later complete are captured) and reconciles the projection against the one broker book;
   identical drift is recorded once, not re-appended every run. This is best-effort — a broker
   hiccup here is logged and never fails the ingest/evaluate cycle that already committed. No new
   secret is required. For the always-on **web service** the equivalent is the 30s poller
   (`ORDER_SYNC_INTERVAL_SECONDS>0`) and/or the real-time stream (`ENABLE_TRADE_UPDATES_STREAM`);
   the poller and stream are mutually safe — a fill is recorded exactly once (compare-and-set).

All state — the append-only event log and the read-model projections (positions, order statuses,
portfolio snapshots) — **persists in Postgres**, so each run builds on the last. The in-memory
store is NOT used here because `DATABASE_URL` is set.

---

## 5. Safety note — only REAL data can trade

The platform has an **anti-placeholder guard** (`REQUIRE_REAL_DATA`). The composition root turns
it **ON automatically** whenever a real Alpaca account is wired (`ENABLE_ALPACA_PAPER=true`) — you
cannot trade a real account on fakes. With it on:

- Synthetic / placeholder / seeded signals are **dropped at ingestion**.
- A real-broker evaluation **aborts** on any non-REAL signal.
- A real sleeve **won't boot** without real market data.

The flip side: this is a **real paper account**, and **real paper orders WILL be placed** the
moment a signal's conviction clears the trade threshold. It is not a dry run. Start with the
spec-default risk caps (`10` / `25` / `20`) and watch the first few runs before loosening them.
(Real *money* / live trading is a separate, off-by-default path — `ALLOW_LIVE` — not used here.)

---

## 6. Verify it worked

1. **Trigger a run** — Actions tab -> Trading cycle -> Run workflow (Section 3), or wait for the
   next cron tick.
2. **Check the run log** — open the run -> the **Run trading cycle** step. A clean run exits `0`
   and prints the cycle summary (signals ingested, evaluated, orders placed). A red run usually
   means a missing/bad secret:
   - Crash at startup mentioning `max_trade_pct_of_sleeve` / `max_position_concentration_pct` /
     `max_drawdown_pct` -> a **risk-cap secret is missing**.
   - Failure in the **Run database migrations** step mentioning `timescaledb` /
     `create_hypertable` -> your Postgres lacks TimescaleDB (see the IMPORTANT box at the top).
   - Connection / auth error -> recheck `DATABASE_URL` (async scheme, no `sslmode` param).
3. **Check Alpaca** — open the
   [Alpaca paper dashboard](https://app.alpaca.markets/paper/dashboard/overview): new orders /
   positions should appear after a run that cleared the trade threshold. (Off-hours, Alpaca queues
   market orders until the next session, so a placed order may show as accepted/pending rather than
   filled.)
4. **Spot-check Postgres** (optional) — your Neon/Supabase SQL console: `SELECT count(*) FROM events;`
   should grow across runs, and `rm_positions` / `rm_order_statuses` reflect current state.
