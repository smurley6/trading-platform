# Trading Platform

Personal automated multi-algorithm trading platform. Runs N strategies side by side,
each in an isolated capital **sleeve** with its own P&L, risk budget, and paper/live mode.
Paper-first; the switch to live trading is configuration + hardening, not a rewrite.

Architecture: **hexagonal (ports & adapters)** modular monolith. The domain core is pure
Python with no I/O; all infrastructure (broker, database, data providers) sits behind
ports and is wired in the composition root. An append-only **event store** is the system
of record — every signal, decision, risk verdict, order transition, fill, and user action
is an immutable event.

See `DESIGN.md` (the spec) for full requirements.

## Layout

```
backend/    Python 3.12 / FastAPI modular monolith (see backend/README.md)
frontend/   React 18 + TypeScript SPA (see frontend/README.md)
```

## Milestone status

- [x] **M1 — Skeleton & rails**: repo, CI, quality gates, config, domain model,
      event store (in-memory + Postgres), `BrokerPort` + `PaperBroker`, trivial
      strategy placing paper orders end-to-end with full event logging and reconciliation.
- [x] **M2 — Political signal pipeline**: ingestion adapters (news/RSS, USAspending,
      optional congressional trades), NLP (gazetteer ticker resolution + lexicon sentiment),
      committee↔sector influence weighting, corroboration/conviction, and a `political_signal`
      strategy. Two isolated sleeves (each its own broker), an ingest→window→evaluate cycle,
      and `/ingest`, `/conviction`, `/signals/feed` endpoints.
- [x] **M3 — UI**: React 18 + TypeScript (strict) SPA. Six screens (dashboard, algorithms,
      signal feed, blotter, risk) built on the mandated libraries — Salt (chrome), AG Grid
      (tables), TradingView Lightweight Charts + widgets (charts), FINOS Perspective
      (streaming conviction). Typed REST client + live WebSocket event stream. Backend gained
      a `/ws/events` broadcast endpoint. All gates green (tsc strict, eslint, vite build).
- [x] **M4 — Backtesting**: replay harness that runs recorded signals through the **same**
      strategy + risk + execution path as live (`EvaluateSignals` in a synthetic environment
      supplied via a factory port — no parallel logic). Produces equity curve, return,
      drawdown, and trade list. Exposed at `POST /backtest`; surfaced in the Algorithms
      screen with a Lightweight Charts equity curve, metrics, and an AG Grid trade list.
- [x] **M5 — Live hardening**: `AlpacaBroker` (live venue behind the existing `BrokerPort`;
      paper↔live is config-only via a mode-routing broker). Per-sleeve live opt-in gated by a
      `AssessLiveReadiness` preflight (master switch, live credentials, hardened risk caps, a
      functional kill-switch test, clean reconciliation) — a sleeve flips to live only when
      all checks pass and the operator confirms; mode changes are recorded as events. Secrets
      are env-only with separate live credentials. Unambiguous LIVE indicator in the UI.

## Quick start (backend)

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then edit risk caps if desired
python -m trading.main        # starts FastAPI on :8000 with in-memory event store
```

Quality gates (all must pass; CI blocks merges on them):

```bash
ruff format --check .   # formatting
ruff check .            # lint
mypy                    # strict typing
lint-imports            # hexagonal layering (dependencies point inward only)
pytest                  # unit + e2e tests
```

## Postgres / event store

Without `DATABASE_URL` the app runs on an in-memory event store (dev/tests).
With Postgres:

```bash
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/trading
cd backend && alembic upgrade head
```

The `events` table is append-only **by construction**: a database trigger rejects every
`UPDATE`/`DELETE`. Corrections are new compensating events.
