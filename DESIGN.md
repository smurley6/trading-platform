# Automated Multi-Algorithm Trading Platform — Design & Requirements

**Version:** 1.0
**Status:** Ready for implementation
**Audience:** Implementing engineer
**Requirement keywords:** MUST / SHOULD / MAY follow RFC 2119.

---

## 1. Overview

A personal automated trading platform that runs multiple trading algorithms side by side, each with its own capital allocation, performance tracking, and independent paper/live mode. The system starts in paper trading and MUST be architected so the switch to real money is a configuration and hardening change, not a rewrite.

The first algorithm is the **Political Signal engine** — a deliberately gimmicky strategy that reacts to U.S. politicians' public statements, government contract awards, and (optionally) their disclosed trades, weighting politicians by role and influence. It is one plugin among many; the platform is general-purpose.

This document specifies the architecture, code structure, data model, functional requirements, and delivery plan. Quality and structural discipline are first-class requirements: the codebase MUST follow conventional layered/hexagonal design so that components can be changed or replaced without cascading failures.

---

## 2. Goals and non-goals

### 2.1 Goals
- Run N algorithms concurrently, each as an isolated capital "sleeve" with its own P&L and risk budget.
- Paper-first, with a clean, documented path to live trading via a single broker abstraction.
- Capture **all** trade, signal, decision, and interaction history in an immutable, append-only event log — no data loss.
- Use established, pre-built UI components for every surface; do not hand-build financial UI.
- Make data sources swappable behind adapters so cost and provider choices are not load-bearing.

### 2.2 Non-goals
- Not a high-frequency or low-latency system. Decision cadence is seconds-to-minutes, not microseconds.
- Not commercial software. Data-provider free tiers carry no commercial-use rights; respect them.
- Not financial advice and not a guarantee of returns. The political algorithm is exploratory.
- No microservices, message brokers, or distributed infrastructure in the initial build (see §4.2).

---

## 3. Glossary

- **Sleeve** — a per-algorithm sub-portfolio with its own allocated capital, positions, and P&L.
- **Signal** — a typed, scored observation emitted by a strategy's input pipeline (e.g., a politician statement about a ticker).
- **Conviction** — a numeric score (0–100) combining signal strength, source corroboration, and politician influence weight.
- **Decision** — a strategy's intended target position, before risk approval.
- **Order lifecycle** — intent → submitted → acknowledged → (partially) filled → final (filled/canceled/rejected).
- **PTR** — Periodic Transaction Report; STOCK Act congressional trade disclosure, lagged up to 45 days.
- **Event** — an immutable, timestamped, append-only record of something that happened in the system.

---

## 4. System architecture

### 4.1 Architectural style
The backend MUST follow **hexagonal architecture (ports and adapters)** with clean-architecture layering. The domain core MUST NOT depend on infrastructure (broker SDK, database, HTTP, data providers). Dependencies point inward only.

Layers (inner to outer):
1. **Domain** — entities, value objects, and domain rules (Signal, Decision, Order, Position, Politician, conviction logic). Pure Python, no I/O.
2. **Application** — use cases / services that orchestrate the domain (e.g., `EvaluateSignals`, `PlaceOrder`, `ReconcilePositions`). Depends only on domain + port interfaces.
3. **Ports** — abstract interfaces: `BrokerPort`, `DataSourcePort`, `EventStorePort`, `MarketDataPort`, `ClockPort`.
4. **Adapters (infrastructure)** — concrete implementations: `AlpacaBroker`, `PaperBroker`, `FmpDataSource`, `FinnhubDataSource`, `UsaSpendingDataSource`, `PostgresEventStore`, etc.
5. **Interface** — FastAPI HTTP/WebSocket layer and the React frontend.

### 4.2 Deployment shape
Build a **modular monolith**, not microservices. A single deployable backend process with clear internal module boundaries is the correct scale for this project and avoids premature distributed-systems complexity. The internal seams (ports) are where a future split would happen if ever needed.

### 4.3 Signal-to-execution flow
```
Data sources → Ingestion adapters → Signal pipeline (NLP, weighting, corroboration, conviction)
   → Strategy (per sleeve, emits target positions)
   → Risk gate (approve / modify / reject)
   → Execution (BrokerPort: paper or live)
   → Fills + Portfolio state
Every step writes immutable events to the Event Store.
Reconciliation periodically diffs local state against the broker's record of truth.
```

The **Risk gate** is a hard, non-bypassable boundary between strategy intent and execution. No order reaches a broker without passing it.

---

## 5. Technology stack

| Concern | Choice | Rationale |
|---|---|---|
| Backend language | Python 3.12+ | NLP, finance, and broker SDK ecosystem |
| Backend framework | FastAPI (async) | Typed, async, OpenAPI out of the box |
| Frontend | React 18 + TypeScript (strict) | Component ecosystem; matches UI library choices |
| Database | PostgreSQL 16 | System of record; WAL durability; JSONB for raw payloads |
| Time-series | TimescaleDB (Postgres extension) | Price snapshots and equity curves in the same DB |
| Migrations | Alembic | Versioned, reviewable schema changes |
| Broker | Alpaca (`alpaca-py`) | Same API for paper and live; multi-leg options; SIPC member |
| HTTP client | httpx (async) | Async data-source calls |
| Validation/config | Pydantic v2 + pydantic-settings | Typed models and env-based config |
| Task scheduling | APScheduler (in-process) | Polling, reconciliation jobs; no external broker needed |

UI component libraries (see §10.10): TradingView widgets, TradingView Lightweight Charts, AG Grid, FINOS Perspective, and the Salt design system. No bespoke financial UI.

---

## 6. Repository and code structure

A monorepo with two top-level apps. Backend organized by clean-architecture layer, not by technical type.

```
trading-platform/
├── backend/
│   ├── pyproject.toml            # deps, tool config (ruff, mypy, pytest)
│   ├── alembic/                  # migrations
│   ├── src/
│   │   └── trading/
│   │       ├── domain/           # pure entities + rules, no I/O
│   │       │   ├── signals.py
│   │       │   ├── orders.py
│   │       │   ├── positions.py
│   │       │   ├── politicians.py
│   │       │   └── conviction.py
│   │       ├── application/       # use cases / services
│   │       │   ├── evaluate_signals.py
│   │       │   ├── place_order.py
│   │       │   ├── reconcile.py
│   │       │   └── ports/         # abstract interfaces (Protocols/ABCs)
│   │       │       ├── broker.py
│   │       │       ├── data_source.py
│   │       │       ├── event_store.py
│   │       │       └── market_data.py
│   │       ├── strategies/        # plugin implementations
│   │       │   ├── base.py        # Strategy interface
│   │       │   └── political_signal/
│   │       ├── infrastructure/    # adapters
│   │       │   ├── brokers/        # alpaca.py, paper.py
│   │       │   ├── data_sources/   # fmp.py, finnhub.py, usaspending.py, news.py
│   │       │   ├── persistence/    # postgres event store, repositories
│   │       │   └── nlp/            # entity extraction, sentiment
│   │       ├── interface/          # FastAPI routers, websockets, schemas (DTOs)
│   │       ├── config.py           # pydantic-settings
│   │       └── main.py             # composition root / dependency wiring
│   └── tests/                      # unit / integration / e2e mirrors src layout
└── frontend/
    ├── package.json
    └── src/
        ├── components/             # wrappers around AG Grid, Perspective, TV widgets
        ├── features/               # dashboard, algorithms, signals, blotter, risk
        ├── lib/                    # api client, websocket client, types
        └── App.tsx
```

Rules:
- Dependency direction MUST be inward only. `domain` imports nothing from `application`, `infrastructure`, or `interface`. Enforce with import-linter or equivalent.
- All wiring of concrete adapters happens in the composition root (`main.py`), via dependency injection. Use cases receive ports, never concrete classes.
- One module = one responsibility. No "utils" dumping grounds.

---

## 7. Coding standards and quality gates

These exist specifically to prevent fragile code that fails later. CI MUST block merges on all of them.

### 7.1 Static quality
- **Formatting:** `ruff format` (backend), Prettier (frontend). Enforced, not optional.
- **Linting:** `ruff` (backend) with a strict ruleset; ESLint (frontend).
- **Typing:** `mypy --strict` (backend); TypeScript `strict: true`, no implicit `any`. Public functions MUST be fully type-annotated.
- **Pre-commit hooks** MUST run format, lint, and type checks before every commit.

### 7.2 Testing
- `pytest` with a clear test pyramid (see §14). Coverage gate SHOULD be ≥ 80% on `domain` and `application`.
- Domain and application layers MUST be testable with zero I/O (ports mocked/faked).

### 7.3 Runtime discipline
- **No silent failures.** Every caught exception is logged with context or re-raised. No bare `except:`.
- **Structured logging** (JSON) with a correlation/trace id propagated from signal → decision → order.
- **Configuration via environment**, validated at startup by pydantic-settings. No secrets or magic values in code.
- **Idempotency:** any operation with external side effects (order submission) MUST carry an idempotency key and be safe to retry.
- **Database access** through the repository pattern; all multi-write operations in transactions.
- **No `UPDATE`/`DELETE` on the events table** — ever (see §9).

### 7.4 CI pipeline (GitHub Actions or equivalent)
On every PR: install → format check → lint → type check → unit tests → integration tests (against Alpaca paper, gated on secrets) → build frontend. Red pipeline blocks merge.

---

## 8. Domain model and data schemas

Core entities (persisted as both current-state tables and as events; see §9):

- **Politician** — `id`, `name`, `role` (executive/committee_chair/senior/member), `committees[]`, `party`, `base_weight`, `track_record_score`.
- **Signal** — `id`, `created_at`, `source_type` (statement/contract/disclosed_trade), `ticker`, `politician_id`, `raw_payload` (JSONB — the original scraped/API record), `parsed` (JSONB), `sentiment`, `politician_weight`, `source_reliability`.
- **ConvictionResult** — `ticker`, `window`, `contributing_signal_ids[]`, `score` (0–100), `action` (trade/watch/ignore).
- **Decision** — `id`, `strategy_id`, `created_at`, `target_position`, `conviction_id`, `risk_verdict` (approved/modified/rejected), `risk_reason`, `final_quantity`.
- **Order** — `id`, `client_order_id` (idempotency key), `broker_order_id`, `decision_id`, `symbol`, `side`, `qty`, `type`, `status`, plus a child table of timestamped status transitions.
- **Fill** — `id`, `order_id`, `price`, `qty`, `fees`, `filled_at`.
- **PortfolioSnapshot** — `sleeve_id`, `taken_at`, `cash`, `equity`, `positions` (JSONB), `pnl`. Time-series (TimescaleDB).
- **MarketContext** — quote snapshot captured at decision time, referenced by `Decision`.
- **Sleeve** — `id`, `strategy_id`, `allocated_capital`, `mode` (paper/live), `status` (running/paused).

Every record carries a `schema_version`.

---

## 9. Event store and audit ("no data lost")

The append-only event log is the **system of record**. Current-state tables are projections derived from it and MAY be rebuilt from the log.

Requirements:
- A single `events` table: `event_id` (UUID), `seq` (monotonic, gap-free per stream), `stream_id`, `event_type`, `schema_version`, `occurred_at`, `recorded_at`, `payload` (JSONB), `correlation_id`.
- The table MUST be append-only: inserts only. Corrections are new compensating events. No updates, no deletes. Enforce via DB permissions/trigger.
- The system MUST capture, at minimum: every signal (including below-threshold "watch" signals, with raw payload), every decision and risk verdict, every order lifecycle transition, every fill, every portfolio snapshot, and every **user interaction** (algo enable/disable, capital reallocation, paper↔live toggle, kill switch, config edit, manual override).
- **Write-ahead-of-action:** order intent MUST be persisted durably before the broker call, so a crash leaves a recoverable record, not a gap.
- **Reconciliation:** a scheduled job MUST pull Alpaca's authoritative orders/positions and diff against local projections, emitting reconciliation events for any drift. The broker is the source of execution truth; local state is never assumed correct without reconciliation.
- Raw provider payloads MUST be retained alongside parsed results so signals can be re-derived if NLP logic changes.
- Point-in-time backups MUST be configured.

---

## 10. Functional requirements

### 10.1 Data ingestion (FR-ING)
- FR-ING-1: Each external source MUST be implemented as a `DataSourcePort` adapter; adding/removing a source MUST NOT affect downstream code.
- FR-ING-2: Provide adapters for: news/RSS (statements), USAspending.gov (contracts), and a congressional-trades source (FMP or Finnhub free/entry tier). The disclosed-trade adapter is OPTIONAL and disabled by default.
- FR-ING-3: Ingestion MUST normalize all inputs to the internal `Signal` schema and persist the raw payload.
- FR-ING-4: Respect provider rate limits and free-tier/no-commercial-use terms; make limits configurable.

### 10.2 Political Signal engine (FR-POL)
- FR-POL-1: Resolve mentioned companies to tickers (NER → ticker resolution) from statement text.
- FR-POL-2: Score statement sentiment/stance toward each resolved ticker.
- FR-POL-3: Compute a per-politician influence weight = `base_weight (by role)` × `topic_relevance (committee↔sector)` × `track_record_score`.
- FR-POL-4: Corroborate signals on the same ticker within a configurable rolling window across the three source types; conviction rises with alignment.
- FR-POL-5: Emit `ConvictionResult` with action by threshold: ≥ trade threshold → propose trade; ≥ watch threshold → watch; else ignore. Thresholds configurable.
- FR-POL-6: The statement and contract legs are real-time and primary; the disclosed-trade leg is optional corroboration only.

### 10.3 Strategy framework (FR-STR)
- FR-STR-1: Define a `Strategy` interface: consumes signals, emits target positions; pure and backtestable; no direct broker or DB access.
- FR-STR-2: Strategies are loaded via a registry; each is bound to one Sleeve with independent capital, mode, and status.
- FR-STR-3: Adding a new strategy MUST require no changes outside its own module and registration.

### 10.4 Risk management (FR-RISK)
- FR-RISK-1: A non-bypassable risk gate evaluates every Decision before execution.
- FR-RISK-2: Enforce: per-trade size cap, per-sleeve exposure cap, max position concentration, and max drawdown circuit breaker.
- FR-RISK-3: Provide a global **kill switch** that halts all new orders across all sleeves immediately.
- FR-RISK-4: Risk verdicts (approve/modify/reject + reason) MUST be recorded as events.

### 10.5 Execution and broker abstraction (FR-EXE)
- FR-EXE-1: All execution goes through `BrokerPort`. Provide `PaperBroker` and `AlpacaBroker` (paper and live differ only by credentials/endpoint).
- FR-EXE-2: Strategy and risk code MUST be broker-agnostic.
- FR-EXE-3: Track full order lifecycle with idempotent submission keyed by `client_order_id`.
- FR-EXE-4: Equities first. Options support MAY be added later via the same port without changing strategy code.

### 10.6 Reconciliation (FR-REC)
- FR-REC-1: Scheduled reconciliation of local orders/positions/cash against Alpaca; drift emits events and surfaces in the UI.

### 10.7 Portfolio and P&L (FR-PNL)
- FR-PNL-1: Maintain per-sleeve and aggregate positions, cash, equity, and realized/unrealized P&L.
- FR-PNL-2: Periodic snapshots stored as time-series for equity-curve rendering.

### 10.8 Backtesting (FR-BT)
- FR-BT-1: Replay recorded signals/market context through the same Strategy and Risk code paths used live (no separate logic).
- FR-BT-2: Produce equity curve, return, drawdown, and trade list.

### 10.9 API layer (FR-API)
- FR-API-1: REST for state/config; WebSocket for live signal feed, order updates, and P&L.
- FR-API-2: Request/response DTOs are explicit Pydantic schemas, decoupled from domain entities.

### 10.10 Frontend / UI (FR-UI)
No bespoke financial UI. Every surface MUST be built on an established component library:

- FR-UI-1: **Market-data views** (symbol charts, ticker tape, screeners, symbol info, news, economic calendar) → TradingView embeddable widgets (display-only; not a data source).
- FR-UI-2: **Own-data charts** (equity curves, backtest results, price with signal markers) → TradingView Lightweight Charts (or AG Charts financial). TradingView attribution MUST be displayed.
- FR-UI-3: **Tabular/app surfaces** (order blotter, positions, algorithm list, political signal feed) → AG Grid with custom cell renderers.
- FR-UI-4: **Streaming real-time views** (live conviction feed, streaming P&L) → FINOS Perspective.
- FR-UI-5: **Chrome / layout / forms / theming** → Salt design system.
- FR-UI-6: Required screens: portfolio dashboard (aggregate + per-sleeve, paper/live indicator, kill switch), algorithm selector, per-algorithm detail (config, equity curve, signals, trades, backtest), political signal feed (politician, weight, source corroboration, conviction, action), order blotter, risk dashboard.

---

## 11. Paper → Live upgrade path

The switch MUST be limited to:
1. Swap `BrokerPort` binding to `AlpacaBroker` with live credentials/endpoint (config only).
2. Harden the risk layer (tighter caps; mandatory kill-switch test).
3. Enable and verify reconciliation.
4. Move secrets to a managed store; rotate keys.
5. Confirm no strategy/domain code changes are required.

A live mode MUST require explicit, per-sleeve opt-in and display an unambiguous live indicator.

---

## 12. Non-functional requirements
- **Durability:** no acknowledged event is ever lost; recoverable across crashes.
- **Reliability:** a failure in one sleeve or data source MUST NOT take down others.
- **Auditability:** any past trade answers "what signals, what conviction, what risk verdict, what market context."
- **Replaceability:** brokers, data sources, and UI widgets are swappable behind their boundaries.
- **Performance:** decision latency in seconds is acceptable; no HFT constraints.

---

## 13. Security
- Secrets via environment / managed secret store only; never committed. `.env` git-ignored; provide `.env.example`.
- Broker keys least-privilege; separate paper and live credentials.
- Backend is single-user/local initially; if exposed, add auth before any live trading.

---

## 14. Testing strategy
- **Unit:** domain and application logic with faked ports. Conviction, weighting, risk rules, order lifecycle.
- **Integration:** adapters against real boundaries — Alpaca paper API, Postgres (test container), data-source sandboxes.
- **End-to-end:** signal ingest → decision → risk → paper order → fill → event log → reconciliation.
- **Backtest validation:** known historical inputs produce expected trade lists.
- Deterministic time via `ClockPort`; no real sleeps in tests.

---

## 15. Observability
- Structured JSON logs with correlation ids spanning the full signal→fill chain.
- Health endpoint; basic metrics (signals/min, orders, reconciliation drift, error counts).
- An in-app event-history view backed by the event store.

---

## 16. Configuration (defaults)
- `MODE_DEFAULT=paper`
- Conviction: `TRADE_THRESHOLD=70`, `WATCH_THRESHOLD=40`, `CORROBORATION_WINDOW_DAYS=21`
- Risk: `MAX_TRADE_PCT_OF_SLEEVE`, `MAX_POSITION_CONCENTRATION_PCT`, `MAX_DRAWDOWN_PCT`, all required at startup.
- Data sources: `DISCLOSED_TRADE_LEG_ENABLED=false`
- All thresholds and caps are environment-configurable; none hardcoded.

---

## 17. Phased delivery roadmap

- **M1 — Skeleton & rails.** Repo, CI, quality gates, config, domain model, event store, `BrokerPort` + `PaperBroker`, a trivial strategy placing paper orders end-to-end with full event logging and reconciliation.
- **M2 — Political signal pipeline.** Ingestion adapters (news, USAspending, optional trades), NLP + ticker resolution, influence weighting, corroboration, conviction, wired into the strategy framework.
- **M3 — UI.** Dashboard, algorithm selector, signal feed, blotter, risk panel using the prescribed component libraries.
- **M4 — Backtesting.** Replay harness over recorded signals; equity-curve and metrics output.
- **M5 — Live hardening.** Risk tightening, secrets management, kill-switch and reconciliation verification, live opt-in flow.

Each milestone MUST be green on all quality gates before the next begins.

---

## 18. Acceptance criteria (high level)
- A new algorithm can be added by creating one strategy module and registering it — nothing else changes.
- Switching a sleeve from paper to live requires only configuration + the §11 checklist.
- Every signal, decision, order, fill, and user action is retrievable from the event log; deleting/altering past events is impossible by construction.
- Reconciliation detects and reports any divergence from Alpaca.
- No financial UI is hand-built; all surfaces use the named libraries.
- CI enforces formatting, linting, strict typing, and tests on every change.

---

## 19. Open items
- Confirm news/statement source(s) and any free-tier limits before M2.
- Decide whether the optional disclosed-trade leg uses FMP or Finnhub if enabled.
- Confirm fixed per-sleeve capital allocation (assumed) vs. dynamic.

---

## Appendix A — External data sources

| Purpose | Source | Cost | Notes |
|---|---|---|---|
| Statements/media | News/RSS + scraping | Free | Core NLP input; build yourself |
| Government contracts | USAspending.gov API | Free | Clean, structured |
| Congressional trades (optional) | FMP or Finnhub | Free tier / ~$12/mo entry | Disabled by default; no commercial use |
| Execution + market data | Alpaca | Free (commission-free) | Same API paper/live |

## Appendix B — UI component mapping

| Surface | Library |
|---|---|
| Symbol charts, tickers, screeners, news, calendar | TradingView widgets |
| Equity curves, backtests, annotated price | Lightweight Charts / AG Charts |
| Blotter, positions, algo list, signal feed | AG Grid |
| Streaming conviction feed, live P&L | FINOS Perspective |
| Layout, forms, theming | Salt design system |
