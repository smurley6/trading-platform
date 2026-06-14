# Backend

Hexagonal (ports & adapters) modular monolith. Dependencies point **inward only** —
enforced in CI by import-linter:

```
interface  →  infrastructure  →  application  →  strategies  →  domain
```

- **domain/** — pure entities and rules (orders, positions, signals, conviction,
  politicians, sleeves, events). No I/O, no framework imports.
- **application/** — use cases (`EvaluateSignals`, `PlaceOrder`, `Reconcile`) and the
  **risk gate**, orchestrating the domain through **ports/** (abstract `Protocol`s).
- **strategies/** — plugins behind a registry (`trivial`, `political_signal`). Adding one
  = a new module + `@register`. Strategies return `StrategyProposal`s carrying conviction
  and the signals behind them, so reasoning flows into the audit log.
- **backtesting** (`application/backtest.py`) — replays recorded signals through the *same*
  `EvaluateSignals` path used live, in a throwaway environment (manual clock, in-memory
  store, historical-price market data, paper broker) supplied via the `BacktestEnvFactory`
  port. No parallel logic; output is equity curve, return, drawdown, and trade list.
- **infrastructure/** — adapters: `PaperBroker`, `AlpacaBroker` (live), `ModeRoutingBroker`
  (paper↔live), `StaticMarketData`/`HistoricalMarketData`, in-memory and Postgres event
  stores, `SystemClock`/`ManualClock`, NLP (`GazetteerEntityResolver`, `LexiconSentiment`),
  the data sources (`NewsSource`, `UsaSpendingSource`, `CongressTradesSource`), the static
  politician roster and sector map.
- **interface/** — FastAPI app and Pydantic DTOs (never exposes domain entities raw).
- **main.py** — the composition root: the only place concrete adapters are wired to ports.

## Key invariants

- **Event store is the system of record** (`domain/events.py`). Current state is a
  projection (`application/projections.py`) rebuildable from the log.
- **Append-only by construction**: the Postgres `events` table has a trigger
  (migration `0001`) that rejects every `UPDATE`/`DELETE`/`TRUNCATE`.
- **Write-ahead-of-action**: `PlaceOrder` persists the order intent *before* the broker
  call, so a crash leaves a recoverable record, not a gap.
- **Risk gate is non-bypassable**: `EvaluateSignals` routes every decision through
  `RiskGate.evaluate` before any order reaches a broker. A global kill switch halts all
  new orders.
- **Deterministic time**: everything reads `ClockPort`; tests use a `FixedClock`, no sleeps.

## Commands

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff format --check . && ruff check .   # formatting + lint
mypy                                    # strict typing (pydantic plugin enabled)
lint-imports                            # hexagonal layering contract
pytest --cov --cov-report=term-missing  # unit + e2e (integration tests skip w/o DB)

MAX_TRADE_PCT_OF_SLEEVE=10 MAX_POSITION_CONCENTRATION_PCT=25 MAX_DRAWDOWN_PCT=20 \
  python -m trading.main                # FastAPI on http://127.0.0.1:8000
```

Integration tests (`-m integration`) need `TEST_DATABASE_URL` pointing at a migrated
Postgres; they verify gap-free sequencing and that the DB itself forbids event mutation.

## The political pipeline (M2)

```
data sources → IngestSignals (records signal.received events, raw payload retained)
   → PoliticalCycle reads the corroboration window back from the event log
   → PoliticalSignalStrategy (per-ticker conviction: trade / watch / ignore)
   → risk gate → PaperBroker → fills → events
```

Each source is a `DataSourcePort` adapter taking an injected `httpx.AsyncClient`. NLP
(ticker resolution + sentiment) and the politician roster / sector map are swappable behind
their ports. Influence weight = `base_weight(role) × committee↔sector relevance ×
track_record`. Conviction rises when distinct source types corroborate the same ticker.
External sources default OFF; configure `NEWS_FEED_URLS` etc. to enable them.

## Try it

```bash
curl -s localhost:8000/health
# Manual push (defaults to the political sleeve): record + evaluate in one correlation id
curl -s -XPOST localhost:8000/signals \
  -H 'content-type: application/json' \
  -d '[{"source_type":"statement","ticker":"AAPL","sentiment":0.9,
        "raw_payload":{"headline":"Senator praises Apple"}}]'
curl -s localhost:8000/conviction        # current per-ticker conviction + action
curl -s localhost:8000/signals/feed      # recent signals (the political feed)
curl -s -XPOST localhost:8000/ingest     # run one ingest→evaluate cycle (if sources configured)
curl -s localhost:8000/events            # full audit chain for what just happened
curl -s -XPOST localhost:8000/reconcile  # per-sleeve diff of local projections vs broker
curl -s -XPOST localhost:8000/backtest   # replay recorded signals; equity curve + metrics
curl -s -XPOST localhost:8000/sleeves/sleeve-political/readiness  # live preflight checklist
curl -s -XPOST localhost:8000/sleeves/sleeve-political/mode \
  -H 'content-type: application/json' -d '{"mode":"live","confirm":true}'  # opt a sleeve into live
```

## Going live (M5, spec §11)

Live trading is off by construction. The paper→live switch is config-only at the broker
boundary: a per-sleeve `ModeRoutingBroker` sends orders to the in-process `PaperBroker` or to
`AlpacaBroker` (live) based on the sleeve's current mode, so strategy/risk/execution code
never changes. A sleeve can only opt into live after `AssessLiveReadiness` passes the §11
checklist — `ALLOW_LIVE` master switch, live Alpaca credentials present, configured risk caps
within the `LIVE_MAX_*` ceilings, a functional kill-switch test, and a clean reconciliation —
and the operator explicitly confirms. The readiness assessment and every mode change are
recorded as immutable events. Live credentials are separate from paper, env-only, and never
committed (`.env` is git-ignored).
