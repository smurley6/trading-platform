# Frontend (M3)

React 18 + TypeScript (strict). Per the spec, no financial UI is hand-built; every surface
uses an established library:

| Surface                                    | Library                        | Where                           |
| ------------------------------------------ | ------------------------------ | ------------------------------- |
| Symbol charts (market data)                | TradingView embeddable widget  | `components/SymbolChart.tsx`    |
| Equity curves                              | TradingView Lightweight Charts | `components/EquityCurve.tsx`    |
| Blotter, positions, algo list, signal feed | AG Grid                        | `components/DataGrid.tsx`       |
| Streaming conviction feed                  | FINOS Perspective              | `components/StreamingTable.tsx` |
| Layout, chrome, forms, theming             | Salt design system             | throughout                      |

## Run

The backend must be running on :8000 (see `../backend`). The Vite dev server proxies
`/api/*` and `/ws/*` to it, so the SPA stays same-origin (no CORS).

```bash
npm install
npm run dev        # http://localhost:5173
```

Quality gates (all green):

```bash
npm run typecheck  # tsc --noEmit, strict
npm run lint       # eslint, --max-warnings 0 (no explicit any)
npm run build      # tsc + vite build (emits the Perspective WASM assets)
npm run format     # prettier
```

## Screens (FR-UI-6)

- **Dashboard** — aggregate + per-sleeve equity, paper/live indicator, kill switch.
- **Algorithms** — selector + per-algorithm detail (config, equity curve, trades, market
  chart) and a **backtest panel** (M4): runs `POST /backtest` and renders the equity curve
  (Lightweight Charts), metrics (return, drawdown, start/end equity, decisions), and the
  trade list (AG Grid).
- **Signal feed** — AG Grid of signals plus a live FINOS Perspective conviction view
  (ticker, score, action, corroboration count).
- **Blotter** — positions and order lifecycle, projected from the event log.
- **Risk** — risk verdicts, reconciliation drift, and manual reconcile/ingest controls.

## How it talks to the backend

- `lib/api.ts` — typed REST client (all calls through `/api`).
- `lib/useEventStream.ts` — subscribes to `/ws/events`; every appended backend event nudges
  a debounced refresh so all screens stay live without polling.
- `lib/types.ts` — DTOs mirroring the backend Pydantic schemas; a contract drift shows up
  as a TypeScript error.
- `lib/projections.ts` — client-side current-state views (blotter, positions, equity series)
  derived from the immutable event log, mirroring the backend's projection idea.

## Note on Perspective

Perspective v3 loads two WASM modules. `lib/perspective.ts` imports them with `?url` and
calls `init_server` / `init_client`, so the production build emits and serves the `.wasm`
assets (verified in `dist/`). The streaming view degrades gracefully (logs and shows the
fallback text) if the engine fails to start.
