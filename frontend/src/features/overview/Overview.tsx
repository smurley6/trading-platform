// Overview `/` (docs/UI-DESIGN.md §5.1) — the rebuilt glance-target screen replacing the old
// Dashboard content. Three-band layout, no scrolling at 1440×900: capital strip / equity curve +
// positions grid / sleeves grid + event ticker. Algorithms is folded in per §2 (sleeves grid
// section) — the Algorithms screen/route itself is NOT removed this phase (see the Phase B
// report for the deferral rationale).
//
// Built ENTIRELY on the §3 token system (colors/layout/shape/type from lib/tokens.ts) — this is
// new-code, so unlike the legacy screens it does not touch the old `color`/`space`/`text` exports.
//
// Owns: the live event stream for the ticker (its own useEventStream subscription, since the
// ticker needs every individual event, not just a refresh signal — App.tsx's refreshKey debounce
// is for the rest of the shell); the Chain Inspector / Ticker Pane overlay state for this screen;
// and the `c`/`t` scoped hotkeys operating on "the currently focused row" (event ticker rows,
// including via SleevesGrid/PositionsGrid symbol click for the Ticker Pane). Both overlays are
// screen-scoped per the Phase B ambiguity note in the final report (§4's global "anywhere" is
// Overview-only until Review/Blotter/Signals are rebuilt in Phase C).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { api } from "../../lib/api";
import { dayPnl, maxDrawdownPct, poolEquitySeries } from "../../lib/projections";
import { colors, layout, shape, type as typeTokens } from "../../lib/tokens";
import type { Sleeve, StoredEventDto } from "../../lib/types";
import { useEventStream } from "../../lib/useEventStream";
import { useLoader } from "../../lib/useLoader";
import { ChainInspector } from "../../shell/ChainInspector";
import { TickerPane } from "../../shell/TickerPane";
import { SlideOverStackProvider } from "../../shell/SlideOverStackProvider";
import { useTickerQueryParam } from "../../shell/useHistoryRoute";
import { CapitalStrip } from "./CapitalStrip";
import { EventTicker } from "./EventTicker";
import { OverviewEquityCurve } from "./OverviewEquityCurve";
import { PositionsGrid } from "./PositionsGrid";
import { livePositionToRow, type PositionRowView } from "./positionRows";
import { positionRows as eventPositionRows } from "../../lib/projections";
import { SleevesGrid, type SleeveRowView } from "./SleevesGrid";

interface OverviewProps {
  refreshKey: number;
  sleeves: Sleeve[];
}

const MAX_TICKER_EVENTS = 500;

export function Overview({ refreshKey, sleeves }: OverviewProps) {
  const eventsLoader = useLoader(() => api.events(1000), [refreshKey]);
  const capital = usePolled(() => api.capital(), refreshKey, {
    enabled: false,
    cash: "0",
    equity: "0",
    buying_power: "0",
    allocations: [],
  });
  const livePositions = usePolled(() => api.livePositions(), refreshKey, {
    enabled: false,
    positions: [],
  });

  // Live rolling tail: this screen's own /ws/events subscription (on top of App.tsx's
  // refreshKey-debounced one) so the ticker gets every individual event to insert-animate,
  // capped at MAX_TICKER_EVENTS so the tail doesn't grow unbounded across a long session.
  const [liveEvents, setLiveEvents] = useState<StoredEventDto[]>([]);
  const onEvent = useCallback((event: StoredEventDto) => {
    setLiveEvents((prev) => {
      if (prev.some((e) => e.event_id === event.event_id)) return prev;
      const next = [...prev, event];
      return next.length > MAX_TICKER_EVENTS ? next.slice(next.length - MAX_TICKER_EVENTS) : next;
    });
  }, []);
  useEventStream(onEvent);

  // Seed the ticker with the initial page of events once, then let the live stream take over —
  // avoids both an empty ticker on first paint and double-counting events already in that page.
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current || !eventsLoader.data) return;
    seeded.current = true;
    setLiveEvents((prev) => {
      const seen = new Set(prev.map((e) => e.event_id));
      const seedRows = eventsLoader.data!.filter((e) => !seen.has(e.event_id));
      return [...seedRows, ...prev].slice(-MAX_TICKER_EVENTS);
    });
  }, [eventsLoader.data]);

  const allEvents = useMemo(() => eventsLoader.data ?? [], [eventsLoader.data]);
  const equitySeriesData = useMemo(() => poolEquitySeries(allEvents), [allEvents]);
  const drawdownPct = useMemo(() => maxDrawdownPct(equitySeriesData), [equitySeriesData]);
  const dayPnlValue = useMemo(() => dayPnl(equitySeriesData), [equitySeriesData]);

  const positionRows: PositionRowView[] = livePositions.enabled
    ? livePositions.positions.map(livePositionToRow)
    : eventPositionRows(allEvents).map((r) => ({
        key: `${r.sleeve_id}|${r.symbol}`,
        sleeve_id: r.sleeve_id,
        symbol: r.symbol,
        quantity: r.quantity,
        avg_price: null,
        last: null,
        market_value: null,
        unrealized_pnl: null,
      }));

  const capBySleeve = new Map(capital.allocations.map((a) => [a.sleeve_id, a] as const));
  const sleeveRows: SleeveRowView[] = sleeves.map((s) => {
    const alloc = capBySleeve.get(s.id);
    return {
      id: s.id,
      strategy_id: s.strategy_id,
      mode: s.mode,
      status: s.status,
      held_value: alloc ? Number(alloc.market_value) : Number(s.allocated_capital),
    };
  });

  // --- Overlay state: Chain Inspector + Ticker Pane, both screen-scoped for now (see docblock).
  const [correlationId, setCorrelationId] = useState<string | null>(null);
  const [ticker, setTicker] = useTickerQueryParam();
  const [focusedEvent, setFocusedEvent] = useState<StoredEventDto | null>(null);
  const lastRowRef = useRef<HTMLElement | null>(null);

  const openChainFor = useCallback((event: StoredEventDto, rowEl?: HTMLElement) => {
    if (rowEl) lastRowRef.current = rowEl;
    setCorrelationId(event.correlation_id);
  }, []);

  const openTickerFor = useCallback(
    (symbol: string, rowEl?: HTMLElement) => {
      if (rowEl) lastRowRef.current = rowEl;
      setTicker(symbol);
    },
    [setTicker],
  );

  // `c` / `t` on the currently focused ticker row (§4/§6: "Invoked with `c` anywhere a row has a
  // correlation id" / "click or `t` on focused row"). Each key is disabled only while ITS OWN
  // overlay is already open (so it doesn't re-fire into that overlay's own content) — the two
  // stay independently available so `t` still opens the Ticker Pane on top of an open Chain
  // Inspector and vice versa, which is exactly how §3.5's stacking behavior gets exercised from
  // the keyboard (open one, then invoke the other on the same focused row).
  useHotkeys(
    "c",
    () => {
      if (focusedEvent) openChainFor(focusedEvent);
    },
    { enabled: correlationId === null, enableOnFormTags: false },
    [focusedEvent, correlationId],
  );
  useHotkeys(
    "t",
    () => {
      const symbol = focusedEvent ? tickerFromEvent(focusedEvent) : null;
      if (symbol) openTickerFor(symbol);
    },
    { enabled: ticker === null, enableOnFormTags: false },
    [focusedEvent, ticker],
  );

  return (
    <SlideOverStackProvider>
      <div
        style={{
          display: "grid",
          gridTemplateRows: "auto 1fr 1fr",
          height: "100%",
          gap: layout.x3,
        }}
      >
        <div
          style={{
            padding: `${layout.x3}px ${layout.x2}px`,
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          <CapitalStrip
            enabled={capital.enabled}
            buyingPower={Number(capital.buying_power)}
            cash={Number(capital.cash)}
            equity={Number(capital.equity)}
            dayPnl={dayPnlValue}
            drawdownPct={drawdownPct}
          />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: layout.x3, minHeight: 0 }}>
          <Band title="Equity curve">
            <OverviewEquityCurve data={equitySeriesData} />
          </Band>
          <Band title="Positions">
            <PositionsGrid rows={positionRows} />
          </Band>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: layout.x3, minHeight: 0 }}>
          <Band title="Sleeves">
            <SleevesGrid rows={sleeveRows} />
          </Band>
          <Band title="Event ticker">
            <EventTicker
              events={liveEvents}
              focusedEventId={focusedEvent?.event_id ?? null}
              onFocusRow={setFocusedEvent}
              onSelect={(event, rowEl) => openChainFor(event, rowEl)}
            />
          </Band>
        </div>
      </div>

      <ChainInspector
        correlationId={correlationId}
        onClose={() => setCorrelationId(null)}
        returnFocusRef={lastRowRef}
      />
      <TickerPane ticker={ticker} onClose={() => setTicker(null)} returnFocusRef={lastRowRef} />
    </SlideOverStackProvider>
  );
}

function Band({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        border: `1px solid ${colors.border}`,
        borderRadius: shape.radiusPane,
        background: colors.bgSurface,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: `${layout.x1}px ${layout.x2}px`,
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.meta,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.04em",
          color: colors.textSecondary,
          borderBottom: `1px solid ${colors.border}`,
          flexShrink: 0,
        }}
      >
        {title}
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>{children}</div>
    </div>
  );
}

function tickerFromEvent(event: StoredEventDto): string | null {
  const p = event.payload;
  if (typeof p.symbol === "string") return p.symbol;
  if (typeof p.ticker === "string") return p.ticker;
  return null;
}

// Lightweight polling loader shared shape (mirrors Dashboard.tsx's useLivePositions/useCapital,
// generalized so this screen doesn't duplicate two near-identical effects).
function usePolled<T extends { enabled: boolean }>(
  load: () => Promise<T>,
  refreshKey: number,
  initial: T,
): T {
  const [state, setState] = useState<T>(initial);
  useEffect(() => {
    let cancelled = false;
    let pollTimer: number | undefined;
    const run = (): void => {
      void load().then((result) => {
        if (cancelled) return;
        setState(result);
        if (result.enabled && pollTimer === undefined) {
          pollTimer = window.setInterval(run, 5_000);
        } else if (!result.enabled && pollTimer !== undefined) {
          window.clearInterval(pollTimer);
          pollTimer = undefined;
        }
      });
    };
    run();
    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearInterval(pollTimer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `load` is a fresh closure each render by design
  }, [refreshKey]);
  return state;
}
