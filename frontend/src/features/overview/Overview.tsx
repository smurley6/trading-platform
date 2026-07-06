// Overview `/` (docs/UI-DESIGN.md §5.1, reduced per operator direction 2026-07-06) — the
// glance-target screen: capital strip (equity / day P&L / drawdown), a compact equity curve,
// and the positions grid. Nothing else: the sleeves grid and live event ticker were removed at
// the Phase C visual gate ("no reason to have that much information on the overview page") —
// sleeve health lives on System, the live feed and chain access live on Signals/Blotter/Review.
// No scrolling at 1440×900 (the host is viewport-locked in App.tsx).
//
// Built ENTIRELY on the §3 token system (colors/layout/shape/type from lib/tokens.ts) — this is
// new-code, so unlike the legacy screens it does not touch the old `color`/`space`/`text` exports.
//
// The Ticker Pane stays mounted for the `?ticker=` deep link (route-addressable per §2) via the
// shared focused-row model (shell/useScreenOverlays.ts); with the event ticker gone this screen
// has no focused-row source, so the `c`/`t` hotkeys are inert here (their getters return null).

import { useEffect, useMemo, useState } from "react";
import { api } from "../../lib/api";
import { dayPnl, maxDrawdownPct, poolEquitySeries } from "../../lib/projections";
import { colors, layout, shape, type as typeTokens } from "../../lib/tokens";
import { useLoader } from "../../lib/useLoader";
import { TickerPane } from "../../shell/TickerPane";
import { SlideOverStackProvider } from "../../shell/SlideOverStackProvider";
import { useScreenOverlays } from "../../shell/useScreenOverlays";
import { CapitalStrip } from "./CapitalStrip";
import { OverviewEquityCurve } from "./OverviewEquityCurve";
import { PositionsGrid } from "./PositionsGrid";
import { livePositionToRow, type PositionRowView } from "./positionRows";
import { positionRows as eventPositionRows } from "../../lib/projections";

interface OverviewProps {
  active: boolean;
  refreshKey: number;
}

// Compact curve band height (a structural dimension, not a §3 token value): tall enough to read
// the pool trend at a glance, small enough that positions own the rest of the viewport.
const EQUITY_BAND_PX = 240;

export function Overview({ active, refreshKey }: OverviewProps) {
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

  const allEvents = useMemo(() => eventsLoader.data ?? [], [eventsLoader.data]);
  const equitySeriesData = useMemo(() => poolEquitySeries(allEvents), [allEvents]);
  const drawdownPct = useMemo(() => maxDrawdownPct(equitySeriesData), [equitySeriesData]);
  const dayPnlValue = useMemo(() => dayPnl(equitySeriesData), [equitySeriesData]);

  // Prefer the endpoint's rows whenever it returns any — even with the Alpaca feed disabled
  // (`enabled: false`) the backend serves PaperBroker-derived positions with real avg/last/
  // market-value/P&L, which beats the event-projection fallback's null prices (a wall of
  // em-dashes; operator visual-gate feedback, 2026-07-06). The projection fallback remains for
  // when the endpoint itself is unreachable (api.ts catch → empty positions).
  const positionRows: PositionRowView[] =
    livePositions.positions.length > 0
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

  // Overlay state via the shared focused-row model: only the `?ticker=` deep-link binding is
  // live on this screen (no focused-row source remains — see docblock).
  const overlays = useScreenOverlays(active, {
    correlationId: () => null,
    ticker: () => null,
  });

  return (
    <SlideOverStackProvider>
      <div
        style={{
          display: "grid",
          gridTemplateRows: `auto ${EQUITY_BAND_PX}px minmax(0, 1fr)`,
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

        <Band title="Equity curve">
          <OverviewEquityCurve data={equitySeriesData} />
        </Band>

        <Band title="Positions">
          <PositionsGrid rows={positionRows} />
        </Band>
      </div>

      <TickerPane
        ticker={overlays.ticker}
        onClose={overlays.closeTicker}
        returnFocusRef={overlays.lastRowRef}
      />
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
