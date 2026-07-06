// PositionRowView + the live-position adapter — split from PositionsGrid.tsx (component-only file)
// so both can be imported without tripping react-refresh's only-export-components rule.

import type { LivePosition } from "../../lib/types";

export interface PositionRowView {
  key: string;
  sleeve_id: string;
  symbol: string;
  quantity: number;
  avg_price: number | null;
  last: number | null;
  market_value: number | null;
  unrealized_pnl: number | null;
}

export function livePositionToRow(p: LivePosition): PositionRowView {
  return {
    key: `${p.sleeve_id}|${p.symbol}`,
    sleeve_id: p.sleeve_id,
    symbol: p.symbol,
    quantity: Number(p.quantity),
    avg_price: Number(p.avg_price),
    last: Number(p.last),
    market_value: Number(p.market_value),
    unrealized_pnl: Number(p.unrealized_pnl),
  };
}
