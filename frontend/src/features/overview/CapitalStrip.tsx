// Capital strip (§5.1): "Buying power · Cash · Equity · Day P&L · Drawdown; 20px mono figures,
// 11px labels above — the only 'big numbers' in the app, earned because this is the glance
// target. Day P&L is the sole colored figure."

import { formatMoney, formatNumber } from "../../lib/format";
import { colors, layout, type as typeTokens } from "../../lib/tokens";
import { FeedWarningBanner } from "../../shell/states";

interface CapitalStripProps {
  enabled: boolean;
  buyingPower: number;
  cash: number;
  equity: number;
  dayPnl: number;
  drawdownPct: number;
}

export function CapitalStrip({
  enabled,
  buyingPower,
  cash,
  equity,
  dayPnl,
  drawdownPct,
}: CapitalStripProps) {
  if (!enabled) {
    return <FeedWarningBanner feed="Capital pool" configFlag="ENABLE_ALPACA_PAPER" />;
  }

  return (
    <div style={{ display: "flex", gap: layout.x8, alignItems: "baseline" }}>
      <Figure label="Buying power" value={formatMoney(buyingPower)} />
      <Figure label="Cash" value={formatMoney(cash)} />
      <Figure label="Equity" value={formatMoney(equity)} />
      <Figure
        label="Day P&L"
        value={`${dayPnl >= 0 ? "+" : ""}${formatMoney(dayPnl)}`}
        color={dayPnl >= 0 ? colors.gain : colors.loss}
      />
      <Figure label="Drawdown" value={`${formatNumber(drawdownPct, 2)}%`} />
    </div>
  );
}

function Figure({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: layout.x1 }}>
      <span
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: typeTokens.scale.meta,
          color: colors.textSecondary,
          textTransform: "uppercase",
          letterSpacing: "0.04em",
        }}
      >
        {label}
      </span>
      <span
        className="tp-numeric"
        style={{
          fontSize: typeTokens.scale.screenTitle,
          fontWeight: 600,
          color: color ?? colors.textPrimary,
        }}
      >
        {value}
      </span>
    </div>
  );
}
