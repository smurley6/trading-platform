// 48px icon nav rail (§4): icons only, tooltips on hover, active state = --accent left rule.
// Exactly the five §4 destinations — Phase C completed the §2 "7 → 5" consolidation, so the
// temporary trailing Markets/Algorithms entries are gone (absorbed by the Ticker Pane and
// Overview's sleeves grid).

import * as Tooltip from "@radix-ui/react-tooltip";
import { colors, shape, type as typeTokens } from "../lib/tokens";
import type { ScreenId } from "./useHistoryRoute";

interface NavEntry {
  id: ScreenId;
  label: string;
  glyph: string;
}

// §4 ASCII diagram glyphs (◉ ▤ ⇄ ☰ ⚙) assigned to the five destinations in order.
const NAV_ENTRIES: NavEntry[] = [
  { id: "dashboard", label: "Overview", glyph: "◉" },
  { id: "proposals", label: "Review", glyph: "▤" },
  { id: "signals", label: "Signals", glyph: "⇄" },
  { id: "blotter", label: "Blotter", glyph: "☰" },
  { id: "risk", label: "System", glyph: "⚙" },
];

export function NavRail({
  active,
  onNavigate,
}: {
  active: ScreenId;
  onNavigate: (screen: ScreenId) => void;
}) {
  return (
    <Tooltip.Provider delayDuration={300}>
      <nav
        aria-label="Screens"
        className="tp-shell"
        style={{
          width: shape.navRailWidth,
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "stretch",
          gap: 2,
          padding: "8px 0",
          background: colors.bgSurface,
          borderRight: `1px solid ${colors.border}`,
        }}
      >
        {NAV_ENTRIES.map((entry) => (
          <Tooltip.Root key={entry.id}>
            <Tooltip.Trigger asChild>
              <button
                type="button"
                aria-label={entry.label}
                aria-current={active === entry.id ? "page" : undefined}
                data-active={active === entry.id}
                className="tp-nav-item"
                onClick={() => onNavigate(entry.id)}
                style={{
                  cursor: "pointer",
                  border: "none",
                  background: "transparent",
                  width: shape.navRailWidth,
                  height: shape.navRailWidth,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: typeTokens.scale.icon,
                  lineHeight: 1,
                  color: active === entry.id ? colors.accent : colors.textSecondary,
                }}
              >
                <span aria-hidden>{entry.glyph}</span>
              </button>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content
                side="right"
                sideOffset={6}
                className="tp-shell"
                style={{
                  background: colors.bgRaised,
                  border: `1px solid ${colors.border}`,
                  borderRadius: shape.radiusPane,
                  padding: "4px 8px",
                  fontSize: typeTokens.scale.body,
                  color: colors.textPrimary,
                  boxShadow: shape.overlayShadow,
                }}
              >
                {entry.label}
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        ))}
      </nav>
    </Tooltip.Provider>
  );
}
