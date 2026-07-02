// cmdk command-palette SKELETON (§4, §10) — Ctrl+K opens/closes it; minimal shell only. The
// full command set (navigate, jump to ticker, run actions, search by correlation id — §2
// Additions) is a later phase; this wires the headless primitives (cmdk's own list/input plus
// @radix-ui/react-dialog for the overlay: focus trap, inert background, Esc handling) styled
// with §3 tokens, with just enough commands (the five screen destinations) to prove the shell.

import { Command } from "cmdk";
import * as Dialog from "@radix-ui/react-dialog";
import { colors, layout, shape, type as typeTokens } from "../lib/tokens";
import type { ScreenId } from "./useHistoryRoute";

const DESTINATIONS: { id: ScreenId; label: string }[] = [
  { id: "dashboard", label: "Overview" },
  { id: "proposals", label: "Review" },
  { id: "signals", label: "Signals" },
  { id: "blotter", label: "Blotter" },
  { id: "risk", label: "System" },
  { id: "markets", label: "Markets" },
  { id: "algorithms", label: "Algorithms" },
];

export function CommandPalette({
  open,
  onOpenChange,
  onNavigate,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (screen: ScreenId) => void;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="tp-palette-overlay" />
        <Dialog.Content className="tp-palette-content tp-shell" aria-describedby={undefined}>
          <Dialog.Title style={{ position: "absolute", width: 1, height: 1, overflow: "hidden" }}>
            Command palette
          </Dialog.Title>
          <Command
            style={{
              background: colors.bgRaised,
              borderRadius: shape.radiusPane,
              color: colors.textPrimary,
              overflow: "hidden",
            }}
          >
            <Command.Input
              autoFocus
              placeholder="Go to…"
              style={{
                width: "100%",
                boxSizing: "border-box",
                border: "none",
                borderBottom: `1px solid ${colors.border}`,
                background: "transparent",
                color: colors.textPrimary,
                fontFamily: "var(--font-sans)",
                fontSize: typeTokens.scale.default,
                padding: `${layout.x3}px ${layout.x3}px`,
                outline: "none",
              }}
            />
            <Command.List style={{ maxHeight: 320, overflowY: "auto", padding: layout.x1 }}>
              <Command.Empty
                style={{
                  padding: layout.x3,
                  fontFamily: "var(--font-sans)",
                  fontSize: typeTokens.scale.body,
                  color: colors.textMuted,
                }}
              >
                No matches.
              </Command.Empty>
              <Command.Group
                heading="Go to"
                style={{
                  fontFamily: "var(--font-sans)",
                }}
              >
                {DESTINATIONS.map((dest) => (
                  <Command.Item
                    key={dest.id}
                    value={dest.label}
                    onSelect={() => {
                      onNavigate(dest.id);
                      onOpenChange(false);
                    }}
                    style={{
                      cursor: "pointer",
                      padding: `${layout.x2}px ${layout.x3}px`,
                      borderRadius: shape.radiusButton,
                      fontFamily: "var(--font-sans)",
                      fontSize: typeTokens.scale.default,
                      color: colors.textPrimary,
                    }}
                  >
                    {dest.label}
                  </Command.Item>
                ))}
              </Command.Group>
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
