// A small "i" (i-in-a-circle) button that reveals an explanatory panel in an anchored popover —
// use it next to any runnable action so the user can verify what it does BEFORE running it.
//
// 2026-07-06: rewritten on @radix-ui/react-popover (§10-approved, already installed — no new
// dep). The panel used to render IN-FLOW, expanding its parent flex row and pushing page content
// downward; it now renders in a Radix Portal, anchored to the trigger via floating-ui collision
// detection, so opening it causes ZERO layout shift and it can never run off the viewport (it
// flips side automatically near an edge). Follows the in-repo Radix pattern in
// shell/NavRail.tsx's Tooltip.Content. Public API (title/children/align) is unchanged so all 4
// call sites (Backtest.tsx, RiskDashboard.tsx ×3) need no changes.

import * as Popover from "@radix-ui/react-popover";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { colors, shape, type as typeTokens } from "../lib/tokens";

interface InfoButtonProps {
  title: string;
  children: ReactNode;
  /** Which edge the popover aligns to under the button; default "left". */
  align?: "left" | "right";
}

export function InfoButton({ title, children, align = "left" }: InfoButtonProps) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // The popover is portalled to <body>, so the shell's keep-alive display:none screen hiding
  // can't hide it — an open panel would float over whatever screen the operator navigates to.
  // While open, watch the trigger: the moment it stops intersecting the viewport (its screen was
  // hidden, or it scrolled fully out of view and the anchored panel would detach), close.
  useEffect(() => {
    if (!open) return;
    const el = triggerRef.current;
    if (!el) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => !entry.isIntersecting)) setOpen(false);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [open]);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          ref={triggerRef}
          type="button"
          aria-label={`What does "${title}" do?`}
          className="tp-info-trigger"
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 18,
            height: 18,
            borderRadius: "50%",
            padding: 0,
            cursor: "pointer",
            fontSize: typeTokens.scale.body,
            fontFamily: typeTokens.fontSans,
            fontWeight: 600,
            fontStyle: "normal",
            lineHeight: 1,
            background: "transparent",
          }}
        >
          i
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="bottom"
          align={align === "right" ? "end" : "start"}
          sideOffset={6}
          collisionPadding={12}
          aria-label={title}
          className="tp-shell tp-info-popover"
          style={{
            width: 360,
            maxWidth: "min(92vw, 520px)",
            padding: 12,
            background: colors.bgRaised,
            border: `1px solid ${colors.border}`,
            borderRadius: shape.radiusPane,
            boxShadow: shape.overlayShadow,
            color: colors.textSecondary,
            fontSize: typeTokens.scale.default,
            lineHeight: 1.5,
            textAlign: "left",
            zIndex: 1100,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8,
              gap: 8,
            }}
          >
            <strong style={{ color: colors.textPrimary, fontWeight: 600 }}>{title}</strong>
            <Popover.Close asChild>
              <button
                type="button"
                aria-label="Close"
                style={{
                  cursor: "pointer",
                  background: "transparent",
                  border: "none",
                  color: colors.textSecondary,
                  fontSize: 16,
                  lineHeight: 1,
                  padding: 0,
                }}
              >
                ×
              </button>
            </Popover.Close>
          </div>
          {children}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
