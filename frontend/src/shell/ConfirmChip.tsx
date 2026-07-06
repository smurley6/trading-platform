// §7 confirmation convention: destructive/committing actions (approve, cancel, kill switch,
// cycle mode) take one explicit confirm — keyboard Enter in a focused confirm chip, never a
// browser confirm(). Used by StatusBar's kill switch; reusable for Phase B/C committing actions.

import { useEffect, useRef } from "react";
import { colors, shape, type as typeTokens, withAlpha } from "../lib/tokens";

export function ConfirmChip({
  label,
  onConfirm,
  onCancel,
  tone = "loss",
}: {
  label: string;
  onConfirm: () => void;
  onCancel: () => void;
  /** --loss for destructive commits (kill switch, cancel, reject — the default); --accent for
   *  affirmative commits (approve — §3.1: accent marks the interactive/primary action). */
  tone?: "loss" | "accent";
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const toneColor = tone === "accent" ? colors.accent : colors.loss;

  useEffect(() => {
    ref.current?.focus();
  }, []);

  return (
    <button
      ref={ref}
      type="button"
      className="tp-shell"
      onClick={onConfirm}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          e.stopPropagation();
          onConfirm();
        } else if (e.key === "Escape") {
          e.preventDefault();
          e.stopPropagation();
          onCancel();
        }
      }}
      onBlur={onCancel}
      style={{
        cursor: "pointer",
        border: `1px solid ${toneColor}`,
        borderRadius: shape.radiusButton,
        background: withAlpha(toneColor, 0.12),
        color: toneColor,
        fontFamily: "var(--font-sans)",
        fontSize: typeTokens.scale.meta,
        fontWeight: 600,
        padding: "2px 8px",
        transition: "background-color var(--duration-overlay) var(--ease-out)",
      }}
    >
      {label} — Enter to confirm
    </button>
  );
}
