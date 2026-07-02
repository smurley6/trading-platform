// §7 confirmation convention: destructive/committing actions (approve, cancel, kill switch,
// cycle mode) take one explicit confirm — keyboard Enter in a focused confirm chip, never a
// browser confirm(). Used by StatusBar's kill switch; reusable for Phase B/C committing actions.

import { useEffect, useRef } from "react";
import { colors, shape, type as typeTokens, withAlpha } from "../lib/tokens";

export function ConfirmChip({
  label,
  onConfirm,
  onCancel,
}: {
  label: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);

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
          onConfirm();
        } else if (e.key === "Escape") {
          e.preventDefault();
          onCancel();
        }
      }}
      onBlur={onCancel}
      style={{
        cursor: "pointer",
        border: `1px solid ${colors.loss}`,
        borderRadius: shape.radiusButton,
        background: withAlpha(colors.loss, 0.12),
        color: colors.loss,
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
