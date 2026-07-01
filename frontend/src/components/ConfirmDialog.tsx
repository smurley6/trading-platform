// Reusable confirmation modal for actions that move money-like state (placing orders, going
// live). Renders a centered overlay; `tone="danger"` styles the confirm button for irreversible
// or real-money actions. Closes on backdrop click or Escape (unless busy).

import { Button } from "@salt-ds/core";
import { type ReactNode, useEffect } from "react";
import { color, radius, z } from "../lib/tokens";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  body: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: "default" | "danger";
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  tone = "default",
  busy = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent): void {
      if (e.key === "Escape" && !busy) onCancel();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, busy, onCancel]);

  if (!open) return null;

  return (
    <div
      role="presentation"
      onClick={() => !busy && onCancel()}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: z.modal,
        background: "rgba(2,6,23,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 440,
          maxWidth: "94vw",
          padding: 20,
          borderRadius: radius.lg,
          background: color.panel,
          border: `1px solid ${color.faintBorder}`,
          boxShadow: "0 20px 50px rgba(0,0,0,0.55)",
        }}
      >
        <strong style={{ display: "block", fontSize: 16, color: color.text, marginBottom: 10 }}>
          {title}
        </strong>
        <div style={{ fontSize: 14, lineHeight: 1.5, color: color.textDim }}>{body}</div>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 18 }}>
          <Button variant="secondary" disabled={busy} onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button
            variant="cta"
            disabled={busy}
            onClick={onConfirm}
            style={
              tone === "danger"
                ? { background: color.danger, borderColor: color.danger }
                : undefined
            }
          >
            {busy ? "Working…" : confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
