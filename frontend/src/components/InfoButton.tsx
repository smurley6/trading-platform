// A small "ⓘ" (i-in-a-circle) button that reveals an explanatory panel on click — use it next
// to any runnable action so the user can verify what it does BEFORE running it.
//
// The panel renders IN-FLOW (it expands the page downward) rather than as a floating overlay, so
// it can never be clipped by a Card's bounds or run off the viewport — the whole thing is always
// scrollable into view. Closes on click-outside or Escape.

import { type ReactNode, useEffect, useRef, useState } from "react";

interface InfoButtonProps {
  title: string;
  children: ReactNode;
  /** Which edge the expanding panel aligns to under the button; default "left". */
  align?: "left" | "right";
}

export function InfoButton({ title, children, align = "left" }: InfoButtonProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent): void {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent): void {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div
      ref={ref}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: align === "right" ? "flex-end" : "flex-start",
        gap: 8,
      }}
    >
      <button
        type="button"
        aria-label={`What does "${title}" do?`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 18,
          height: 18,
          borderRadius: "50%",
          padding: 0,
          cursor: "pointer",
          fontSize: 12,
          fontStyle: "italic",
          fontFamily: "Georgia, 'Times New Roman', serif",
          fontWeight: 700,
          lineHeight: 1,
          color: open ? "#7cb3ff" : "#94a3b8",
          background: "transparent",
          border: `1px solid ${open ? "#4c9aff" : "rgba(148,163,184,0.55)"}`,
        }}
      >
        i
      </button>
      {open && (
        <div
          role="region"
          aria-label={title}
          style={{
            width: 360,
            maxWidth: "min(92vw, 520px)",
            padding: 14,
            borderRadius: 10,
            background: "#0f172a",
            border: "1px solid rgba(148,163,184,0.3)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.45)",
            fontSize: 13,
            lineHeight: 1.5,
            color: "#cbd5e1",
            textAlign: "left",
            cursor: "default",
            whiteSpace: "normal",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8,
            }}
          >
            <strong style={{ color: "#e2e8f0" }}>{title}</strong>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close"
              style={{
                cursor: "pointer",
                background: "transparent",
                border: "none",
                color: "#94a3b8",
                fontSize: 16,
                lineHeight: 1,
                padding: 0,
              }}
            >
              ×
            </button>
          </div>
          {children}
        </div>
      )}
    </div>
  );
}
