// App-wide toast notifications. Wrap the app in <ToastProvider> (see main.tsx) and call
// useToast() anywhere to surface success/error/info feedback for an action. Toasts auto-dismiss
// after a few seconds and can be clicked to dismiss early.

import { createContext, type ReactNode, useCallback, useContext, useMemo, useState } from "react";
import { color, radius, z } from "../lib/tokens";

type ToastKind = "success" | "error" | "info";

interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastApi {
  success: (message: string) => void;
  error: (message: string) => void;
  info: (message: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

// eslint-disable-next-line react-refresh/only-export-components -- hook is colocated with its provider
export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within <ToastProvider>");
  return ctx;
}

let nextId = 1;
const AUTO_DISMISS_MS = 5000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const remove = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (kind: ToastKind, message: string) => {
      const id = nextId++;
      setToasts((current) => [...current, { id, kind, message }]);
      window.setTimeout(() => remove(id), AUTO_DISMISS_MS);
    },
    [remove],
  );

  const apiValue = useMemo<ToastApi>(
    () => ({
      success: (m) => push("success", m),
      error: (m) => push("error", m),
      info: (m) => push("info", m),
    }),
    [push],
  );

  return (
    <ToastContext.Provider value={apiValue}>
      {children}
      <div
        style={{
          position: "fixed",
          bottom: 16,
          right: 16,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          zIndex: z.toast,
          maxWidth: "min(92vw, 420px)",
        }}
      >
        {toasts.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => remove(t.id)}
            style={{
              textAlign: "left",
              cursor: "pointer",
              padding: "10px 14px",
              borderRadius: radius.lg,
              background: color.panel,
              border: `1px solid ${toneBorder(t.kind)}`,
              borderLeft: `4px solid ${toneBorder(t.kind)}`,
              color: color.textDim,
              fontSize: 13,
              lineHeight: 1.4,
              boxShadow: "0 10px 30px rgba(0,0,0,0.45)",
            }}
          >
            <span style={{ color: toneBorder(t.kind), fontWeight: 700, marginRight: 8 }}>
              {toneIcon(t.kind)}
            </span>
            {t.message}
          </button>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

function toneBorder(kind: ToastKind): string {
  if (kind === "success") return color.ok;
  if (kind === "error") return color.dangerText;
  return color.accentText;
}

function toneIcon(kind: ToastKind): string {
  if (kind === "success") return "✓";
  if (kind === "error") return "✕";
  return "ℹ";
}
