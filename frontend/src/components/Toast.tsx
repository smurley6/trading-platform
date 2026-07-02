// App-wide toast notifications — thin wrapper over `sonner` (docs/UI-DESIGN.md §10, replacing
// the hand-rolled ToastProvider/context this file used to implement). Every call site keeps
// using `useToast()` (success/error/info), so Proposals.tsx / RiskDashboard.tsx needed zero
// changes for this swap (Phase A ambiguity #5: a clean swap, so Toast.tsx IS replaced — sonner's
// <Toaster> is rendered once in main.tsx per §7: bottom-right, mono, 4s, failures persist until
// dismissed).
//
// No <ToastProvider> is needed any more (sonner's `toast()` is a module-level singleton that
// talks to the mounted <Toaster>), but the export is kept as a passthrough no-op so main.tsx's
// existing `<ToastProvider><App /></ToastProvider>` wrapping isn't required to change either.

import type { ReactNode } from "react";
import { toast as sonnerToast } from "sonner";

interface ToastApi {
  success: (message: string) => void;
  error: (message: string) => void;
  info: (message: string) => void;
}

const api: ToastApi = {
  success: (message) => sonnerToast.success(message),
  // §7: failures persist until dismissed (Infinity duration overrides the 4s default).
  error: (message) => sonnerToast.error(message, { duration: Infinity }),
  info: (message) => sonnerToast.info(message),
};

// eslint-disable-next-line react-refresh/only-export-components -- hook is colocated with its provider
export function useToast(): ToastApi {
  return api;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
