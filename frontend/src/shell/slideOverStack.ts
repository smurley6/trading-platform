// Shared stack context for slide-overs (docs/UI-DESIGN.md §3.5 "Overlay behavior": stacking).
// Split from SlideOver.tsx/SlideOverStackProvider.tsx (component-only files) so the context +
// `useSlideOverDepth` hook can live in one non-component module, per the project's react-refresh
// convention (see .eslintrc.cjs's react-refresh/only-export-components rule).

import { createContext, useContext, useEffect } from "react";

export interface StackContextValue {
  register: (id: string, open: boolean) => void;
}

export const StackContext = createContext<StackContextValue | null>(null);
export const StackOrderContext = createContext<string[]>([]);

/**
 * Registers `id` as open/closed in the shared stack (mount order = stack order) and returns its
 * depth (0 = topmost/frontmost, higher = further back) so the caller can pass it to
 * <SlideOver depth={...}>. Must be used within a <SlideOverStackProvider>.
 */
export function useSlideOverDepth(id: string, open: boolean): number {
  const ctx = useContext(StackContext);
  const order = useContext(StackOrderContext);

  useEffect(() => {
    ctx?.register(id, open);
  }, [ctx, id, open]);

  if (!ctx) {
    throw new Error("useSlideOverDepth must be used within a SlideOverStackProvider");
  }
  const idx = order.indexOf(id);
  if (idx === -1) return 0;
  // Frontmost = last registered (most recently opened) = highest index.
  return order.length - 1 - idx;
}
