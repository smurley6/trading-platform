// SlideOverStackProvider — tracks which slide-overs are currently open (mount order = stack
// order) so each <SlideOver> can compute its own depth for the §3.5 push-back depth cue. Overview
// mounts one provider; Chain Inspector and Ticker Pane both register through useSlideOverDepth
// (shell/slideOverStack.ts).

import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import { StackContext, StackOrderContext, type StackContextValue } from "./slideOverStack";

export function SlideOverStackProvider({ children }: { children: ReactNode }) {
  const [order, setOrder] = useState<string[]>([]);

  const value = useMemo<StackContextValue>(
    () => ({
      register: (id, open) => {
        setOrder((prev) => {
          if (open) {
            return prev.includes(id) ? prev : [...prev, id];
          }
          return prev.filter((x) => x !== id);
        });
      },
    }),
    [],
  );

  return (
    <StackContext.Provider value={value}>
      <StackOrderContext.Provider value={order}>{children}</StackOrderContext.Provider>
    </StackContext.Provider>
  );
}
