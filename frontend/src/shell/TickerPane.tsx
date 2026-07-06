// Ticker Pane (docs/UI-DESIGN.md §2 "demotion of Markets") — a right slide-over hosting the
// TradingView-style symbol chart, invoked from any ticker anywhere (click or `t` on a focused
// row). Reuses the existing SymbolChart/tvSymbol components as-is (display-only embed, unchanged
// per §9 out-of-scope: "any change to API auth or backend architecture beyond the three endpoints
// named in §8" — this is purely a new container around an existing component).
//
// Route-addressable as `?ticker=LMT` via useTickerQueryParam (shell/useHistoryRoute.ts) so deep
// links survive reload — Overview.tsx owns wiring that hook's state to this component's
// open/close. Stacks with the Chain Inspector via SlideOver's shared depth cue.

import { SymbolChart } from "../components/SymbolChart";
import { tvSymbol } from "../components/tvSymbol";
import { SlideOver } from "./SlideOver";
import { useSlideOverDepth } from "./slideOverStack";

interface TickerPaneProps {
  /** null = closed. A ticker symbol opens the pane on that symbol. */
  ticker: string | null;
  onClose: () => void;
  returnFocusRef?: React.RefObject<HTMLElement>;
}

export function TickerPane({ ticker, onClose, returnFocusRef }: TickerPaneProps) {
  const open = ticker !== null;
  const depth = useSlideOverDepth("ticker-pane", open);

  return (
    <SlideOver
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
      title={ticker ?? ""}
      depth={depth}
      width={520}
      returnFocusRef={returnFocusRef}
    >
      <div style={{ flex: 1, minHeight: 0 }}>
        {ticker && <SymbolChart symbol={tvSymbol(ticker)} height={520} />}
      </div>
    </SlideOver>
  );
}
