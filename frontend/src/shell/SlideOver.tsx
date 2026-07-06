// Shared right slide-over surface (docs/UI-DESIGN.md §3.5, §6) for Chain Inspector and Ticker
// Pane — "same surface as Ticker Pane, they stack". Built on @radix-ui/react-dialog (§10) so
// focus-trap / inert background / Esc handling come for free; this file only adds the layout
// (fixed right pane, translate+fade enter) and the stacking depth cue.
//
// Stacking (§3.5 "Overlay behavior"): opening a second slide-over pushes the first left 8px at
// 60% opacity — still one Esc per layer, since each is its own Radix Dialog.Root and Radix only
// closes the topmost (last-mounted) one on Escape. `depth` is 0 for the frontmost pane, 1 for one
// pushed back, etc. — SlideOverStackProvider + useSlideOverDepth (shell/slideOverStack.ts) compute
// it from mount order.
//
// Compositor-only per §3.5: the enter animation and the stacking push both animate only
// `transform`/`opacity`, never width/height/left/right (see the .tp-slideover-* rules in
// theme.css).

import * as Dialog from "@radix-ui/react-dialog";
import type { ReactNode } from "react";

export interface SlideOverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  /** Stack depth: 0 = frontmost, 1 = pushed back one layer, etc. */
  depth: number;
  width?: number;
  children: ReactNode;
  /** Focus returns here on close (§3.5 "focus returns to the invoking row"). */
  returnFocusRef?: React.RefObject<HTMLElement>;
}

export function SlideOver({
  open,
  onOpenChange,
  title,
  depth,
  width = 420,
  children,
  returnFocusRef,
}: SlideOverProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="tp-slideover-overlay" style={{ zIndex: 1300 + depth * 10 }} />
        <Dialog.Content
          className="tp-slideover-content tp-shell"
          data-depth={depth}
          aria-describedby={undefined}
          style={{
            width,
            zIndex: 1301 + depth * 10,
          }}
          onCloseAutoFocus={(e) => {
            if (returnFocusRef?.current) {
              e.preventDefault();
              returnFocusRef.current.focus();
            }
          }}
        >
          <Dialog.Title className="tp-slideover-title">{title}</Dialog.Title>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
