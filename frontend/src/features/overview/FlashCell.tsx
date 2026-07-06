// FlashFigure — the presentational half of the §3.4 cell-flash pair; see useCellFlash.ts for the
// animation/debounce logic (split into its own file so this one stays component-only, per the
// project's react-refresh convention).

import { layout, shape } from "../../lib/tokens";
import { useCellFlash } from "./useCellFlash";

export function FlashFigure({
  value,
  flashKey,
  children,
  color,
}: {
  value: number;
  flashKey: string;
  children: React.ReactNode;
  color?: string;
}) {
  const ref = useCellFlash(value, flashKey);
  return (
    <span
      ref={ref}
      className="tp-numeric"
      style={{
        display: "inline-block",
        padding: `1px ${layout.x1}px`,
        borderRadius: shape.radiusBadge,
        color,
      }}
    >
      {children}
    </span>
  );
}
