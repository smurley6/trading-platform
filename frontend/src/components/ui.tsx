// Shared visual primitives for the new design language. Feature screens compose these instead of
// re-deriving card/border/figure styles inline, so every surface reads as one system. All props
// are additive/optional; nothing here wires data — this is presentation only.

import { Spinner } from "@salt-ds/core";
import type { CSSProperties, ReactNode } from "react";
import {
  color,
  eyebrowStyle,
  font,
  numeric,
  pillStyle,
  pnlColor,
  radius,
  sectionHeaderStyle,
  sectionTitleStyle,
  shadow,
  space,
  text,
  transitions,
} from "../lib/tokens";

// ---------------------------------------------------------------------------------------------
// Panel — the standard raised content card. A thin, palette-correct alternative to Salt's Card
// for surfaces we fully own. Salt's <Card> is still fine where it's already used.
// ---------------------------------------------------------------------------------------------
export function Panel({
  children,
  raised,
  padded = true,
  style,
  ...rest
}: {
  children: ReactNode;
  raised?: boolean;
  padded?: boolean;
  style?: CSSProperties;
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      style={{
        background: raised ? color.panelRaised : color.panel,
        border: `1px solid ${color.hairline}`,
        borderRadius: radius.lg,
        boxShadow: raised ? shadow.md : shadow.sm,
        padding: padded ? space.xl : 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// SectionHeader — a titled row with an optional eyebrow/caption and right-aligned actions.
// ---------------------------------------------------------------------------------------------
export function SectionHeader({
  title,
  eyebrow,
  caption,
  actions,
  style,
}: {
  title: ReactNode;
  eyebrow?: ReactNode;
  caption?: ReactNode;
  actions?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div style={{ ...sectionHeaderStyle(), ...style }}>
      <div style={{ minWidth: 0 }}>
        {eyebrow != null && <div style={eyebrowStyle()}>{eyebrow}</div>}
        <div style={sectionTitleStyle()}>{title}</div>
        {caption != null && (
          <div style={{ ...text.caption, color: color.muted, marginTop: 2 }}>{caption}</div>
        )}
      </div>
      {actions != null && (
        <div style={{ display: "flex", alignItems: "center", gap: space.sm, flexShrink: 0 }}>
          {actions}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// MetricTile — the KPI figure used across dashboards. Mono/tabular value, optional tone + delta.
// ---------------------------------------------------------------------------------------------
export function MetricTile({
  label,
  value,
  tone = "neutral",
  delta,
  hint,
  style,
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: "neutral" | "ok" | "danger" | "accent" | "warn";
  /** Optional signed delta figure shown under the value; number auto-colors by sign. */
  delta?: ReactNode | number;
  hint?: ReactNode;
  style?: CSSProperties;
}) {
  const valueColor =
    tone === "neutral"
      ? color.text
      : tone === "ok"
        ? color.ok
        : tone === "danger"
          ? color.dangerText
          : tone === "warn"
            ? color.warnText
            : color.accentText;
  const deltaColor = typeof delta === "number" ? pnlColor(delta) : color.muted;
  return (
    <Panel style={{ display: "flex", flexDirection: "column", gap: space.xs, ...style }}>
      <div style={eyebrowStyle()}>{label}</div>
      <div style={{ ...text.h1, ...numeric(), color: valueColor }}>{value}</div>
      {delta != null && (
        <div style={{ ...text.caption, ...numeric(), color: deltaColor }}>
          {typeof delta === "number" ? `${delta >= 0 ? "+" : ""}${delta}` : delta}
        </div>
      )}
      {hint != null && <div style={{ ...text.caption, color: color.muted }}>{hint}</div>}
    </Panel>
  );
}

// ---------------------------------------------------------------------------------------------
// Badge — thin wrapper over pillStyle so screens don't re-derive it. Optional leading dot.
// ---------------------------------------------------------------------------------------------
export function Badge({
  children,
  tone = "muted",
  dot,
  title,
  style,
}: {
  children: ReactNode;
  tone?: "accent" | "ok" | "danger" | "warn" | "muted";
  dot?: boolean;
  title?: string;
  style?: CSSProperties;
}) {
  const base = pillStyle(tone);
  return (
    <span title={title} style={{ ...base, ...style }}>
      {dot && (
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "currentColor",
            boxShadow: "0 0 6px currentColor",
          }}
        />
      )}
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------------------------
// EmptyState — the standard "nothing here yet" placeholder for cards/tables.
// ---------------------------------------------------------------------------------------------
export function EmptyState({
  title,
  message,
  icon,
  action,
  style,
}: {
  title?: ReactNode;
  message: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        gap: space.sm,
        padding: `${space.xxl}px ${space.lg}px`,
        color: color.muted,
        ...style,
      }}
    >
      {icon != null && <div style={{ fontSize: 22, opacity: 0.7 }}>{icon}</div>}
      {title != null && <div style={{ ...text.h3, color: color.textDim }}>{title}</div>}
      <div style={{ ...text.body, color: color.muted, maxWidth: 420 }}>{message}</div>
      {action != null && <div style={{ marginTop: space.xs }}>{action}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// LoadingState — centered spinner + optional label for card/table loading.
// ---------------------------------------------------------------------------------------------
export function LoadingState({
  label = "Loading…",
  style,
}: {
  label?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: space.md,
        padding: `${space.xxl}px ${space.lg}px`,
        color: color.muted,
        ...style,
      }}
    >
      <Spinner size="medium" />
      {label != null && <div style={{ ...text.caption, color: color.muted }}>{label}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// Figure — inline mono/tabular numeric span (prices, P&L). Auto-colors when `pnl` is set.
// ---------------------------------------------------------------------------------------------
export function Figure({
  children,
  pnl,
  color: colorProp,
  weight = 600,
  style,
}: {
  children: ReactNode;
  /** Numeric value to color by sign (green/red). Overridden by explicit `color`. */
  pnl?: number;
  color?: string;
  weight?: number;
  style?: CSSProperties;
}) {
  const resolved = colorProp ?? (pnl != null ? pnlColor(pnl) : undefined);
  return (
    <span style={{ ...numeric(), fontWeight: weight, color: resolved, ...style }}>{children}</span>
  );
}

// ---------------------------------------------------------------------------------------------
// Segmented — a pill toggle group (e.g. sleeve selectors, view switches). Presentational.
// ---------------------------------------------------------------------------------------------
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = "md",
  style,
}: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (value: T) => void;
  size?: "sm" | "md";
  style?: CSSProperties;
}) {
  return (
    <div
      role="tablist"
      style={{
        display: "inline-flex",
        padding: 2,
        gap: 2,
        borderRadius: radius.lg,
        background: color.panelInset,
        border: `1px solid ${color.hairline}`,
        ...style,
      }}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            style={{
              cursor: "pointer",
              border: "1px solid transparent",
              borderRadius: radius.md,
              padding: size === "sm" ? "3px 10px" : "5px 14px",
              fontFamily: font.sans,
              fontSize: size === "sm" ? 12 : 13,
              fontWeight: 600,
              color: active ? color.accentText : color.muted,
              background: active ? color.accentSoft : "transparent",
              borderColor: active ? color.accentBorder : "transparent",
              transition: transitions("background", "color", "border-color"),
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
