// Markets screen — live stock tracking built on TradingView embed widgets (genuinely live,
// no API keys). Layout: a scrolling ticker tape across the top; a watchlist panel beside a
// large advanced chart (clicking a watchlist symbol drives the chart); then a market overview
// and a grid of mini sparkline charts.
//
// Watchlist symbols come from the backend (`GET /quotes/tracked`); when that returns nothing we
// fall back to a sensible default set. An OPTIONAL numeric quote feed (`GET /quotes`) overlays
// live bid/ask/last next to each symbol — but it DEFAULTS TO OFF (no credentials), so the screen
// is designed to look complete with charts alone and only paints numbers when `enabled` is true.
// A disabled or failing feed never breaks the screen (the api client resolves errors to a
// disabled shape).

import { StackLayout } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { AnimatedPrice } from "../../components/AnimatedPrice";
import { MarketOverview } from "../../components/MarketOverview";
import { MiniSymbolChart } from "../../components/MiniSymbolChart";
import { SymbolChart } from "../../components/SymbolChart";
import { TickerTape } from "../../components/TickerTape";
import { tvSymbol } from "../../components/tvSymbol";
import { Badge, EmptyState, LoadingState, Panel, SectionHeader } from "../../components/ui";
import { api } from "../../lib/api";
import { formatNumber } from "../../lib/format";
import { color, font, numeric, radius, space, text, transitions } from "../../lib/tokens";
import type { Quote } from "../../lib/types";
import { useLoader } from "../../lib/useLoader";
import { useQuoteStream } from "../../lib/useQuoteStream";

const DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "SPCX"];

interface MarketsProps {
  refreshKey: number;
}

export function Markets({ refreshKey }: MarketsProps) {
  const tracked = useLoader(() => api.trackedSymbols(), [refreshKey]);

  const symbols = useMemo(() => {
    const fromBackend = tracked.data?.symbols ?? [];
    return fromBackend.length > 0 ? fromBackend : DEFAULT_SYMBOLS;
  }, [tracked.data]);

  // Targeted buys: the top-10 proposed BUY tickers from the daily cycle. Auto-updates whenever the
  // live event stream bumps `refreshKey`. The server already sorts by conviction desc and caps to
  // ten, so we just take the proposals in order.
  const proposals = useLoader(() => api.proposals(), [refreshKey]);
  const targetedBuys = useMemo(
    () => (proposals.data?.proposals ?? []).filter((p) => p.side === "buy"),
    [proposals.data],
  );

  // Live numeric quote overlay (optional). Driven by the real-time stream hook: it pushes
  // bid/ask/last updates as prices move (or falls back to interval polling), and stays fully
  // dormant when the feed is off — so the watchlist is charts-only / FEED OFF exactly as today.
  // Feed BOTH lists' symbols (de-duped) so quotes cover the big-ticket watchlist AND targeted buys.
  const feedSymbols = useMemo(() => {
    const seen = new Set<string>();
    const merged: string[] = [];
    for (const sym of [...symbols, ...targetedBuys.map((p) => p.symbol)]) {
      if (!seen.has(sym)) {
        seen.add(sym);
        merged.push(sym);
      }
    }
    return merged;
  }, [symbols, targetedBuys]);
  const { enabled: numericFeedOn, quotes: quoteBySymbol } = useQuoteStream(feedSymbols);

  const [selected, setSelected] = useState<string | null>(null);
  const active = selected ?? symbols[0] ?? "AAPL";

  const trackedByPlatform = Boolean(tracked.data && tracked.data.symbols.length > 0);

  return (
    <StackLayout gap={2}>
      <Panel padded={false} style={{ overflow: "hidden" }}>
        <TickerTape symbols={symbols} />
      </Panel>

      <div
        style={{ display: "grid", gridTemplateColumns: "minmax(260px, 340px) 1fr", gap: space.lg }}
      >
        <StackLayout gap={2}>
          <Panel>
            <SectionHeader
              eyebrow="Watchlist"
              title="Big-ticket"
              caption={
                trackedByPlatform
                  ? "Tracked by the platform"
                  : "Default set (no tracked symbols yet)"
              }
              actions={<FeedBadge enabled={numericFeedOn} />}
            />
            <div style={{ display: "flex", flexDirection: "column", gap: space.xs }}>
              {symbols.map((sym) => (
                <WatchlistItem
                  key={sym}
                  symbol={sym}
                  active={sym === active}
                  quote={quoteBySymbol.get(sym)}
                  onSelect={() => setSelected(sym)}
                />
              ))}
            </div>
          </Panel>

          <Panel>
            <SectionHeader
              eyebrow="Daily cycle"
              title="Targeted buys"
              caption="Top BUY proposals from the daily cycle"
              actions={
                <Badge tone="accent" dot>
                  AUTO — TOP 10
                </Badge>
              }
            />
            {proposals.loading && targetedBuys.length === 0 ? (
              <LoadingState label="Loading proposals…" style={{ padding: `${space.lg}px 0` }} />
            ) : targetedBuys.length === 0 ? (
              <EmptyState
                message="No targeted buys yet — the cycle hasn't proposed any."
                style={{ padding: `${space.lg}px 0` }}
              />
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: space.xs }}>
                {targetedBuys.map((p) => (
                  <WatchlistItem
                    key={p.symbol}
                    symbol={p.symbol}
                    active={p.symbol === active}
                    quote={quoteBySymbol.get(p.symbol)}
                    secondary={`conv ${Math.round(p.conviction_score)}`}
                    onSelect={() => setSelected(p.symbol)}
                  />
                ))}
              </div>
            )}
          </Panel>
        </StackLayout>

        <Panel>
          <SectionHeader
            eyebrow="Advanced chart"
            title={active}
            caption="Live chart via TradingView"
          />
          <SymbolChart symbol={tvSymbol(active)} height={460} allowSymbolChange withDetails />
        </Panel>
      </div>

      <Panel>
        <SectionHeader eyebrow="Snapshot" title="Market overview" />
        <MarketOverview symbols={symbols} />
      </Panel>

      <Panel>
        <SectionHeader
          eyebrow="Sparklines"
          title="Mini charts"
          caption="Click any symbol to drive the advanced chart"
        />
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: space.md,
          }}
        >
          {symbols.map((sym) => {
            const isActive = sym === active;
            return (
              <div
                key={sym}
                role="button"
                tabIndex={0}
                onClick={() => setSelected(sym)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") setSelected(sym);
                }}
                style={{
                  cursor: "pointer",
                  borderRadius: radius.lg,
                  border: `1px solid ${isActive ? color.accentBorder : color.hairline}`,
                  boxShadow: isActive ? `0 0 0 1px ${color.accentBorder}` : "none",
                  overflow: "hidden",
                  transition: transitions("border-color", "box-shadow"),
                }}
              >
                <MiniSymbolChart symbol={sym} />
              </div>
            );
          })}
        </div>
      </Panel>
    </StackLayout>
  );
}

function FeedBadge({ enabled }: { enabled: boolean }) {
  return (
    <Badge
      tone={enabled ? "ok" : "muted"}
      dot={enabled}
      title={
        enabled
          ? "Numeric quote feed enabled"
          : "Numeric feed off — charts are live via TradingView"
      }
    >
      {enabled ? "LIVE QUOTES" : "FEED OFF"}
    </Badge>
  );
}

function WatchlistItem({
  symbol,
  active,
  quote,
  secondary,
  onSelect,
}: {
  symbol: string;
  active: boolean;
  quote: Quote | undefined;
  // Optional secondary line shown in place of bid/ask when there's no live quote (e.g. "conv 100").
  secondary?: string;
  onSelect: () => void;
}) {
  const last = quote ? Number(quote.last) : NaN;
  const bid = quote ? Number(quote.bid) : NaN;
  const ask = quote ? Number(quote.ask) : NaN;

  return (
    <button
      onClick={onSelect}
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: space.sm,
        cursor: "pointer",
        textAlign: "left",
        color: color.text,
        borderRadius: radius.md,
        padding: "8px 10px",
        border: `1px solid ${active ? color.accentBorder : color.hairline}`,
        background: active ? color.accentSoft : color.panelSoft,
        fontFamily: font.sans,
        transition: transitions("background", "border-color"),
      }}
    >
      <span
        style={{ ...text.label, fontWeight: 700, color: active ? color.accentText : color.text }}
      >
        {symbol}
      </span>
      {quote ? (
        <span style={{ textAlign: "right", lineHeight: 1.3 }}>
          <AnimatedPrice
            value={last}
            decimals={2}
            style={{ ...numeric(), fontSize: 14, fontWeight: 600, display: "inline-block" }}
          />
          <br />
          <span style={{ ...numeric(), fontSize: 11, color: color.muted }}>
            {Number.isNaN(bid) ? "—" : formatNumber(bid, 2)} /{" "}
            {Number.isNaN(ask) ? "—" : formatNumber(ask, 2)}
          </span>
        </span>
      ) : (
        <span style={{ ...text.caption, ...numeric(), color: color.faint }}>
          {secondary ?? "chart →"}
        </span>
      )}
    </button>
  );
}
