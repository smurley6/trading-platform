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

import { Card, H3, StackLayout, Text } from "@salt-ds/core";
import { useMemo, useState } from "react";
import { AnimatedPrice } from "../../components/AnimatedPrice";
import { MarketOverview } from "../../components/MarketOverview";
import { MiniSymbolChart } from "../../components/MiniSymbolChart";
import { SymbolChart } from "../../components/SymbolChart";
import { TickerTape } from "../../components/TickerTape";
import { tvSymbol } from "../../components/tvSymbol";
import { api } from "../../lib/api";
import { formatNumber } from "../../lib/format";
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

  return (
    <StackLayout gap={2}>
      <Card style={{ padding: 0, overflow: "hidden" }}>
        <TickerTape symbols={symbols} />
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 320px) 1fr", gap: 16 }}>
        <StackLayout gap={2}>
          <Card>
            <div
              style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}
            >
              <H3 style={{ margin: 0 }}>Big-ticket</H3>
              <FeedBadge enabled={numericFeedOn} />
            </div>
            <Text color="secondary" styleAs="label">
              {tracked.data && tracked.data.symbols.length > 0
                ? "Tracked by the platform"
                : "Default set (no tracked symbols yet)"}
            </Text>
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
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
          </Card>

          <Card>
            <div
              style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}
            >
              <H3 style={{ margin: 0 }}>Targeted buys</H3>
              <TargetedBadge />
            </div>
            <Text color="secondary" styleAs="label">
              Top BUY proposals from the daily cycle
            </Text>
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
              {proposals.loading && targetedBuys.length === 0 ? (
                <Text color="secondary" styleAs="label" style={{ padding: "4px 2px" }}>
                  Loading proposals…
                </Text>
              ) : targetedBuys.length === 0 ? (
                <Text color="secondary" styleAs="label" style={{ padding: "4px 2px" }}>
                  No targeted buys yet — the cycle hasn't proposed any.
                </Text>
              ) : (
                targetedBuys.map((p) => (
                  <WatchlistItem
                    key={p.symbol}
                    symbol={p.symbol}
                    active={p.symbol === active}
                    quote={quoteBySymbol.get(p.symbol)}
                    secondary={`conv ${Math.round(p.conviction_score)}`}
                    onSelect={() => setSelected(p.symbol)}
                  />
                ))
              )}
            </div>
          </Card>
        </StackLayout>

        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <H3 style={{ margin: 0 }}>{active}</H3>
            <Text color="secondary" styleAs="label">
              Live chart via TradingView
            </Text>
          </div>
          <SymbolChart symbol={tvSymbol(active)} height={460} allowSymbolChange withDetails />
        </Card>
      </div>

      <Card>
        <H3 style={{ marginTop: 0 }}>Market overview</H3>
        <MarketOverview symbols={symbols} />
      </Card>

      <Card>
        <H3 style={{ marginTop: 0 }}>Mini charts</H3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
            gap: 12,
          }}
        >
          {symbols.map((sym) => (
            <div
              key={sym}
              role="button"
              tabIndex={0}
              onClick={() => setSelected(sym)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") setSelected(sym);
              }}
              style={{ cursor: "pointer" }}
            >
              <MiniSymbolChart symbol={sym} />
            </div>
          ))}
        </div>
      </Card>
    </StackLayout>
  );
}

function FeedBadge({ enabled }: { enabled: boolean }) {
  return (
    <span
      title={
        enabled
          ? "Numeric quote feed enabled"
          : "Numeric feed off — charts are live via TradingView"
      }
      style={{
        padding: "2px 8px",
        borderRadius: 10,
        fontSize: 11,
        fontWeight: 700,
        whiteSpace: "nowrap",
        background: enabled ? "rgba(34,197,94,0.18)" : "rgba(148,163,184,0.15)",
        color: enabled ? "#4ade80" : "#94a3b8",
        border: `1px solid ${enabled ? "#22c55e" : "rgba(148,163,184,0.4)"}`,
      }}
    >
      {enabled ? "LIVE QUOTES" : "FEED OFF"}
    </span>
  );
}

// Signals that the "Targeted buys" card auto-tracks the cycle's top-10 BUY proposals. Mirrors the
// FeedBadge styling so the two cards read as a set.
function TargetedBadge() {
  return (
    <span
      title="Auto-tracks the top 10 BUY proposals from the daily cycle"
      style={{
        padding: "2px 8px",
        borderRadius: 10,
        fontSize: 11,
        fontWeight: 700,
        whiteSpace: "nowrap",
        background: "rgba(76,154,255,0.18)",
        color: "#4c9aff",
        border: "1px solid rgba(76,154,255,0.5)",
      }}
    >
      AUTO — top 10
    </span>
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
        gap: 8,
        cursor: "pointer",
        textAlign: "left",
        color: "inherit",
        borderRadius: 6,
        padding: "8px 10px",
        border: "1px solid rgba(255,255,255,0.12)",
        background: active ? "rgba(76,154,255,0.2)" : "transparent",
      }}
    >
      <span style={{ fontWeight: 600, fontSize: 14 }}>{symbol}</span>
      {quote ? (
        <span style={{ textAlign: "right", lineHeight: 1.3 }}>
          <AnimatedPrice
            value={last}
            decimals={2}
            style={{ fontSize: 14, fontWeight: 600, display: "inline-block" }}
          />
          <br />
          <span style={{ fontSize: 11, color: "#94a3b8" }}>
            {Number.isNaN(bid) ? "—" : formatNumber(bid, 2)} /{" "}
            {Number.isNaN(ask) ? "—" : formatNumber(ask, 2)}
          </span>
        </span>
      ) : (
        <span style={{ fontSize: 11, color: "#6b7280" }}>{secondary ?? "chart →"}</span>
      )}
    </button>
  );
}
