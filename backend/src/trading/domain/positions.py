"""Positions and per-sleeve portfolio state (long-only equities for M1)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from trading.domain.orders import Fill, OrderSide


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: Decimal
    avg_entry_price: Decimal

    def market_value(self, price: Decimal) -> Decimal:
        return self.quantity * price


@dataclass(frozen=True, slots=True)
class HeldPosition:
    """A held lot to be marked: identity (symbol + owning sleeve) plus cost basis."""

    symbol: str
    sleeve_id: str
    quantity: Decimal
    avg_price: Decimal


@dataclass(frozen=True, slots=True)
class MarkedPosition:
    symbol: str
    sleeve_id: str
    quantity: Decimal
    avg_price: Decimal
    last: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


def mark_positions(
    positions: Sequence[HeldPosition], marks: Mapping[str, Decimal]
) -> list[MarkedPosition]:
    """Pure: value each held position at a mark price and compute unrealized P&L.

    `marks` maps symbol -> last price (e.g. from a live quote feed). A symbol absent from
    `marks` falls back to its own `avg_price`, so a disabled/partial feed still yields useful
    numbers (zero unrealized P&L) and never raises (FR live-P&L deliverable).
    """
    marked: list[MarkedPosition] = []
    for held in positions:
        last = marks.get(held.symbol, held.avg_price)
        market_value = held.quantity * last
        unrealized = (last - held.avg_price) * held.quantity
        marked.append(
            MarkedPosition(
                symbol=held.symbol,
                sleeve_id=held.sleeve_id,
                quantity=held.quantity,
                avg_price=held.avg_price,
                last=last,
                market_value=market_value,
                unrealized_pnl=unrealized,
            )
        )
    return marked


@dataclass(frozen=True, slots=True)
class Portfolio:
    cash: Decimal
    positions: dict[str, Position] = field(default_factory=dict)

    def quantity(self, symbol: str) -> Decimal:
        position = self.positions.get(symbol)
        return position.quantity if position else Decimal("0")

    def equity(self, prices: Mapping[str, Decimal]) -> Decimal:
        total = self.cash
        for position in self.positions.values():
            total += position.market_value(prices[position.symbol])
        return total

    def apply_fill(self, fill: Fill) -> "Portfolio":
        positions = dict(self.positions)
        existing = positions.get(fill.symbol)
        if fill.side is OrderSide.BUY:
            cash = self.cash - fill.price * fill.quantity - fill.fees
            if existing:
                total_qty = existing.quantity + fill.quantity
                avg = (
                    existing.quantity * existing.avg_entry_price + fill.quantity * fill.price
                ) / total_qty
                positions[fill.symbol] = Position(fill.symbol, total_qty, avg)
            else:
                positions[fill.symbol] = Position(fill.symbol, fill.quantity, fill.price)
        else:
            held = existing.quantity if existing else Decimal("0")
            if fill.quantity > held:
                raise ValueError(
                    f"cannot sell {fill.quantity} {fill.symbol}: only {held} held (long-only)"
                )
            cash = self.cash + fill.price * fill.quantity - fill.fees
            remaining = held - fill.quantity
            if existing and remaining > 0:
                positions[fill.symbol] = Position(fill.symbol, remaining, existing.avg_entry_price)
            else:
                positions.pop(fill.symbol, None)
        return Portfolio(cash=cash, positions=positions)
