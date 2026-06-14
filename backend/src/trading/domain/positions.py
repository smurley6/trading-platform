"""Positions and per-sleeve portfolio state (long-only equities for M1)."""

from collections.abc import Mapping
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
