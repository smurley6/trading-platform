"""In-process paper broker: fills market orders immediately at the current quote.

Long-only equities, full fills, zero fees. Idempotent on client_order_id (FR-EXE-3).
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderNotFoundError,
)
from trading.application.ports.clock import ClockPort
from trading.application.ports.market_data import MarketDataPort
from trading.domain.orders import Fill, OrderSide, OrderStatus, OrderType


@dataclass
class _Holding:
    quantity: Decimal
    avg_entry_price: Decimal


class PaperBroker:
    def __init__(
        self, *, market_data: MarketDataPort, clock: ClockPort, starting_cash: Decimal
    ) -> None:
        self._market_data = market_data
        self._clock = clock
        self._cash = starting_cash
        self._holdings: dict[str, _Holding] = {}
        self._orders: dict[str, BrokerOrder] = {}
        self._lock = asyncio.Lock()
        self._order_counter = 0

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        async with self._lock:
            existing = self._orders.get(client_order_id)
            if existing is not None:
                return existing  # idempotent resubmission

            if order_type is not OrderType.MARKET:
                return self._record_rejection(client_order_id, symbol, side, quantity, order_type)

            quote = await self._market_data.get_quote(symbol)
            price = quote.ask if side is OrderSide.BUY else quote.bid
            if side is OrderSide.BUY and price * quantity > self._cash:
                return self._record_rejection(client_order_id, symbol, side, quantity, order_type)
            held = self._holdings.get(symbol)
            if side is OrderSide.SELL and (held is None or held.quantity < quantity):
                return self._record_rejection(client_order_id, symbol, side, quantity, order_type)

            if side is OrderSide.BUY:
                self._cash -= price * quantity
                if held is None:
                    self._holdings[symbol] = _Holding(quantity, price)
                else:
                    total = held.quantity + quantity
                    held.avg_entry_price = (
                        held.quantity * held.avg_entry_price + quantity * price
                    ) / total
                    held.quantity = total
            else:
                assert held is not None
                self._cash += price * quantity
                held.quantity -= quantity
                if held.quantity == 0:
                    del self._holdings[symbol]

            fill = Fill(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                fees=Decimal("0"),
                filled_at=self._clock.now(),
            )
            order = BrokerOrder(
                client_order_id=client_order_id,
                broker_order_id=self._next_broker_order_id(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.FILLED,
                fills=(fill,),
            )
            self._orders[client_order_id] = order
            return order

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                raise OrderNotFoundError(f"no order with client_order_id {client_order_id}")
            # Paper fills are immediate, so every order is already terminal — nothing to
            # cancel. Return current state (no-op), consistent with the port contract.
            return order

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return list(self._orders.values())

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return [
            BrokerPosition(symbol, holding.quantity, holding.avg_entry_price)
            for symbol, holding in self._holdings.items()
        ]

    async def get_account(self) -> BrokerAccount:
        equity = self._cash
        for symbol, holding in self._holdings.items():
            quote = await self._market_data.get_quote(symbol)
            equity += holding.quantity * quote.last
        return BrokerAccount(cash=self._cash, equity=equity)

    def _record_rejection(
        self,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType,
    ) -> BrokerOrder:
        order = BrokerOrder(
            client_order_id=client_order_id,
            broker_order_id=self._next_broker_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=OrderStatus.REJECTED,
        )
        self._orders[client_order_id] = order
        return order

    def _next_broker_order_id(self) -> str:
        self._order_counter += 1
        return f"paper-{self._order_counter}"
