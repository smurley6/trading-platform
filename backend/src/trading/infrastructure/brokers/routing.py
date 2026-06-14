"""Mode-routing broker: directs a sleeve's orders to its paper or live venue at runtime.

This is how the paper->live switch is "config only" at the boundary (spec section 11.1): the
strategy, risk gate, and execution use cases hold one `BrokerPort` and never change; this
adapter decides, per call, which underlying broker to hit based on the sleeve's current mode.
If a sleeve is somehow in live mode with no live broker wired, orders are refused rather than
silently falling back to paper.
"""

from collections.abc import Callable, Sequence
from decimal import Decimal

from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPort,
    BrokerPosition,
)
from trading.domain.orders import OrderSide, OrderType
from trading.domain.sleeves import TradingMode


class LiveBrokerUnavailableError(Exception):
    pass


class ModeRoutingBroker:
    def __init__(
        self,
        *,
        paper: BrokerPort,
        live: BrokerPort | None,
        mode: Callable[[], TradingMode],
    ) -> None:
        self._paper = paper
        self._live = live
        self._mode = mode

    def _active(self) -> BrokerPort:
        if self._mode() is TradingMode.LIVE:
            if self._live is None:
                raise LiveBrokerUnavailableError(
                    "sleeve is in live mode but no live broker is configured"
                )
            return self._live
        return self._paper

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        return await self._active().submit_order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
        )

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return await self._active().list_orders()

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return await self._active().list_positions()

    async def get_account(self) -> BrokerAccount:
        return await self._active().get_account()
