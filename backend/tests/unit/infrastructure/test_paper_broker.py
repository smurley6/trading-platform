from decimal import Decimal

import pytest

from trading.application.ports.broker import OrderNotFoundError
from trading.domain.orders import OrderSide, OrderStatus
from trading.infrastructure.brokers.paper import PaperBroker


async def test_buy_fills_at_quote_and_updates_account(broker: PaperBroker) -> None:
    order = await broker.submit_order(
        client_order_id="ord-1", symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("5")
    )

    assert order.status is OrderStatus.FILLED
    assert order.fills[0].price == Decimal("100")

    account = await broker.get_account()
    assert account.cash == Decimal("9500")
    assert account.equity == Decimal("10000")  # cash + 5 * 100

    positions = {p.symbol: p for p in await broker.list_positions()}
    assert positions["AAPL"].quantity == Decimal("5")


async def test_resubmission_is_idempotent(broker: PaperBroker) -> None:
    first = await broker.submit_order(
        client_order_id="ord-1", symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("5")
    )
    second = await broker.submit_order(
        client_order_id="ord-1", symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("5")
    )

    assert second is first
    assert (await broker.get_account()).cash == Decimal("9500")  # not double-charged


async def test_insufficient_cash_rejects(broker: PaperBroker) -> None:
    order = await broker.submit_order(
        client_order_id="ord-1", symbol="MSFT", side=OrderSide.SELL, quantity=Decimal("1")
    )
    assert order.status is OrderStatus.REJECTED

    big = await broker.submit_order(
        client_order_id="ord-2", symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("999")
    )
    assert big.status is OrderStatus.REJECTED
    assert (await broker.get_account()).cash == Decimal("10000")


async def test_cancel_unknown_order_raises(broker: PaperBroker) -> None:
    with pytest.raises(OrderNotFoundError):
        await broker.cancel_order(client_order_id="nope")


async def test_cancel_known_order_is_noop(broker: PaperBroker) -> None:
    # Paper fills are immediate, so cancel returns the (terminal) order unchanged.
    submitted = await broker.submit_order(
        client_order_id="ord-1", symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("5")
    )
    canceled = await broker.cancel_order(client_order_id="ord-1")
    assert canceled is submitted
    assert canceled.status is OrderStatus.FILLED


async def test_round_trip_returns_cash(broker: PaperBroker) -> None:
    await broker.submit_order(
        client_order_id="ord-1", symbol="AAPL", side=OrderSide.BUY, quantity=Decimal("5")
    )
    await broker.submit_order(
        client_order_id="ord-2", symbol="AAPL", side=OrderSide.SELL, quantity=Decimal("5")
    )

    assert (await broker.get_account()).cash == Decimal("10000")
    assert await broker.list_positions() == []
