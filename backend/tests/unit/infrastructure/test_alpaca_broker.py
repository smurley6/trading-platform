"""AlpacaBroker against a mock HTTP transport — real request shaping + response parsing,
no network. Verifies the live broker satisfies BrokerPort and is idempotent (FR-EXE-3)."""

from collections.abc import Callable
from decimal import Decimal

import httpx
import pytest

from trading.application.ports.broker import OrderNotFoundError
from trading.domain.orders import OrderSide, OrderStatus, OrderType
from trading.infrastructure.brokers.alpaca import AlpacaBroker, AlpacaBrokerError

Handler = Callable[[httpx.Request], httpx.Response]

_FILLED_ORDER = {
    "id": "broker-1",
    "client_order_id": "ord-1",
    "symbol": "LMT",
    "qty": "10",
    "filled_qty": "10",
    "filled_avg_price": "450.25",
    "side": "buy",
    "type": "market",
    "status": "filled",
    "filled_at": "2026-06-13T14:00:00Z",
}


def _broker(handler: Handler) -> AlpacaBroker:
    return AlpacaBroker(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        key_id="k",
        secret_key="s",
        base_url="https://api.alpaca.markets",
    )


async def test_submit_sends_auth_and_parses_fill() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("APCA-API-KEY-ID")
        captured["path"] = request.url.path
        return httpx.Response(200, json=_FILLED_ORDER)

    order = await _broker(handler).submit_order(
        client_order_id="ord-1", symbol="LMT", side=OrderSide.BUY, quantity=Decimal("10")
    )

    assert captured["auth"] == "k"
    assert captured["path"] == "/v2/orders"
    assert order.status is OrderStatus.FILLED
    assert order.broker_order_id == "broker-1"
    assert order.fills[0].price == Decimal("450.25")
    assert order.fills[0].quantity == Decimal("10")


async def test_duplicate_client_order_id_is_idempotent() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            return httpx.Response(422, json={"message": "client_order_id must be unique"})
        return httpx.Response(200, json=_FILLED_ORDER)

    order = await _broker(handler).submit_order(
        client_order_id="ord-1", symbol="LMT", side=OrderSide.BUY, quantity=Decimal("10")
    )

    assert order.client_order_id == "ord-1"
    assert "POST /v2/orders" in calls
    assert "GET /v2/orders:by_client_order_id" in calls  # resolved existing instead of dup


async def test_submit_error_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    with pytest.raises(AlpacaBrokerError):
        await _broker(handler).submit_order(
            client_order_id="ord-1", symbol="LMT", side=OrderSide.BUY, quantity=Decimal("1")
        )


async def test_list_positions_and_account() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/positions":
            return httpx.Response(
                200,
                json=[{"symbol": "LMT", "qty": "10", "avg_entry_price": "450.25"}],
            )
        return httpx.Response(
            200,
            json={"cash": "55000.00", "equity": "100000.00", "buying_power": "180000.00"},
        )

    broker = _broker(handler)
    positions = await broker.list_positions()
    account = await broker.get_account()

    assert positions[0].symbol == "LMT"
    assert positions[0].quantity == Decimal("10")
    assert account.cash == Decimal("55000.00")
    assert account.equity == Decimal("100000.00")
    # buying_power backs the shared capital pool's headline (margin => can exceed cash).
    assert account.buying_power == Decimal("180000.00")


async def test_account_buying_power_falls_back_to_cash_when_absent() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"cash": "55000.00", "equity": "100000.00"})

    account = await _broker(handler).get_account()

    assert account.buying_power == Decimal("55000.00")


async def test_cancel_open_order_deletes_then_refetches() -> None:
    calls: list[str] = []
    open_order = {
        "id": "b9",
        "client_order_id": "ord-9",
        "symbol": "LMT",
        "qty": "10",
        "filled_qty": "0",
        "filled_avg_price": None,
        "side": "buy",
        "type": "market",
        "status": "accepted",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "DELETE":
            return httpx.Response(204)
        # First lookup -> open; after the DELETE -> canceled.
        status = "canceled" if any(c.startswith("DELETE") for c in calls) else "accepted"
        return httpx.Response(200, json={**open_order, "status": status})

    order = await _broker(handler).cancel_order(client_order_id="ord-9")

    assert order.status is OrderStatus.CANCELED
    assert "DELETE /v2/orders/b9" in calls  # canceled by broker order id


async def test_cancel_terminal_order_is_noop() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(200, json=_FILLED_ORDER)  # already filled

    order = await _broker(handler).cancel_order(client_order_id="ord-1")

    assert order.status is OrderStatus.FILLED
    assert "DELETE" not in calls  # nothing to cancel


async def test_cancel_missing_order_raises_not_found() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "order not found"})

    with pytest.raises(OrderNotFoundError):
        await _broker(handler).cancel_order(client_order_id="ghost")


async def test_pending_order_has_no_fills() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "b2",
                "client_order_id": "ord-2",
                "symbol": "LMT",
                "qty": "10",
                "filled_qty": "0",
                "filled_avg_price": None,
                "side": "buy",
                "type": "market",
                "status": "accepted",
            },
        )

    order = await _broker(handler).submit_order(
        client_order_id="ord-2", symbol="LMT", side=OrderSide.BUY, quantity=Decimal("10")
    )
    assert order.status is OrderStatus.ACKNOWLEDGED
    assert order.fills == ()
    assert order.order_type is OrderType.MARKET


def _order_json(index: int, submitted_at: str) -> dict[str, object]:
    return {
        "id": f"broker-{index}",
        "client_order_id": f"ord-{index}",
        "symbol": "LMT",
        "qty": "1",
        "filled_qty": "0",
        "side": "buy",
        "type": "market",
        "status": "new",
        "submitted_at": submitted_at,
    }


async def test_list_orders_paginates_past_the_500_order_page() -> None:
    """F18: history beyond one 500-order page is fully paginated with the `after` cursor.

    A single page would truncate at 500, so every older order would read back as broker='missing'
    drift and brick the readiness reconciliation check. The adapter must return ALL 650 orders.
    """
    total = 650
    all_orders = [
        _order_json(i, f"2026-06-13T14:{i // 60:02d}:{i % 60:02d}Z") for i in range(total)
    ]
    seen_after: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/orders"
        after = request.url.params.get("after")
        seen_after.append(after)
        # Serve orders strictly after the cursor, up to the 500 page limit, ascending.
        start = (
            0
            if after is None
            else next(i for i, o in enumerate(all_orders) if o["submitted_at"] == after) + 1
        )
        page = all_orders[start : start + 500]
        return httpx.Response(200, json=page)

    orders = await _broker(handler).list_orders()

    assert len(orders) == total  # nothing truncated
    assert {o.client_order_id for o in orders} == {f"ord-{i}" for i in range(total)}
    # Paged: first request has no cursor, later ones advance it.
    assert seen_after[0] is None
    assert len(seen_after) >= 2
