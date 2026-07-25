"""Alpaca broker adapter (FR-EXE, spec section 11): BrokerPort over the Alpaca Trading API v2.

Paper and live differ ONLY by base URL and credentials — no strategy or domain code changes
when switching (spec section 11.5). Implemented against the documented REST API with the
httpx client already in the stack, which keeps the adapter fully unit-testable with a mock
transport; the `BrokerPort` seam means a swap to `alpaca-py` later changes nothing upstream.

Submission is idempotent on `client_order_id` (FR-EXE-3): a duplicate is resolved by fetching
the existing order rather than creating a second one.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderNotFoundError,
)
from trading.domain.orders import TERMINAL_STATUSES, Fill, OrderSide, OrderStatus, OrderType

# Alpaca order states -> our lifecycle. Unknown/transient states map to SUBMITTED.
_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.ACKNOWLEDGED,
    "accepted_for_bidding": OrderStatus.ACKNOWLEDGED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "done_for_day": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "stopped": OrderStatus.REJECTED,
}


# Alpaca caps one /v2/orders page at 500. Local order history grows without bound, so a single
# 500-order page eventually stops covering it and every older order reads back as broker='missing'
# drift — false drift that buries real drift and bricks the readiness `reconciliation_clean` check
# (review F18). We page through the FULL history with the `after` cursor instead.
_ORDER_PAGE_LIMIT = 500
_MAX_ORDER_PAGES = 200  # safety bound: 100k orders — far beyond any personal-account lifetime


class AlpacaBrokerError(Exception):
    pass


class AlpacaBroker:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        key_id: str,
        secret_key: str,
        base_url: str,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        response = await self._client.post(
            f"{self._base_url}/v2/orders",
            headers=self._headers,
            json={
                "symbol": symbol,
                "qty": str(quantity),
                "side": side.value,
                "type": order_type.value,
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )
        if response.status_code == 422 and "client_order_id" in response.text:
            # Duplicate submission — return the order that already exists (idempotency).
            return await self._get_by_client_order_id(client_order_id)
        if response.status_code >= 400:
            raise AlpacaBrokerError(
                f"order submit failed ({response.status_code}): {response.text}"
            )
        return _to_order(response.json())

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        order = await self._lookup(client_order_id)
        if order.status in TERMINAL_STATUSES:
            return order  # already final — nothing to cancel
        response = await self._client.delete(
            f"{self._base_url}/v2/orders/{order.broker_order_id}", headers=self._headers
        )
        # 204 cancel accepted; 422 not cancelable (e.g. it just filled). Either way, re-fetch
        # the authoritative state rather than assume the outcome.
        if response.status_code not in (200, 204, 422):
            raise AlpacaBrokerError(f"cancel failed ({response.status_code}): {response.text}")
        return await self._lookup(client_order_id)

    async def _lookup(self, client_order_id: str) -> BrokerOrder:
        try:
            return await self._get_by_client_order_id(client_order_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OrderNotFoundError(
                    f"no order with client_order_id {client_order_id}"
                ) from exc
            raise

    async def list_orders(self) -> Sequence[BrokerOrder]:
        """Every order in the account's history, paged past the 500-order limit (review F18).

        Walks pages oldest-first with the `after` (submitted_at, exclusive) cursor, advancing by
        the last order's raw timestamp until a short page ends the history. Deduplicates by
        client_order_id (belt-and-suspenders against a boundary overlap) and bounds the walk so a
        pathological cursor can never loop forever.
        """
        orders: list[BrokerOrder] = []
        seen: set[str] = set()
        after: str | None = None
        for _ in range(_MAX_ORDER_PAGES):
            params: dict[str, Any] = {
                "status": "all",
                "limit": _ORDER_PAGE_LIMIT,
                "direction": "asc",
            }
            if after is not None:
                params["after"] = after
            response = await self._client.get(
                f"{self._base_url}/v2/orders", headers=self._headers, params=params
            )
            response.raise_for_status()
            items = response.json()
            for item in items:
                order = _to_order(item)
                if order.client_order_id in seen:
                    continue
                seen.add(order.client_order_id)
                orders.append(order)
            if len(items) < _ORDER_PAGE_LIMIT:
                break
            # Advance the cursor by the last order's RAW submitted_at (never the reparsed value,
            # which can lose sub-microsecond precision). A cursor that fails to advance ends the
            # walk rather than spinning.
            last_raw = items[-1].get("submitted_at") or items[-1].get("created_at")
            if not last_raw or last_raw == after:
                break
            after = str(last_raw)
        return orders

    async def list_positions(self) -> Sequence[BrokerPosition]:
        response = await self._client.get(f"{self._base_url}/v2/positions", headers=self._headers)
        response.raise_for_status()
        return [
            BrokerPosition(
                symbol=str(item["symbol"]),
                quantity=Decimal(str(item["qty"])),
                avg_entry_price=Decimal(str(item["avg_entry_price"])),
            )
            for item in response.json()
        ]

    async def get_account(self) -> BrokerAccount:
        response = await self._client.get(f"{self._base_url}/v2/account", headers=self._headers)
        response.raise_for_status()
        data = response.json()
        # `buying_power` is the shared capital pool's spendable headline (Alpaca paper accounts are
        # margin-enabled, so it typically exceeds cash). Fall back to cash if the field is absent.
        return BrokerAccount(
            cash=Decimal(str(data["cash"])),
            equity=Decimal(str(data["equity"])),
            buying_power=Decimal(str(data.get("buying_power", data["cash"]))),
        )

    async def _get_by_client_order_id(self, client_order_id: str) -> BrokerOrder:
        response = await self._client.get(
            f"{self._base_url}/v2/orders:by_client_order_id",
            headers=self._headers,
            params={"client_order_id": client_order_id},
        )
        response.raise_for_status()
        return _to_order(response.json())


def _to_order(data: dict[str, Any]) -> BrokerOrder:
    status = _STATUS_MAP.get(str(data.get("status", "")), OrderStatus.SUBMITTED)
    side = OrderSide(str(data["side"]))
    filled_qty = Decimal(str(data.get("filled_qty") or "0"))
    fills: tuple[Fill, ...] = ()
    if filled_qty > 0 and data.get("filled_avg_price"):
        fills = (
            Fill(
                client_order_id=str(data["client_order_id"]),
                symbol=str(data["symbol"]),
                side=side,
                price=Decimal(str(data["filled_avg_price"])),
                quantity=filled_qty,
                fees=Decimal("0"),  # Alpaca equities are commission-free
                filled_at=_parse_time(data.get("filled_at") or data.get("updated_at")),
            ),
        )
    submitted_raw = data.get("submitted_at") or data.get("created_at")
    return BrokerOrder(
        client_order_id=str(data["client_order_id"]),
        broker_order_id=str(data["id"]),
        symbol=str(data["symbol"]),
        side=side,
        quantity=Decimal(str(data["qty"])),
        order_type=OrderType(str(data.get("type", "market"))),
        status=status,
        fills=fills,
        submitted_at=_parse_time(submitted_raw) if submitted_raw else None,
    )


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
