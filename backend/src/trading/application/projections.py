"""Projections: current-state views derived from the event log.

The event log is the system of record; these are rebuildable at any time (spec section 9).
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from trading.domain.events import EventType, StoredEvent
from trading.domain.orders import OrderSide
from trading.domain.signals import Signal, SignalSource


def signals_from_events(
    events: Sequence[StoredEvent],
    *,
    now: datetime | None = None,
    window_days: int | None = None,
) -> list[Signal]:
    """Reconstruct Signals from recorded `signal.received` events.

    When `now` and `window_days` are given, only signals created within the rolling
    corroboration window are returned (FR-POL-4).
    """
    cutoff = None if now is None or window_days is None else now - timedelta(days=window_days)
    signals: list[Signal] = []
    for stored in events:
        if stored.event.event_type != EventType.SIGNAL_RECEIVED:
            continue
        payload = stored.event.payload
        created_at = datetime.fromisoformat(str(payload["created_at"]))
        if cutoff is not None and created_at < cutoff:
            continue
        signals.append(
            Signal(
                id=str(payload["signal_id"]),
                created_at=created_at,
                source_type=SignalSource(str(payload["source_type"])),
                ticker=str(payload["ticker"]),
                politician_id=(
                    None if payload["politician_id"] is None else str(payload["politician_id"])
                ),
                sentiment=float(payload["sentiment"]),
                politician_weight=float(payload["politician_weight"]),
                source_reliability=float(payload["source_reliability"]),
                raw_payload=dict(payload.get("raw_payload", {})),
            )
        )
    return signals


def _matches_sleeve(payload: dict[str, object], sleeve_id: str | None) -> bool:
    return sleeve_id is None or payload.get("sleeve_id") == sleeve_id


def positions_from_events(
    events: Sequence[StoredEvent], *, sleeve_id: str | None = None
) -> dict[str, Decimal]:
    """Net quantity per symbol implied by recorded fills, optionally scoped to one sleeve."""
    quantities: dict[str, Decimal] = {}
    for stored in events:
        if stored.event.event_type != EventType.ORDER_FILLED:
            continue
        payload = stored.event.payload
        if not _matches_sleeve(payload, sleeve_id):
            continue
        symbol = str(payload["symbol"])
        quantity = Decimal(str(payload["quantity"]))
        if payload["side"] == OrderSide.SELL.value:
            quantity = -quantity
        quantities[symbol] = quantities.get(symbol, Decimal("0")) + quantity
    return {symbol: qty for symbol, qty in quantities.items() if qty != 0}


def order_statuses_from_events(
    events: Sequence[StoredEvent], *, sleeve_id: str | None = None
) -> dict[str, str]:
    """Latest known local status per client_order_id, optionally scoped to one sleeve."""
    statuses: dict[str, str] = {}
    for stored in events:
        event_type = stored.event.event_type
        payload = stored.event.payload
        if not _matches_sleeve(payload, sleeve_id):
            continue
        if event_type == EventType.ORDER_INTENT:
            statuses[str(payload["client_order_id"])] = "intent"
        elif event_type == EventType.ORDER_SUBMITTED:
            statuses[str(payload["client_order_id"])] = "submitted"
        elif event_type == EventType.ORDER_SUBMIT_FAILED:
            statuses[str(payload["client_order_id"])] = "submit_failed"
        elif event_type == EventType.ORDER_STATUS_CHANGED:
            statuses[str(payload["client_order_id"])] = str(payload["status"])
    return statuses
