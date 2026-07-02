"""Projections: current-state views derived from the event log.

The event log is the system of record; these are rebuildable at any time (spec section 9).
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from trading.domain.events import EventType, StoredEvent
from trading.domain.orders import Fill, OrderSide
from trading.domain.positions import HeldPosition, Portfolio
from trading.domain.provenance import PitConfidence, Provenance
from trading.domain.signals import Signal, SignalSource


def _signal_from_payload(payload: dict[str, Any]) -> Signal:
    return Signal(
        id=str(payload["signal_id"]),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        source_type=SignalSource(str(payload["source_type"])),
        ticker=str(payload["ticker"]),
        politician_id=(None if payload["politician_id"] is None else str(payload["politician_id"])),
        sentiment=float(payload["sentiment"]),
        politician_weight=float(payload["politician_weight"]),
        source_reliability=float(payload["source_reliability"]),
        raw_payload=dict(payload.get("raw_payload", {})),
        # Legacy events (recorded before provenance existed) default to SYNTHETIC — fail-closed,
        # so old data can never be mistaken for real and slip past the trade-time guard.
        provenance=Provenance(str(payload.get("provenance", Provenance.SYNTHETIC.value))),
        # Legacy events (recorded before pit_confidence existed) default to "exact" so pre-backfill
        # REAL/SYNTHETIC signals — all live-timestamped — are not disturbed.
        pit_confidence=cast(PitConfidence, str(payload.get("pit_confidence", "exact"))),
    )


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
        signal = _signal_from_payload(stored.event.payload)
        if cutoff is not None and signal.created_at < cutoff:
            continue
        signals.append(signal)
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


def held_positions_from_events(events: Sequence[StoredEvent]) -> list[HeldPosition]:
    """Open positions with cost basis, derived from recorded fills, grouped per sleeve.

    Replays each sleeve's fills through the same `Portfolio.apply_fill` math used live, so
    avg cost basis is consistent with the rest of the system. Fully-closed positions drop
    out (the portfolio removes them at zero quantity).
    """
    by_sleeve: dict[str, Portfolio] = {}
    for stored in events:
        if stored.event.event_type != EventType.ORDER_FILLED:
            continue
        payload = stored.event.payload
        sleeve_id = str(payload.get("sleeve_id", ""))
        fill = Fill(
            client_order_id=str(payload.get("client_order_id", "")),
            symbol=str(payload["symbol"]),
            side=OrderSide(str(payload["side"])),
            price=Decimal(str(payload["price"])),
            quantity=Decimal(str(payload["quantity"])),
            fees=Decimal(str(payload.get("fees", "0"))),
            filled_at=datetime.fromisoformat(str(payload["filled_at"])),
        )
        portfolio = by_sleeve.get(sleeve_id, Portfolio(cash=Decimal("0")))
        by_sleeve[sleeve_id] = portfolio.apply_fill(fill)

    held: list[HeldPosition] = []
    for sleeve_id, portfolio in by_sleeve.items():
        for position in portfolio.positions.values():
            held.append(
                HeldPosition(
                    symbol=position.symbol,
                    sleeve_id=sleeve_id,
                    quantity=position.quantity,
                    avg_price=position.avg_entry_price,
                )
            )
    return held


def order_sleeve(events: Sequence[StoredEvent], client_order_id: str) -> str | None:
    """The sleeve that owns an order, from its recorded lifecycle events (None if unknown)."""
    for stored in events:
        payload = stored.event.payload
        if payload.get("client_order_id") == client_order_id and payload.get("sleeve_id"):
            return str(payload["sleeve_id"])
    return None


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
