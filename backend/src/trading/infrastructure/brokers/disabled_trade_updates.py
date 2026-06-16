"""Null TradeUpdatesStreamPort used when the real-time trade-updates stream is not configured.

Default-off by construction: a bare ``python -m trading.main`` injects this so the app opens
no account socket and makes no network call. ``enabled`` is False and ``run`` returns
immediately without dispatching anything.
"""

from trading.application.ports.trade_updates import TradeUpdateHandler


class DisabledTradeUpdatesStream:
    @property
    def enabled(self) -> bool:
        return False

    async def run(self, handler: TradeUpdateHandler) -> None:
        return None
