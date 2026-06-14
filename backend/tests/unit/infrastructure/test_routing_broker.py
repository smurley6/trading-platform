from decimal import Decimal

import pytest

from tests.conftest import FixedClock
from trading.domain.sleeves import TradingMode
from trading.infrastructure.brokers.paper import PaperBroker
from trading.infrastructure.brokers.routing import (
    LiveBrokerUnavailableError,
    ModeRoutingBroker,
)
from trading.infrastructure.market_data.static import StaticMarketData


def _paper(cash: str, clock: FixedClock) -> PaperBroker:
    md = StaticMarketData({"AAPL": Decimal("100")}, clock)
    return PaperBroker(market_data=md, clock=clock, starting_cash=Decimal(cash))


async def test_routes_to_paper_in_paper_mode(clock: FixedClock) -> None:
    mode = {"v": TradingMode.PAPER}
    router = ModeRoutingBroker(
        paper=_paper("10000", clock), live=_paper("99999", clock), mode=lambda: mode["v"]
    )
    assert (await router.get_account()).cash == Decimal("10000")


async def test_routes_to_live_in_live_mode(clock: FixedClock) -> None:
    mode = {"v": TradingMode.LIVE}
    router = ModeRoutingBroker(
        paper=_paper("10000", clock), live=_paper("99999", clock), mode=lambda: mode["v"]
    )
    assert (await router.get_account()).cash == Decimal("99999")


async def test_live_mode_without_live_broker_refuses(clock: FixedClock) -> None:
    router = ModeRoutingBroker(
        paper=_paper("10000", clock), live=None, mode=lambda: TradingMode.LIVE
    )
    with pytest.raises(LiveBrokerUnavailableError):
        await router.get_account()
