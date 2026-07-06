"""Read-only risk pre-check use case (UI Phase C §5.2).

Exercises PrecheckRisk with an in-memory FakeBroker + static market data + a real
RiskGate/RiskLimits/KillSwitch, mirroring test_risk.py's caps and test_proposals.py's fakes.
The pre-check re-derives each named gate's arithmetic, so a lockstep test asserts its overall
verdict/approved_quantity equals RiskGate.evaluate(...) on the same context (drift guard).
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from trading.application.ports.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
)
from trading.application.ports.market_data import Quote
from trading.application.risk import KillSwitch, RiskContext, RiskGate, RiskLimits
from trading.application.risk_precheck import PrecheckRisk
from trading.domain.decisions import RiskVerdict
from trading.domain.orders import OrderSide, OrderType

_AS_OF = datetime(2026, 6, 13, 20, 0, tzinfo=UTC)

LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("10"),
    max_position_concentration_pct=Decimal("25"),
    max_drawdown_pct=Decimal("20"),
)


class FakeBroker:
    """Minimal BrokerPort: only get_account + list_positions are read by the pre-check."""

    def __init__(
        self,
        *,
        cash: Decimal = Decimal("10000"),
        equity: Decimal = Decimal("10000"),
        buying_power: Decimal = Decimal("0"),
        positions: Sequence[BrokerPosition] = (),
    ) -> None:
        self._account = BrokerAccount(cash=cash, equity=equity, buying_power=buying_power)
        self._positions = list(positions)

    async def submit_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
    ) -> BrokerOrder:
        raise NotImplementedError

    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        raise NotImplementedError

    async def list_orders(self) -> Sequence[BrokerOrder]:
        return []

    async def list_positions(self) -> Sequence[BrokerPosition]:
        return self._positions

    async def get_account(self) -> BrokerAccount:
        return self._account


class StaticQuote:
    def __init__(self, last: Decimal) -> None:
        self._last = last

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, bid=self._last, ask=self._last, last=self._last, as_of=_AS_OF)


def _checker(
    broker: FakeBroker,
    *,
    price: str = "100",
    kill_switch: KillSwitch | None = None,
) -> tuple[PrecheckRisk, RiskGate, KillSwitch]:
    ks = kill_switch or KillSwitch()
    gate = RiskGate(LIMITS, ks)
    checker = PrecheckRisk(
        risk_gate=gate,
        risk_limits=LIMITS,
        broker=broker,
        market_data=StaticQuote(Decimal(price)),
        kill_switch=ks,
    )
    return checker, gate, ks


def _by_name(result: object) -> dict[str, object]:
    return {c.name: c for c in result.checks}  # type: ignore[attr-defined]


async def test_clean_buy_within_all_caps_approves() -> None:
    checker, _gate, _ks = _checker(FakeBroker())
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("5"), symbol="LMT")

    assert result.verdict == RiskVerdict.APPROVED.value
    assert result.approved_quantity == Decimal("5")
    assert result.symbol == "LMT"
    assert result.side == "buy"
    checks = _by_name(result)
    # BUY => concentration + BP present; every check passes on a clean order.
    assert set(checks) == {
        "kill_switch",
        "price",
        "sleeve_equity",
        "drawdown",
        "trade_size",
        "concentration",
        "buying_power",
    }
    assert all(c.passed for c in result.checks)
    # limit/current/projected populated for the sizing checks.
    assert checks["trade_size"].limit is not None  # type: ignore[attr-defined]
    assert checks["trade_size"].projected == "$500.00"  # type: ignore[attr-defined]
    assert checks["concentration"].current == "$0.00"  # type: ignore[attr-defined]


async def test_trade_size_cap_binds_modifies() -> None:
    # 50 @ 100 = 5000 notional; cap 10% of 10000 = 1000 -> 10 shares.
    checker, _gate, _ks = _checker(FakeBroker())
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("50"), symbol="LMT")

    assert result.verdict == RiskVerdict.MODIFIED.value
    assert result.approved_quantity == Decimal("10")
    assert _by_name(result)["trade_size"].passed is False  # type: ignore[attr-defined]


async def test_concentration_headroom_exhausted_rejects() -> None:
    # Already holding 25% of equity in the symbol (25 @ 100 = 2500 = concentration cap).
    broker = FakeBroker(
        positions=[
            BrokerPosition(symbol="LMT", quantity=Decimal("25"), avg_entry_price=Decimal("100"))
        ]
    )
    checker, _gate, _ks = _checker(broker)
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("1"), symbol="LMT")

    assert result.verdict == RiskVerdict.REJECTED.value
    assert result.approved_quantity == Decimal("0")
    conc = _by_name(result)["concentration"]
    assert conc.passed is False  # type: ignore[attr-defined]
    assert "concentration" in conc.reason  # type: ignore[attr-defined]


async def test_buying_power_binds() -> None:
    # remaining_buying_power 300 < requested notional (5 @ 100 = 500) but within trade + conc caps.
    broker = FakeBroker(buying_power=Decimal("300"))
    checker, _gate, _ks = _checker(broker)
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("5"), symbol="LMT")

    assert result.verdict == RiskVerdict.MODIFIED.value
    assert result.approved_quantity == Decimal("3")  # 300 / 100
    assert _by_name(result)["buying_power"].passed is False  # type: ignore[attr-defined]


async def test_kill_switch_engaged_rejects() -> None:
    ks = KillSwitch()
    ks.engage()
    checker, _gate, _ks = _checker(FakeBroker(), kill_switch=ks)
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("1"), symbol="LMT")

    assert result.verdict == RiskVerdict.REJECTED.value
    ks_check = _by_name(result)["kill_switch"]
    assert ks_check.passed is False  # type: ignore[attr-defined]


async def test_sell_omits_concentration_and_buying_power() -> None:
    checker, _gate, _ks = _checker(FakeBroker())
    result = await checker.execute(side=OrderSide.SELL, quantity=Decimal("5"), symbol="LMT")

    assert result.verdict == RiskVerdict.APPROVED.value
    names = set(_by_name(result))
    assert "concentration" not in names
    assert "buying_power" not in names
    assert {"kill_switch", "price", "sleeve_equity", "drawdown", "trade_size"} <= names


async def test_unusable_price_fails_price_check() -> None:
    checker, _gate, _ks = _checker(FakeBroker(), price="0")
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("1"), symbol="LMT")

    assert result.verdict == RiskVerdict.REJECTED.value
    assert _by_name(result)["price"].passed is False  # type: ignore[attr-defined]


async def test_zero_equity_fails_sleeve_equity_check() -> None:
    checker, _gate, _ks = _checker(FakeBroker(equity=Decimal("0")))
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("1"), symbol="LMT")

    assert result.verdict == RiskVerdict.REJECTED.value
    assert _by_name(result)["sleeve_equity"].passed is False  # type: ignore[attr-defined]


async def test_drawdown_check_reports_zero_in_precheck() -> None:
    # peak_equity == equity in the pre-check => drawdown always 0% (conservative simplification),
    # so the check passes even though the live cycle could see a real breach.
    checker, _gate, _ks = _checker(FakeBroker(equity=Decimal("7500")))
    result = await checker.execute(side=OrderSide.BUY, quantity=Decimal("1"), symbol="LMT")

    drawdown = _by_name(result)["drawdown"]
    assert drawdown.passed is True  # type: ignore[attr-defined]
    assert drawdown.current == "0.00%"  # type: ignore[attr-defined]


async def test_precheck_matches_risk_gate_lockstep() -> None:
    # Drift guard: the re-derived overall verdict/approved_quantity MUST equal RiskGate.evaluate
    # on the exact same context, for a range of order sizes/sides.
    for side, qty, bp, pos in [
        (OrderSide.BUY, Decimal("5"), Decimal("0"), Decimal("0")),
        (OrderSide.BUY, Decimal("50"), Decimal("0"), Decimal("0")),
        (OrderSide.BUY, Decimal("5"), Decimal("300"), Decimal("0")),
        (OrderSide.BUY, Decimal("5"), Decimal("0"), Decimal("2400")),
        (OrderSide.SELL, Decimal("5"), Decimal("0"), Decimal("0")),
    ]:
        current_qty = pos / Decimal("100")
        broker = FakeBroker(
            buying_power=bp,
            positions=(
                [BrokerPosition(symbol="LMT", quantity=current_qty, avg_entry_price=Decimal("100"))]
                if current_qty > 0
                else []
            ),
        )
        checker, gate, _ks = _checker(broker)
        result = await checker.execute(side=side, quantity=qty, symbol="LMT")
        expected = gate.evaluate(
            side,
            qty,
            RiskContext(
                sleeve_equity=Decimal("10000"),
                peak_equity=Decimal("10000"),
                current_position_value=pos,
                price=Decimal("100"),
                remaining_buying_power=bp,
            ),
        )
        assert result.verdict == expected.verdict.value, (side, qty, bp, pos)
        assert result.approved_quantity == expected.approved_quantity, (side, qty, bp, pos)
