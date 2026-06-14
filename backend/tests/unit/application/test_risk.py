from decimal import Decimal

from trading.application.risk import KillSwitch, RiskContext, RiskGate, RiskLimits
from trading.domain.decisions import RiskVerdict
from trading.domain.orders import OrderSide

LIMITS = RiskLimits(
    max_trade_pct_of_sleeve=Decimal("10"),
    max_position_concentration_pct=Decimal("25"),
    max_drawdown_pct=Decimal("20"),
)


def _context(
    equity: str = "10000",
    peak: str = "10000",
    position_value: str = "0",
    price: str = "100",
) -> RiskContext:
    return RiskContext(
        sleeve_equity=Decimal(equity),
        peak_equity=Decimal(peak),
        current_position_value=Decimal(position_value),
        price=Decimal(price),
    )


def _gate(kill_switch: KillSwitch | None = None) -> RiskGate:
    return RiskGate(LIMITS, kill_switch or KillSwitch())


def test_within_limits_is_approved() -> None:
    assessment = _gate().evaluate(OrderSide.BUY, Decimal("5"), _context())
    assert assessment.verdict is RiskVerdict.APPROVED
    assert assessment.approved_quantity == Decimal("5")


def test_kill_switch_rejects_everything() -> None:
    kill_switch = KillSwitch()
    kill_switch.engage()
    assessment = _gate(kill_switch).evaluate(OrderSide.BUY, Decimal("1"), _context())
    assert assessment.verdict is RiskVerdict.REJECTED
    assert "kill switch" in assessment.reason


def test_per_trade_cap_modifies_quantity() -> None:
    # 50 shares @ 100 = 5000 notional; cap is 10% of 10000 = 1000 -> 10 shares.
    assessment = _gate().evaluate(OrderSide.BUY, Decimal("50"), _context())
    assert assessment.verdict is RiskVerdict.MODIFIED
    assert assessment.approved_quantity == Decimal("10")


def test_concentration_cap_blocks_buys_at_limit() -> None:
    # Already holding 25% of sleeve equity in this symbol.
    assessment = _gate().evaluate(OrderSide.BUY, Decimal("1"), _context(position_value="2500"))
    assert assessment.verdict is RiskVerdict.REJECTED
    assert "concentration" in assessment.reason


def test_concentration_cap_trims_buys_near_limit() -> None:
    # Headroom is 2500 - 2400 = 100 -> 1 share at price 100.
    assessment = _gate().evaluate(OrderSide.BUY, Decimal("5"), _context(position_value="2400"))
    assert assessment.verdict is RiskVerdict.MODIFIED
    assert assessment.approved_quantity == Decimal("1")


def test_sells_are_not_concentration_limited() -> None:
    assessment = _gate().evaluate(OrderSide.SELL, Decimal("5"), _context(position_value="2500"))
    assert assessment.verdict is RiskVerdict.APPROVED


def test_drawdown_circuit_breaker() -> None:
    # Equity 7500 vs peak 10000 = 25% drawdown >= 20% limit.
    assessment = _gate().evaluate(
        OrderSide.BUY, Decimal("1"), _context(equity="7500", peak="10000")
    )
    assert assessment.verdict is RiskVerdict.REJECTED
    assert "drawdown" in assessment.reason


def test_unusable_price_rejects() -> None:
    assessment = _gate().evaluate(OrderSide.BUY, Decimal("1"), _context(price="0"))
    assert assessment.verdict is RiskVerdict.REJECTED
