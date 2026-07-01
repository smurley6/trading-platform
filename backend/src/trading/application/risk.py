"""Risk gate: the hard, non-bypassable boundary between strategy intent and execution.

No order reaches a broker without passing `RiskGate.evaluate` (FR-RISK-1).
"""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from trading.domain.decisions import RiskVerdict
from trading.domain.orders import OrderSide

_QTY_STEP = Decimal("0.0001")  # fractional shares, floored


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """All percentages are of sleeve equity, expressed 0-100. Required at startup."""

    max_trade_pct_of_sleeve: Decimal
    max_position_concentration_pct: Decimal
    max_drawdown_pct: Decimal


class KillSwitch:
    """Global halt for all new orders across all sleeves (FR-RISK-3)."""

    def __init__(self) -> None:
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self) -> None:
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False


@dataclass(frozen=True, slots=True)
class RiskContext:
    sleeve_equity: Decimal  # denominator for the % caps; the shared pool equity for a live account
    peak_equity: Decimal
    current_position_value: Decimal  # market value already held in this symbol (the shared book)
    price: Decimal
    # Spendable buying power remaining in the shared pool. 0 => guard disabled (unknown/simulator),
    # so pre-existing callers and the paper simulator are unaffected. When >0 (a live account) a
    # BUY is capped so combined orders can't over-commit the one shared account.
    remaining_buying_power: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    verdict: RiskVerdict
    reason: str
    approved_quantity: Decimal


def _floor_quantity(value: Decimal) -> Decimal:
    return value.quantize(_QTY_STEP, rounding=ROUND_DOWN)


class RiskGate:
    def __init__(self, limits: RiskLimits, kill_switch: KillSwitch) -> None:
        self._limits = limits
        self._kill_switch = kill_switch

    def evaluate(self, side: OrderSide, quantity: Decimal, context: RiskContext) -> RiskAssessment:
        if self._kill_switch.engaged:
            return self._reject("kill switch engaged")
        if context.price <= 0:
            return self._reject(f"no usable price ({context.price})")
        if context.sleeve_equity <= 0:
            return self._reject("sleeve equity is zero or negative")

        if context.peak_equity > 0:
            drawdown_pct = (context.peak_equity - context.sleeve_equity) / context.peak_equity * 100
            if drawdown_pct >= self._limits.max_drawdown_pct:
                return self._reject(
                    f"max drawdown breached: {drawdown_pct:.2f}% >= "
                    f"{self._limits.max_drawdown_pct}% — circuit breaker open"
                )

        approved = quantity
        reasons: list[str] = []

        max_trade_notional = context.sleeve_equity * self._limits.max_trade_pct_of_sleeve / 100
        if approved * context.price > max_trade_notional:
            approved = _floor_quantity(max_trade_notional / context.price)
            reasons.append(
                f"trade capped at {self._limits.max_trade_pct_of_sleeve}% of sleeve equity"
            )

        if side is OrderSide.BUY:
            concentration_cap = (
                context.sleeve_equity * self._limits.max_position_concentration_pct / 100
            )
            headroom = concentration_cap - context.current_position_value
            if headroom <= 0:
                return self._reject(
                    f"position concentration cap "
                    f"({self._limits.max_position_concentration_pct}%) already reached"
                )
            if approved * context.price > headroom:
                approved = _floor_quantity(headroom / context.price)
                reasons.append(
                    f"buy capped by {self._limits.max_position_concentration_pct}% "
                    "concentration limit"
                )
            # Shared-pool guard: never let a BUY (alone or combined with earlier BUYs in the same
            # cycle, once the caller decrements this) exceed the spendable buying power of the one
            # shared account. Skipped when unknown/zero (simulator/back-compat).
            if (
                context.remaining_buying_power > 0
                and approved * context.price > context.remaining_buying_power
            ):
                approved = _floor_quantity(context.remaining_buying_power / context.price)
                reasons.append("buy capped by remaining pool buying power")

        if approved <= 0:
            return self._reject("risk caps reduced quantity to zero")
        if approved == quantity:
            return RiskAssessment(RiskVerdict.APPROVED, "within limits", approved)
        return RiskAssessment(RiskVerdict.MODIFIED, "; ".join(reasons), approved)

    @staticmethod
    def _reject(reason: str) -> RiskAssessment:
        return RiskAssessment(RiskVerdict.REJECTED, reason, Decimal("0"))
