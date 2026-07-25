"""Read-only risk pre-check use case (UI Phase C, §5.2).

Runs the SAME risk evaluation the live cycle (`EvaluateSignals`) would run for a single
prospective order, so the operator sees the verdict *before* approving in the Review screen —
never discovered later in the blotter. It is strictly read-only: it reconstructs the
`RiskContext` from live broker/market-data reads, calls `RiskGate.evaluate` for the overall
verdict, and writes NO events and mutates NO state.

`RiskGate.evaluate` returns only verdict + reason + approved_quantity — it emits no per-check
detail. §5.2's evidence pane needs a per-named-check breakdown ("RISK PRE-CHECK ✓ size ✓
concentration ✓ BP"), so this use case RE-DERIVES each named gate from the same
`RiskLimits` + `RiskContext` arithmetic as `application.risk.RiskGate.evaluate` (risk.py). A
shared unit test asserts the re-derived overall verdict/approved_quantity equals
`risk_gate.evaluate(...)` on the same context, so any drift from that arithmetic is caught.

peak_equity simplification: the live drawdown circuit-breaker uses `EvaluateSignals`' private
in-memory running peak equity, which a stateless read-only pre-check cannot reach. This
use case passes `peak_equity = account.equity`, making the drawdown check always report 0%
(never the binding constraint for a fresh order). This is an intentional, conservative
fidelity gap: the pre-check can show "approved" for a size the real cycle would reject if a
drawdown breach is currently active. The drawdown check's reason flags this.
"""

from dataclasses import dataclass
from decimal import Decimal

from trading.application.ports.broker import BrokerPort
from trading.application.ports.market_data import MarketDataPort
from trading.application.risk import KillSwitch, RiskContext, RiskGate, RiskLimits
from trading.domain.orders import OrderSide


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    """One named risk gate's outcome, domain-free (mapped to a DTO in the interface layer)."""

    name: str
    label: str
    passed: bool
    limit: str | None
    current: str | None
    projected: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class RiskPrecheckResult:
    symbol: str
    side: str
    requested_quantity: Decimal
    approved_quantity: Decimal
    verdict: str
    reason: str
    checks: tuple[RiskCheckResult, ...]


def _money(value: Decimal) -> str:
    return f"${value:,.2f}"


def _pct(value: Decimal) -> str:
    return f"{value}%"


class PrecheckRisk:
    """Read-only single-order risk pre-check. Receives ports only; writes nothing."""

    def __init__(
        self,
        *,
        risk_gate: RiskGate,
        risk_limits: RiskLimits,
        broker: BrokerPort,
        market_data: MarketDataPort,
        kill_switch: KillSwitch,
    ) -> None:
        self._risk_gate = risk_gate
        self._risk_limits = risk_limits
        self._broker = broker
        self._market_data = market_data
        self._kill_switch = kill_switch

    async def execute(
        self, *, side: OrderSide, quantity: Decimal, symbol: str
    ) -> RiskPrecheckResult:
        account = await self._broker.get_account()
        current_qty = Decimal("0")
        for position in await self._broker.list_positions():
            if position.symbol == symbol:
                current_qty = position.quantity
                break
        quote = await self._market_data.get_quote(symbol)
        price = quote.last
        current_position_value = current_qty * price

        # Reconstruct the SAME context EvaluateSignals builds (evaluate_signals.py L151-160),
        # except peak_equity = equity (see module docstring: read-only pre-check cannot reach
        # the live running peak, so drawdown reports 0% — a conservative simplification).
        context = RiskContext(
            sleeve_equity=account.equity,
            peak_equity=account.equity,
            current_position_value=current_position_value,
            price=price,
            remaining_buying_power=account.buying_power,
        )

        # Authoritative overall verdict/approved_quantity/reason.
        assessment = self._risk_gate.evaluate(side, quantity, context)

        checks = self._derive_checks(side=side, quantity=quantity, context=context)

        return RiskPrecheckResult(
            symbol=symbol,
            side=side.value,
            requested_quantity=quantity,
            approved_quantity=assessment.approved_quantity,
            verdict=assessment.verdict.value,
            reason=assessment.reason,
            checks=checks,
        )

    def _derive_checks(
        self, *, side: OrderSide, quantity: Decimal, context: RiskContext
    ) -> tuple[RiskCheckResult, ...]:
        """Re-derive each named gate from RiskLimits + RiskContext, mirroring risk.py L70-122.

        This duplicates RiskGate.evaluate's arithmetic because RiskGate returns no per-check
        detail. The order matches the gate's short-circuit order: kill_switch, price,
        sleeve_equity, drawdown, trade_size, then (BUY-only) concentration, buying_power.
        """
        limits = self._risk_limits
        checks: list[RiskCheckResult] = []

        # 1) Kill switch (global halt). No limit/current/projected — it is a boolean gate.
        engaged = self._kill_switch.engaged
        checks.append(
            RiskCheckResult(
                name="kill_switch",
                label="kill switch",
                passed=not engaged,
                limit=None,
                current=None,
                projected=None,
                reason="kill switch engaged" if engaged else "kill switch disengaged",
            )
        )

        # 2) Usable price.
        price = context.price
        price_ok = price > 0
        checks.append(
            RiskCheckResult(
                name="price",
                label="price",
                passed=price_ok,
                limit=None,
                current=_money(price) if price_ok else str(price),
                projected=None,
                reason="usable price" if price_ok else f"no usable price ({price})",
            )
        )

        # 3) Sleeve equity positive.
        equity = context.sleeve_equity
        equity_ok = equity > 0
        checks.append(
            RiskCheckResult(
                name="sleeve_equity",
                label="equity",
                passed=equity_ok,
                limit=None,
                current=_money(equity),
                projected=None,
                reason="sleeve equity positive"
                if equity_ok
                else "sleeve equity is zero or negative",
            )
        )

        # 4) Drawdown circuit breaker. peak_equity == equity in the pre-check, so this always
        # reports 0% — flag the simplification in the reason (see module docstring).
        drawdown_pct = Decimal("0")
        if context.peak_equity > 0:
            drawdown_pct = (context.peak_equity - equity) / context.peak_equity * 100
        drawdown_ok = drawdown_pct < limits.max_drawdown_pct
        checks.append(
            RiskCheckResult(
                name="drawdown",
                label="drawdown",
                passed=drawdown_ok,
                limit=_pct(limits.max_drawdown_pct),
                current=f"{drawdown_pct:.2f}%",
                projected=None,
                reason=(
                    "drawdown not evaluated in pre-check (peak equity unavailable); "
                    f"reported {drawdown_pct:.2f}% < {limits.max_drawdown_pct}% cap"
                ),
            )
        )

        # 5) Per-trade size cap: notional vs sleeve_equity * max_trade_pct / 100.
        max_trade_notional = equity * limits.max_trade_pct_of_sleeve / 100
        requested_notional = quantity * price
        trade_size_ok = requested_notional <= max_trade_notional
        checks.append(
            RiskCheckResult(
                name="trade_size",
                label="size",
                passed=trade_size_ok,
                limit=f"{limits.max_trade_pct_of_sleeve}% of sleeve equity "
                f"({_money(max_trade_notional)})",
                # The trade-size cap is on THIS order's notional, so the "current" value going
                # into a fresh evaluation is zero (not the existing holding).
                current=_money(Decimal("0")),
                projected=_money(requested_notional),
                reason=(
                    f"order {_money(requested_notional)} within "
                    f"{limits.max_trade_pct_of_sleeve}% cap ({_money(max_trade_notional)})"
                    if trade_size_ok
                    else f"trade capped at {limits.max_trade_pct_of_sleeve}% of sleeve equity "
                    f"({_money(max_trade_notional)})"
                ),
            )
        )

        # 6 & 7) BUY-only gates: concentration headroom, then remaining pool buying power.
        if side is OrderSide.BUY:
            concentration_cap = equity * limits.max_position_concentration_pct / 100
            headroom = concentration_cap - context.current_position_value
            projected_position_value = context.current_position_value + requested_notional
            concentration_ok = headroom > 0 and requested_notional <= headroom
            if headroom <= 0:
                conc_reason = (
                    f"position concentration cap ({limits.max_position_concentration_pct}%) "
                    "already reached"
                )
            elif not concentration_ok:
                conc_reason = (
                    f"buy capped by {limits.max_position_concentration_pct}% concentration limit"
                )
            else:
                conc_reason = (
                    f"projected {_money(projected_position_value)} within "
                    f"{limits.max_position_concentration_pct}% cap ({_money(concentration_cap)})"
                )
            checks.append(
                RiskCheckResult(
                    name="concentration",
                    label="concentration",
                    passed=concentration_ok,
                    limit=f"{limits.max_position_concentration_pct}% concentration "
                    f"({_money(concentration_cap)})",
                    current=_money(context.current_position_value),
                    projected=_money(projected_position_value),
                    reason=conc_reason,
                )
            )

            # Shared-pool buying-power guard, mirroring the gate's three-state reading of the
            # reserve exactly (review F17): `None` is an UNKNOWN reserve, which the gate SKIPS —
            # report it as passing with no binding limit. A KNOWN reserve is enforced even at
            # exactly 0 (an exhausted account floors the buy to zero → the gate rejects), so 0
            # must NOT read as "guard disabled" here either.
            bp = context.remaining_buying_power
            if bp is not None:
                bp_ok = requested_notional <= bp
                bp_reason = (
                    f"order {_money(requested_notional)} within buying power ({_money(bp)})"
                    if bp_ok
                    else "buy capped by remaining pool buying power"
                )
            else:
                bp_ok = True
                bp_reason = "buying-power guard not evaluated (unknown reserve)"
            checks.append(
                RiskCheckResult(
                    name="buying_power",
                    label="BP",
                    passed=bp_ok,
                    limit=_money(bp) if bp is not None else None,
                    current=None,
                    projected=_money(requested_notional),
                    reason=bp_reason,
                )
            )

        return tuple(checks)
