"""Mutable per-sleeve runtime state (mode, status).

The domain `Sleeve` is immutable identity/metadata; a sleeve's *mode* changes at runtime via
the live opt-in flow, so it lives here rather than on the entity. The registry is the single
source of truth the routing broker and the API read.
"""

from dataclasses import dataclass

from trading.domain.sleeves import Sleeve, SleeveStatus, TradingMode


@dataclass(slots=True)
class SleeveState:
    sleeve: Sleeve
    mode: TradingMode
    status: SleeveStatus


class UnknownSleeveError(KeyError):
    pass


class SleeveRegistry:
    def __init__(self, states: list[SleeveState]) -> None:
        self._states = {state.sleeve.id: state for state in states}

    def get(self, sleeve_id: str) -> SleeveState:
        try:
            return self._states[sleeve_id]
        except KeyError:
            raise UnknownSleeveError(sleeve_id) from None

    def mode(self, sleeve_id: str) -> TradingMode:
        return self.get(sleeve_id).mode

    def set_mode(self, sleeve_id: str, mode: TradingMode) -> None:
        self.get(sleeve_id).mode = mode

    def all(self) -> list[SleeveState]:
        return list(self._states.values())
