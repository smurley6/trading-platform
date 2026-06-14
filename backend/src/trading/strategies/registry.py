"""Strategy registry (FR-STR-2/3): adding a strategy = one module + one registration."""

from collections.abc import Callable
from typing import TypeVar

from trading.strategies.base import Strategy

_REGISTRY: dict[str, Callable[[], Strategy]] = {}

_F = TypeVar("_F", bound=Callable[[], Strategy])


def register(strategy_id: str) -> Callable[[_F], _F]:
    def decorator(factory: _F) -> _F:
        if strategy_id in _REGISTRY:
            raise ValueError(f"strategy '{strategy_id}' is already registered")
        _REGISTRY[strategy_id] = factory
        return factory

    return decorator


def create(strategy_id: str) -> Strategy:
    try:
        factory = _REGISTRY[strategy_id]
    except KeyError:
        raise KeyError(
            f"unknown strategy '{strategy_id}'; available: {sorted(_REGISTRY)}"
        ) from None
    return factory()


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
