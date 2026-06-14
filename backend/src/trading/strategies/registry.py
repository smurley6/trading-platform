"""Strategy registry (FR-STR-2/3): adding a strategy = one module + one registration."""

from collections.abc import Callable
from typing import TypeVar

from trading.strategies.base import Strategy, StrategyConfig

# A factory builds a strategy from an explicit, typed config so the same registry can serve
# config-free strategies (which ignore it) and config-driven ones (which read it).
StrategyFactory = Callable[[StrategyConfig], Strategy]

_REGISTRY: dict[str, StrategyFactory] = {}

_F = TypeVar("_F", bound=StrategyFactory)


def register(strategy_id: str) -> Callable[[_F], _F]:
    def decorator(factory: _F) -> _F:
        if strategy_id in _REGISTRY:
            raise ValueError(f"strategy '{strategy_id}' is already registered")
        _REGISTRY[strategy_id] = factory
        return factory

    return decorator


def create(strategy_id: str, config: StrategyConfig) -> Strategy:
    try:
        factory = _REGISTRY[strategy_id]
    except KeyError:
        raise KeyError(
            f"unknown strategy '{strategy_id}'; available: {sorted(_REGISTRY)}"
        ) from None
    return factory(config)


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
