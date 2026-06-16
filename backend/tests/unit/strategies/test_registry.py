"""Registry conformance (spec section 18): a strategy is built solely via registration.

Importing `trading.strategies` runs each module's `@register` side effect, so both shipped
strategies must be creatable through `registry.create(...)` with a typed `StrategyConfig` —
no direct construction in the composition root.
"""

import pytest

from trading.domain.conviction import ConvictionParams, ConvictionThresholds
from trading.strategies import StrategyConfig, registry
from trading.strategies.political_signal import PoliticalSignalStrategy
from trading.strategies.trivial import TrivialStrategy


def _config() -> StrategyConfig:
    return StrategyConfig(thresholds=ConvictionThresholds(trade=65.0, watch=30.0), window_days=14)


def test_both_shipped_strategies_are_registered() -> None:
    assert {"political_signal", "trivial"} <= set(registry.available())


def test_create_trivial_ignores_config() -> None:
    strategy = registry.create("trivial", _config())
    assert isinstance(strategy, TrivialStrategy)
    assert strategy.strategy_id == "trivial"


def test_create_political_applies_config() -> None:
    config = _config()
    strategy = registry.create("political_signal", config)
    assert isinstance(strategy, PoliticalSignalStrategy)
    assert strategy.strategy_id == "political_signal"
    assert strategy.thresholds == config.thresholds
    assert strategy.window_days == config.window_days


def test_config_default_conviction_params_match_m1() -> None:
    params = StrategyConfig().conviction_params()
    assert params == ConvictionParams()


def test_config_threads_conviction_tunables() -> None:
    config = StrategyConfig(
        decay_half_life_days=10.0,
        corroboration_bonus_per_source=0.3,
        min_sources_for_full_corroboration=2,
        max_corroboration_factor=1.4,
        conviction_scale=30.0,
    )
    params = config.conviction_params()
    assert params.decay_half_life_days == 10.0
    assert params.corroboration_bonus_per_source == 0.3
    assert params.min_sources_for_full_corroboration == 2
    assert params.max_corroboration_factor == 1.4
    assert params.scale == 30.0


def test_create_political_applies_conviction_params() -> None:
    config = StrategyConfig(decay_half_life_days=5.0)
    strategy = registry.create("political_signal", config)
    assert isinstance(strategy, PoliticalSignalStrategy)
    assert strategy.conviction_params == config.conviction_params()
    assert strategy.conviction_params.decay_half_life_days == 5.0


def test_create_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError, match="unknown strategy 'nope'"):
        registry.create("nope", _config())


def test_register_rejects_duplicate_id() -> None:
    def _factory(config: StrategyConfig) -> TrivialStrategy:  # pragma: no cover
        return TrivialStrategy()

    # "trivial" was registered at import time, so re-registering must fail.
    with pytest.raises(ValueError, match="already registered"):
        registry.register("trivial")(_factory)
