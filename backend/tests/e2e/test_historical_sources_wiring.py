"""Composition-root wiring for the backfill-capable (historical) source registry (Task 04a).

`sec_form4` must be registered in `historical_sources` whenever `SEC_FORM4_USER_AGENT` is set --
INDEPENDENTLY of `SEC_FORM4_ENABLED` (the live-feed flag). Backfill is a research op with its own
gate (`BACKFILL_ENABLED`, checked at `--confirm` time by the job), not the live ingestion switch.
With no user agent set (the default), the registry stays empty and no network-capable client is
constructed for it -- matching the "bare run makes no network calls" contract.
"""

import pytest

from trading.config import Settings
from trading.infrastructure.data_sources.sec_form4 import SecForm4Source
from trading.main import build_dependencies

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CLEARED_ENV = (
    "DATABASE_URL",
    "SEC_FORM4_ENABLED",
    "SEC_FORM4_USER_AGENT",
    "ENABLE_ALPACA_PAPER",
    "ALLOW_LIVE",
    "REQUIRE_REAL_DATA",
    "API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _RISK_CAPS.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)


async def test_no_user_agent_means_empty_historical_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = build_dependencies(Settings(_env_file=None))
    try:
        assert "sec_form4" not in deps.historical_sources
    finally:
        for aclose in deps.aclosers:
            await aclose()


async def test_user_agent_registers_sec_form4_even_when_live_feed_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_FORM4_USER_AGENT", "trading-platform research test@example.com")
    monkeypatch.setenv("SEC_FORM4_ENABLED", "false")  # live feed OFF -- must not matter here
    deps = build_dependencies(Settings(_env_file=None))
    try:
        assert "sec_form4" in deps.historical_sources
        assert isinstance(deps.historical_sources["sec_form4"], SecForm4Source)
    finally:
        for aclose in deps.aclosers:
            await aclose()


async def test_user_agent_registers_sec_form4_when_live_feed_also_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_FORM4_USER_AGENT", "trading-platform research test@example.com")
    monkeypatch.setenv("SEC_FORM4_ENABLED", "true")
    deps = build_dependencies(Settings(_env_file=None))
    try:
        assert "sec_form4" in deps.historical_sources
        # Both the live cycle and the historical registry are wired independently: enabling the
        # live feed too must not remove or duplicate the historical registration.
        assert deps.political_cycle is not None
    finally:
        for aclose in deps.aclosers:
            await aclose()
