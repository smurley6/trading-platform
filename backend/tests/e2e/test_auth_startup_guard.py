"""Fail-closed auth startup guard for the SERVED app (review F12).

When a real Alpaca broker (paper or live) is wired but `API_KEY` is unset, `build_dependencies`
must refuse to start the served app (`serves_http=True`) — the order-placing / risk-control
endpoints would otherwise be open to anyone with the URL. The headless cron (`serves_http=False`,
no endpoints) is unaffected, and an explicit `ALLOW_UNAUTHENTICATED=true` opts out.
"""

from collections.abc import AsyncIterator

import pytest

from trading.config import Settings
from trading.interface.app import AppDeps
from trading.main import build_dependencies

_RISK_CAPS = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
# A real paper broker AND a resolvable live data feed (so the earlier market-data guard passes and
# we actually reach the auth guard). Constructing the Alpaca clients dials nothing.
_WIRED_REAL_BROKER = {
    **_RISK_CAPS,
    "ENABLE_ALPACA_PAPER": "true",
    "ALPACA_PAPER_KEY_ID": "PKTEST",
    "ALPACA_PAPER_SECRET_KEY": "paper-secret",
    "ENABLE_LIVE_QUOTES": "true",
    "ALPACA_DATA_KEY_ID": "DKTEST",
    "ALPACA_DATA_SECRET_KEY": "data-secret",
}
_CLEARED = (
    "API_KEY",
    "ALLOW_UNAUTHENTICATED",
    "DATABASE_URL",
    "MODE_DEFAULT",
    "ALPACA_LIVE_KEY_ID",
    "ALPACA_LIVE_SECRET_KEY",
    "ALLOW_LIVE",
)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key in _CLEARED:
        monkeypatch.delenv(key, raising=False)
    for key, value in {**_WIRED_REAL_BROKER, **env}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


async def _closed(deps: AppDeps) -> AsyncIterator[AppDeps]:
    try:
        yield deps
    finally:
        for aclose in deps.aclosers:
            await aclose()


def test_served_app_refuses_real_broker_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="API_KEY"):
        build_dependencies(_settings(monkeypatch), serves_http=True)


async def test_served_app_starts_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = build_dependencies(_settings(monkeypatch, API_KEY="s3cret"), serves_http=True)
    async for d in _closed(deps):
        assert d.api_key == "s3cret"


async def test_served_app_starts_with_explicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, ALLOW_UNAUTHENTICATED="true")
    deps = build_dependencies(settings, serves_http=True)
    async for d in _closed(deps):
        assert d.api_key is None


async def test_cron_path_is_unaffected_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The headless cron builds deps with serves_http defaulting False: no HTTP surface to protect,
    # so a real paper broker with no API_KEY is fine (the currently-shipping paper deployment).
    deps = build_dependencies(_settings(monkeypatch))
    async for d in _closed(deps):
        assert d.pool_broker is not None  # the real paper broker IS wired


async def test_non_token_safe_api_key_warns_about_the_browser_websocket_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A base64-style key authenticates over HTTP but is unrepresentable as a
    `Sec-WebSocket-Protocol` token, so the browser event stream would fail silently (task-22).
    The startup warning is the only signal an operator gets."""
    caplog.set_level("WARNING", logger="trading.main")
    deps = build_dependencies(_settings(monkeypatch, API_KEY="aB3+xY/z9Q=="), serves_http=True)
    async for _ in _closed(deps):
        pass

    assert "Sec-WebSocket-Protocol" in caplog.text or "WebSocket" in caplog.text
    # The key itself must never be echoed into the log line.
    assert "aB3+xY/z9Q==" not in caplog.text


async def test_token_safe_api_key_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="trading.main")
    deps = build_dependencies(_settings(monkeypatch, API_KEY="a" * 64), serves_http=True)
    async for _ in _closed(deps):
        pass

    assert "WebSocket" not in caplog.text
