"""Settings credential resolvers (data + paper key fallback).

`_env_file=None` plus delenv keeps any developer `.env` / shell exports out of these tests.
"""

import pytest

from trading.config import Settings

_RISK = {
    "MAX_TRADE_PCT_OF_SLEEVE": "10",
    "MAX_POSITION_CONCENTRATION_PCT": "25",
    "MAX_DRAWDOWN_PCT": "20",
}
_CRED_KEYS = (
    "ALPACA_DATA_KEY_ID",
    "ALPACA_DATA_SECRET_KEY",
    "ALPACA_PAPER_KEY_ID",
    "ALPACA_PAPER_SECRET_KEY",
)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key in _CRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {**_RISK, **env}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_paper_keys_fall_back_to_data_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, ALPACA_DATA_KEY_ID="DK", ALPACA_DATA_SECRET_KEY="DS")
    assert s.paper_key_id() == "DK"
    assert s.paper_secret_key() == "DS"


def test_dedicated_paper_keys_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(
        monkeypatch,
        ALPACA_DATA_KEY_ID="DK",
        ALPACA_DATA_SECRET_KEY="DS",
        ALPACA_PAPER_KEY_ID="PK",
        ALPACA_PAPER_SECRET_KEY="PS",
    )
    assert s.paper_key_id() == "PK"
    assert s.paper_secret_key() == "PS"


def test_paper_keys_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch)
    assert s.paper_key_id() is None
    assert s.paper_secret_key() is None
    assert s.enable_alpaca_paper is False  # off by construction


def test_hardening_defaults_are_off(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch)
    assert s.api_key is None  # open by default (dev)
    assert s.enable_trade_updates_stream is False  # no socket by default
    assert s.enable_metrics is True


def test_trade_updates_stream_url_derived_from_paper_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _settings(monkeypatch)
    # https paper base -> wss + /stream.
    assert s.trade_updates_stream_url() == "wss://paper-api.alpaca.markets/stream"


def test_trade_updates_stream_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(monkeypatch, ALPACA_TRADE_UPDATES_STREAM_URL="wss://custom/stream")
    assert s.trade_updates_stream_url() == "wss://custom/stream"
