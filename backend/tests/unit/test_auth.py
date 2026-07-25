"""ApiKeyAuth dependency: open vs enforced, header parsing, WebSocket handshake auth."""

from typing import Any

import pytest
from fastapi import HTTPException, WebSocket

from trading.interface.auth import ApiKeyAuth, _extract_key, websocket_subprotocols


def test_extract_key_prefers_bearer() -> None:
    assert _extract_key("Bearer abc", "xyz") == "abc"
    assert _extract_key("bearer abc", None) == "abc"  # case-insensitive scheme


def test_extract_key_falls_back_to_x_api_key() -> None:
    assert _extract_key(None, "xyz") == "xyz"
    assert _extract_key("Basic zzz", "xyz") == "xyz"  # non-bearer auth ignored


def test_extract_key_none_when_absent_or_empty() -> None:
    assert _extract_key(None, None) is None
    assert _extract_key("Bearer ", "  ") is None


async def test_open_mode_permits_any_request() -> None:
    auth = ApiKeyAuth(None)
    assert auth.enabled is False
    # No exception regardless of headers.
    await auth(authorization=None, x_api_key=None)


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
async def test_blank_api_key_is_treated_as_open(blank: str) -> None:
    # `API_KEY=` in .env resolves to an empty string, not None — it must mean "open", never
    # "enforce a key nothing can present" (which would 401 every write).
    auth = ApiKeyAuth(blank)
    assert auth.enabled is False
    await auth(authorization=None, x_api_key=None)


async def test_enforced_mode_accepts_valid_key() -> None:
    auth = ApiKeyAuth("secret")
    assert auth.enabled is True
    await auth(authorization="Bearer secret", x_api_key=None)
    await auth(authorization=None, x_api_key="secret")


async def test_enforced_mode_rejects_invalid_or_missing() -> None:
    auth = ApiKeyAuth("secret")
    with pytest.raises(HTTPException) as missing:
        await auth(authorization=None, x_api_key=None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        await auth(authorization="Bearer nope", x_api_key=None)
    assert wrong.value.status_code == 401


def _handshake(
    *,
    headers: dict[str, str] | None = None,
    scope_subprotocols: list[str] | None = None,
    raw_header: str | None = None,
) -> WebSocket:
    """A WebSocket frozen at the handshake — all the auth decision ever looks at.

    `scope_subprotocols` models the ALREADY comma-split shape (Starlette's TestClient, uvicorn's
    `websockets_impl`); `raw_header` models the un-split raw header line (uvicorn's
    `websockets_sansio_impl` puts that straight into the scope list).
    """
    raw: list[tuple[bytes, bytes]] = [
        (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
    ]
    if raw_header is not None:
        raw.append((b"sec-websocket-protocol", raw_header.encode()))
    scope: dict[str, Any] = {
        "type": "websocket",
        "path": "/ws/events",
        "headers": raw,
        "subprotocols": [] if scope_subprotocols is None else scope_subprotocols,
    }

    async def receive() -> Any:
        raise AssertionError("handshake auth must never read from the socket")

    async def send(message: Any) -> None:
        raise AssertionError("handshake auth must never write to the socket")

    return WebSocket(scope, receive, send)


def test_websocket_subprotocols_normalizes_every_asgi_shape() -> None:
    # Comma-split into the scope list AND present raw: the duplicate collapses.
    split = _handshake(scope_subprotocols=["api-key", "abc"], raw_header="api-key, abc")
    assert websocket_subprotocols(split) == ["api-key", "abc"]
    # One scope entry per raw header line, commas intact — normalized identically.
    unsplit = _handshake(scope_subprotocols=["api-key, abc"], raw_header="api-key, abc")
    assert websocket_subprotocols(unsplit) == ["api-key", "abc"]
    # Nothing offered.
    assert websocket_subprotocols(_handshake()) == []


def test_websocket_open_mode_permits_and_still_echoes_the_offer() -> None:
    # No key configured: open, exactly as before. The offer is still echoed, because a browser
    # fails the handshake when the server answers its offer with no selection.
    auth = ApiKeyAuth(None)
    offered = _handshake(scope_subprotocols=["api-key", "whatever"])
    assert auth.authorize_websocket(offered) == (True, "api-key")
    assert auth.authorize_websocket(_handshake()) == (True, None)


def test_websocket_headers_are_checked_before_the_browser_path() -> None:
    auth = ApiKeyAuth("secret")
    bearer = _handshake(headers={"Authorization": "Bearer secret"})
    assert auth.authorize_websocket(bearer) == (True, None)
    assert auth.authorize_websocket(_handshake(headers={"X-API-Key": "secret"})) == (True, None)


def test_websocket_browser_offer_carries_the_key_and_echoes_the_sentinel() -> None:
    auth = ApiKeyAuth("secret")
    # The echoed subprotocol is the sentinel, NEVER the key — echoing the key would put the
    # secret into the handshake response header.
    assert auth.authorize_websocket(_handshake(scope_subprotocols=["api-key", "secret"])) == (
        True,
        "api-key",
    )


@pytest.mark.parametrize(
    "offered",
    [
        ["api-key", "wrong"],  # wrong key
        ["api-key"],  # sentinel with no key
        ["api-key", "secret", "extra"],  # ambiguous: extra token
        ["extra", "api-key", "secret"],  # ambiguous: sentinel not first
        ["secret", "api-key"],  # reversed
        ["API-KEY", "secret"],  # subprotocol tokens are case-sensitive
        [],  # no credential at all
    ],
)
def test_websocket_rejects_wrong_or_ambiguous_offers(offered: list[str]) -> None:
    # Fail closed: anything that is not exactly ["api-key", "<key>"] is rejected, never guessed.
    auth = ApiKeyAuth("secret")
    assert auth.authorize_websocket(_handshake(scope_subprotocols=offered)) == (False, None)
