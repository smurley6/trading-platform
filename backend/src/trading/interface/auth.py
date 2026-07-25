"""API-key authentication for the mutating/sensitive endpoints (spec section 13, NFRs).

Default-open by construction: when no `api_key` is configured the guard is a no-op (dev
default) and the app logs a single startup warning so an unprotected deployment is never
silent. When an `api_key` IS configured, every protected endpoint requires it via either the
`Authorization: Bearer <key>` or the `X-API-Key` header; a missing or mismatched key is 401.

Writes and SENSITIVE reads (account financials, the full event log, the event firehose) are
guarded when a key is configured (review F24); genuinely public reads (sleeve list, conviction,
health) stay open regardless. The guard is a FastAPI dependency so wiring it onto a route is one
`Depends(...)` and nothing in the application or domain layers learns about HTTP auth; the
WebSocket routes reuse the same check via `is_authorized` / `authorize_websocket`.

The browser `WebSocket` constructor cannot set request headers, so a browser can never present
`Authorization`/`X-API-Key` on a handshake. `authorize_websocket` therefore also accepts the key
as a `Sec-WebSocket-Protocol` offer — `new WebSocket(url, ["api-key", "<key>"])` (task-22). A
query parameter is deliberately NOT supported: it would land in the access log and every proxy
log, which is the opposite of what the F24 guard exists to achieve.
"""

import hmac
import logging

from fastapi import Header, HTTPException, WebSocket, status

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "

# Sentinel subprotocol marking a browser-shaped credential offer: the client offers exactly
# ["api-key", "<key>"] and the server echoes back the SENTINEL (never the key — echoing it would
# put the secret into the handshake RESPONSE header, and thus into proxy logs).
WS_API_KEY_SUBPROTOCOL = "api-key"


class ApiKeyAuth:
    """Callable FastAPI dependency that enforces the configured API key.

    Constructed once per app from the resolved `api_key`. With `api_key is None` it permits
    every request (and logs one warning at construction). Otherwise it constant-time compares
    the presented key against the configured one.
    """

    def __init__(self, api_key: str | None) -> None:
        # An empty or whitespace-only key means "unset" (open): writing `API_KEY=` in .env is a
        # common way to say "no auth", and must never enforce a key that nothing can present
        # (which would 401 every write). Normalise such values to None here.
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        if self._api_key is None:
            logger.warning(
                "API_KEY is not set: mutating endpoints are UNPROTECTED (open). "
                "Set API_KEY to require authentication on writes."
            )

    @property
    def enabled(self) -> bool:
        return self._api_key is not None

    def is_authorized(self, authorization: str | None, x_api_key: str | None) -> bool:
        """Return whether the presented headers satisfy the guard, WITHOUT raising.

        Open mode (no configured key) authorizes everything. Otherwise the presented key must
        constant-time match. Used by the WebSocket routes, which cannot lean on the
        ``Depends``/``HTTPException`` flow the HTTP routes use (a rejected socket is closed, not
        answered with a 401 body)."""
        if self._api_key is None:
            return True  # open mode
        presented = _extract_key(authorization, x_api_key)
        return presented is not None and hmac.compare_digest(presented, self._api_key)

    def authorize_websocket(self, websocket: WebSocket) -> tuple[bool, str | None]:
        """Authorize a handshake; return `(authorized, subprotocol to echo on accept)`.

        Headers are checked FIRST, so non-browser clients (services, curl, the backend's own
        tests) behave exactly as before. Only if they do not authorize is the browser path
        considered: the `Sec-WebSocket-Protocol` offer, the sole credential channel the browser
        `WebSocket` constructor exposes.

        Fail closed: the offer must be EXACTLY `[WS_API_KEY_SUBPROTOCOL, "<key>"]`. Any other
        shape (extra tokens, reordered, sentinel alone) is ambiguous and is rejected rather than
        guessed at. The key is compared by the same constant-time path as the header key.

        The returned subprotocol MUST be passed to `websocket.accept(subprotocol=...)`: RFC 6455
        requires the server's selection to be one the client offered, and browsers fail the
        handshake otherwise. It is always the sentinel, never the key. Note that subprotocol
        tokens cannot carry commas or whitespace, so a key containing either is unrepresentable
        here and is simply rejected (fail closed).
        """
        offered = websocket_subprotocols(websocket)
        # Whatever authorizes the socket, an accept must answer the client's offer with a token
        # the client actually sent — including in open mode (no API_KEY), where the console still
        # offers the sentinel.
        echo = WS_API_KEY_SUBPROTOCOL if WS_API_KEY_SUBPROTOCOL in offered else None
        if self.is_authorized(
            websocket.headers.get("authorization"), websocket.headers.get("x-api-key")
        ):
            return True, echo
        if (
            len(offered) == 2
            and offered[0] == WS_API_KEY_SUBPROTOCOL
            and self.is_authorized(None, offered[1])
        ):
            return True, WS_API_KEY_SUBPROTOCOL
        return False, None

    async def __call__(
        self,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if not self.is_authorized(authorization, x_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )


def websocket_subprotocols(websocket: WebSocket) -> list[str]:
    """The subprotocol tokens the client offered, normalized across ASGI servers.

    `scope["subprotocols"]` is NOT populated uniformly: uvicorn's `websockets_impl` and
    Starlette's `TestClient` comma-split the `Sec-WebSocket-Protocol` header, while
    `websockets_sansio_impl` hands back one entry per raw header line with the commas intact.
    Both shapes (plus the raw header, in case a server leaves the scope key off entirely) are
    folded here — every candidate is comma-split and stripped — so the credential contract does
    not depend on which server is serving. Order is preserved; duplicates collapse (the scope
    list and the raw header normally carry the same tokens twice).
    """
    candidates: list[str] = []
    scoped = websocket.scope.get("subprotocols")
    if isinstance(scoped, list | tuple):
        candidates.extend(str(value) for value in scoped)
    candidates.extend(websocket.headers.getlist("sec-websocket-protocol"))
    tokens: list[str] = []
    for candidate in candidates:
        for raw in candidate.split(","):
            token = raw.strip()
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the presented key from either header. `Authorization: Bearer <key>` wins; the
    `X-API-Key` header is the fallback for simple clients."""
    if authorization is not None and authorization.lower().startswith(_BEARER_PREFIX):
        token = authorization[len(_BEARER_PREFIX) :].strip()
        if token:
            return token
    if x_api_key is not None and x_api_key.strip():
        return x_api_key.strip()
    return None
