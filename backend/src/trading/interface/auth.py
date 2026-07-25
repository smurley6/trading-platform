"""API-key authentication for the mutating/sensitive endpoints (spec section 13, NFRs).

Default-open by construction: when no `api_key` is configured the guard is a no-op (dev
default) and the app logs a single startup warning so an unprotected deployment is never
silent. When an `api_key` IS configured, every protected endpoint requires it via either the
`Authorization: Bearer <key>` or the `X-API-Key` header; a missing or mismatched key is 401.

Writes and SENSITIVE reads (account financials, the full event log, the event firehose) are
guarded when a key is configured (review F24); genuinely public reads (sleeve list, conviction,
health) stay open regardless. The guard is a FastAPI dependency so wiring it onto a route is one
`Depends(...)` and nothing in the application or domain layers learns about HTTP auth; the
WebSocket routes reuse the same check via `is_authorized`.
"""

import hmac
import logging

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "


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
