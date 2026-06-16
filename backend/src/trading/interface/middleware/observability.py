"""Structured request logging + metrics middleware (spec §15).

Every request gets a request/correlation id (taken from an inbound ``X-Request-ID`` header
when present, else generated) that is:
  * bound to a contextvar so any log record emitted while handling the request carries it,
  * echoed back on the response as ``X-Request-ID``,
  * included in one structured JSON access-log line per request.

The same hook feeds the metrics registry (request count by method/path/status + latency), so
``/metrics`` reflects live traffic. Dependency-light: Starlette's BaseHTTPMiddleware + stdlib.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from trading.infrastructure.observability.metrics import MetricsRegistry

_REQUEST_HEADER = "X-Request-ID"

# Bound per request; log records pick it up via CorrelationIdFilter (see configure below).
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

logger = logging.getLogger("trading.access")


class CorrelationIdFilter(logging.Filter):
    """Attach the current request id to every log record as ``correlation_id`` (if unset).

    The JSON formatter already serialises ``correlation_id``; this filter sources it from the
    request contextvar so logs emitted deep in a request handler are correlated without every
    call site threading the id through manually.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "correlation_id", None) is None:
            rid = request_id_ctx.get()
            if rid is not None:
                # LogRecord carries arbitrary fields in __dict__; assign there so the JSON
                # formatter's getattr finds it without tripping a static attr-defined error.
                record.__dict__["correlation_id"] = rid
        return True


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, metrics: MetricsRegistry) -> None:
        super().__init__(app)
        self._metrics = metrics

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get(_REQUEST_HEADER) or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[_REQUEST_HEADER] = rid
            return response
        finally:
            duration = time.perf_counter() - start
            # Use the route template (e.g. /sleeves/{sleeve_id}/mode) when matched so the
            # request-count series does not explode on path parameters.
            path = _route_template(request)
            self._metrics.observe_request(
                method=request.method,
                path=path,
                status_code=status_code,
                duration=duration,
            )
            logger.info(
                "request",
                extra={
                    "correlation_id": rid,
                    "http_method": request.method,
                    "http_path": path,
                    "http_status": status_code,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            request_id_ctx.reset(token)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None)
    if isinstance(path_format, str):
        return path_format
    return request.url.path
