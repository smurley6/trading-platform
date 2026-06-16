"""Structured JSON logging with correlation ids (spec section 15)."""

import json
import logging
import sys
from typing import Any

# Structured extras the request middleware attaches; surfaced as top-level JSON keys so an
# access log line is machine-parseable (method/path/status/latency) alongside the message.
_EXTRA_FIELDS = ("http_method", "http_path", "http_status", "duration_ms")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id is not None:
            data["correlation_id"] = correlation_id
        for field_name in _EXTRA_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                data[field_name] = value
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging(level: str = "INFO") -> None:
    # Import here to avoid an interface->observability import at module load (the filter lives
    # in the interface layer); configure_logging is only ever called from the composition root.
    from trading.interface.middleware.observability import CorrelationIdFilter

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(CorrelationIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
