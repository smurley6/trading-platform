"""Hand-rolled, dependency-light metrics registry + Prometheus text exposition (spec §15).

No heavy client library: a couple of counters and one latency histogram are all the spec asks
for (requests, latency, plus a few domain counters), so a ~100-line registry that renders the
Prometheus text-exposition format keeps the dependency footprint at zero. Thread/async safe
enough for our single-process app: increments are plain attribute mutations under the GIL and
the FastAPI event loop is single-threaded.

Exposes:
  * a request counter labelled by method/path/status,
  * a request-latency histogram (fixed buckets, seconds),
  * named domain counters (events appended, orders placed) bumped from the wiring layer.
"""

from dataclasses import dataclass, field

# Latency buckets in seconds (Prometheus convention: cumulative "le" upper bounds).
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{name}="{_escape_label(value)}"' for name, value in labels)
    return "{" + inner + "}"


@dataclass(slots=True)
class _Histogram:
    """A single-series cumulative-bucket latency histogram."""

    buckets: tuple[float, ...] = _LATENCY_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1


class MetricsRegistry:
    """In-process registry. One instance per app, shared with the middleware and wiring."""

    def __init__(self) -> None:
        # request totals keyed by (method, path, status_code-as-str)
        self._request_counts: dict[tuple[tuple[str, str], ...], int] = {}
        self._latency = _Histogram()
        # free-form domain counters keyed by metric name (no labels) -> value
        self._counters: dict[str, int] = {}

    # --- recording -------------------------------------------------------------------------
    def observe_request(self, *, method: str, path: str, status_code: int, duration: float) -> None:
        key = (("method", method), ("path", path), ("status", str(status_code)))
        self._request_counts[key] = self._request_counts.get(key, 0) + 1
        self._latency.observe(duration)

    def increment(self, name: str, amount: int = 1) -> None:
        """Bump a named domain counter (e.g. ``events_appended_total``)."""
        if amount <= 0:
            return
        self._counters[name] = self._counters.get(name, 0) + amount

    def counter_value(self, name: str) -> int:
        return self._counters.get(name, 0)

    # --- exposition ------------------------------------------------------------------------
    def render(self) -> str:
        lines: list[str] = []
        lines += self._render_requests()
        lines += self._render_latency()
        lines += self._render_counters()
        return "\n".join(lines) + "\n"

    def _render_requests(self) -> list[str]:
        out = [
            "# HELP http_requests_total Total HTTP requests by method, path and status.",
            "# TYPE http_requests_total counter",
        ]
        for key, value in sorted(self._request_counts.items()):
            out.append(f"http_requests_total{_render_labels(key)} {value}")
        return out

    def _render_latency(self) -> list[str]:
        h = self._latency
        out = [
            "# HELP http_request_duration_seconds HTTP request latency in seconds.",
            "# TYPE http_request_duration_seconds histogram",
        ]
        cumulative = 0
        for i, upper in enumerate(h.buckets):
            cumulative += h.counts[i]
            out.append(
                f'http_request_duration_seconds_bucket{{le="{_format_le(upper)}"}} {cumulative}'
            )
        out.append(f'http_request_duration_seconds_bucket{{le="+Inf"}} {h.count}')
        out.append(f"http_request_duration_seconds_sum {h.total}")
        out.append(f"http_request_duration_seconds_count {h.count}")
        return out

    def _render_counters(self) -> list[str]:
        out: list[str] = []
        for name, value in sorted(self._counters.items()):
            out.append(f"# HELP {name} Domain counter.")
            out.append(f"# TYPE {name} counter")
            out.append(f"{name} {value}")
        return out


def _format_le(upper: float) -> str:
    # Render integer bounds without a trailing ".0" but keep fractional bounds intact.
    return str(int(upper)) if upper == int(upper) else repr(upper)
