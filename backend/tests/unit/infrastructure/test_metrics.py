"""MetricsRegistry: counting and Prometheus text exposition."""

from trading.infrastructure.observability.metrics import MetricsRegistry


def test_request_counts_render_in_prometheus_format() -> None:
    m = MetricsRegistry()
    m.observe_request(method="GET", path="/health", status_code=200, duration=0.01)
    m.observe_request(method="GET", path="/health", status_code=200, duration=0.02)
    m.observe_request(method="POST", path="/signals", status_code=401, duration=0.003)

    text = m.render()
    assert "# TYPE http_requests_total counter" in text
    assert 'http_requests_total{method="GET",path="/health",status="200"} 2' in text
    assert 'http_requests_total{method="POST",path="/signals",status="401"} 1' in text


def test_latency_histogram_buckets_are_cumulative() -> None:
    m = MetricsRegistry()
    for d in (0.001, 0.02, 0.3, 7.0):
        m.observe_request(method="GET", path="/x", status_code=200, duration=d)

    text = m.render()
    assert "# TYPE http_request_duration_seconds histogram" in text
    # All four observations are <= +Inf, sum/count reflect every observation.
    assert 'http_request_duration_seconds_bucket{le="+Inf"} 4' in text
    assert "http_request_duration_seconds_count 4" in text
    # The 0.005s bucket only contains the 0.001 observation.
    assert 'http_request_duration_seconds_bucket{le="0.005"} 1' in text


def test_domain_counters() -> None:
    m = MetricsRegistry()
    m.increment("events_appended_total")
    m.increment("events_appended_total", 3)
    m.increment("orders_placed_total")
    m.increment("ignored_zero", 0)

    assert m.counter_value("events_appended_total") == 4
    text = m.render()
    assert "events_appended_total 4" in text
    assert "orders_placed_total 1" in text
    assert "ignored_zero" not in text


def test_label_values_are_escaped() -> None:
    m = MetricsRegistry()
    m.observe_request(method="GET", path='/a"b', status_code=200, duration=0.01)
    assert '\\"' in m.render()
