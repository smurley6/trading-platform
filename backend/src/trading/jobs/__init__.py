"""One-shot CLI job entrypoints (scheduled runs, e.g. GitHub Actions cron).

These modules compose the application like the FastAPI composition root in ``trading.main``
and drive a single use-case to completion, then tear down all async resources and exit. They
are the headless counterpart to the always-on server: the same wiring, no HTTP surface.
"""
