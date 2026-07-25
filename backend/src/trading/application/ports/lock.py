"""Cross-process lock port: a named advisory lock for serializing a critical section across
separate OS processes (review F6).

The in-process overlap (a double-clicked Execute, the scheduler racing POST /ingest) is closed by
an ``asyncio.Lock`` held inside the use case; that lock cannot see a *different* process. The
cron-vs-web case (the daily cron process racing the always-on web app) needs a lock the database
arbitrates. The Postgres adapter implements this with a session-level advisory lock; the default
in-memory deployment (no database) uses a no-op adapter, since there is only one process.
"""

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class CrossProcessLockPort(Protocol):
    def guard(self, key: str) -> AbstractAsyncContextManager[None]:
        """Hold a named advisory lock for the duration of the returned async context.

        Entering blocks until the lock is granted; exiting MUST release it on every path
        (normal return or exception), so a mid-critical-section failure never strands the lock.
        """
        ...
