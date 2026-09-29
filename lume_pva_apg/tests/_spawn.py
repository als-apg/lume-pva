"""Wait for a spawned test server to come up.

A "spawn" child re-imports the package and its transports before it can serve,
and on a loaded or emulated host that alone can outlast an operation timeout.
Startup therefore gets its own, longer deadline. The wait polls the child
between short waits, so a child that dies during startup fails the test at
once with its exit code instead of running out the deadline.
"""

from __future__ import annotations

import time
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as mpEvent

# Upper bound for a spawned child to import, build its Runner and serve.
STARTUP_TIMEOUT = 120.0
_POLL = 0.5


def wait_until_ready(proc: BaseProcess, ready: mpEvent) -> None:
    """Block until ``ready`` is set; fail if ``proc`` dies or the deadline passes."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not ready.wait(timeout=_POLL):
        if not proc.is_alive():
            raise AssertionError(f"child Runner exited during startup (exitcode {proc.exitcode})")
        if time.monotonic() >= deadline:
            raise AssertionError(f"child Runner never became ready within {STARTUP_TIMEOUT:.0f} s")
