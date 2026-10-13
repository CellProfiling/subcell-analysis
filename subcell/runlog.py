"""A timestamped progress log.

``log(msg)`` prints *msg* prefixed with the seconds since import (process start
for a script), so a long run shows which stage it is in and how long the previous
one took. ``stage(msg)`` is a context manager that logs a stage's start and
duration. Plain prints rather than the ``logging`` module: the output is a
narration for a person watching a job, with one format for every call site.
"""

from __future__ import annotations

import time

_T0 = time.perf_counter()


def log(msg: str = "") -> None:
    """Print *msg* prefixed with seconds since import, flushed immediately.

    Flushed so the log is readable while a batch job is still running.
    """
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def elapsed() -> float:
    """Seconds since import."""
    return time.perf_counter() - _T0


class stage:
    """Context manager logging a stage's start and duration.

    ``with stage("fitting UMAP"): ...``
    """

    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(self.msg)
        return self

    def __exit__(self, *exc):
        log(f"  {self.msg}: {time.perf_counter() - self.t0:.0f}s")
        return False
