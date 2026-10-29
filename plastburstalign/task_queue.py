"""Concurrency utilities and multiprocessing context handling."""

import multiprocessing
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.context import BaseContext


def get_mp_context(force_context: str | None = None) -> BaseContext:
    """Returns an appropriate multiprocessing context based on platform and configuration.

    - Windows (win32): Defaults to 'spawn'.
    - macOS (darwin): Defaults to 'spawn' (due to crash risks with fork in macOS
    frameworks).
    - Linux/Unix: Defaults to 'fork', but supports 'spawn' via PBA_MP_CONTEXT
    environment variable.
    """
    if force_context:
        return multiprocessing.get_context(force_context)

    env_override = os.environ.get("PBA_MP_CONTEXT")
    if env_override:
        return multiprocessing.get_context(env_override)

    if sys.platform in ("win32", "darwin"):
        return multiprocessing.get_context("spawn")

    return multiprocessing.get_context("fork")


class CoreBudgetPool:
    """Thread-safe dynamic core budget manager for parallel task execution.

    When many tasks are queued, tasks lease 1 core each to maximize task
    parallelism. When remaining task count drops below the total core count
    (tail/straggler phase), remaining tasks lease multiple cores up to
    `max_cores_per_task`, preventing idle-core waste.
    """

    def __init__(
        self,
        total_cores: int,
        total_tasks: int = 1,
        max_cores_per_task: int | None = None,
    ) -> None:
        self.total_cores = max(1, int(total_cores))
        self._remaining_tasks = max(1, int(total_tasks))
        self.max_cores_per_task = (
            max(1, int(max_cores_per_task))
            if max_cores_per_task is not None
            else self.total_cores
        )
        self._available_cores = self.total_cores
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining_tasks

    @property
    def available_cores(self) -> int:
        with self._lock:
            return self._available_cores

    def task_completed(self) -> None:
        """Decrement the count of remaining uncompleted tasks."""
        with self._cond:
            self._remaining_tasks = max(0, self._remaining_tasks - 1)
            self._cond.notify_all()

    def lease_cores(self, file_size: int = 0) -> int:
        preferred = self.estimate_preferred_cores(file_size)
        min_acceptable = min(preferred, max(1, self.total_cores // 2)) if preferred >= 4 else 1

        with self._cond:
            while self._available_cores < min_acceptable:
                self._cond.wait()

            fair_share = max(1, self.total_cores // max(1, self._remaining_tasks))
            target = max(fair_share, preferred)

            allocated = min(self._available_cores, target, self.max_cores_per_task)
            allocated = max(1, allocated)

            self._available_cores -= allocated
            return allocated

    def release_cores(self, count: int) -> None:
        """Return leased cores back to the pool and notify waiting workers."""
        with self._cond:
            self._available_cores = min(self.total_cores, self._available_cores + count)
            self._cond.notify_all()

    @contextmanager
    def lease(self, file_size: int = 0) -> Iterator[int]:
        """Context manager for acquiring and safely releasing a core lease."""
        allocated = self.lease_cores(file_size=file_size)
        try:
            yield allocated
        finally:
            self.release_cores(allocated)

    @staticmethod
    def estimate_preferred_cores(file_size_bytes: int) -> int:
        """Heuristic suggesting thread count based on unaligned FASTA file size."""
        if file_size_bytes < 2 * 1024 * 1024:  # < 2 MB: 1 core
            return 1
        elif file_size_bytes < 10 * 1024 * 1024:  # 2 - 10 MB: 2 cores
            return 2
        elif file_size_bytes < 25 * 1024 * 1024:  # 10 - 25 MB: 4 cores
            return 4
        else:  # > 25 MB: up to 8 cores
            return 8
