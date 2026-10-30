"""Concurrency utilities, task queues, and dynamic core budget management."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.context import BaseContext


def get_mp_context(force_context: str | None = None) -> BaseContext:
    """Returns an appropriate multiprocessing context based on platform and configuration."""
    if force_context:
        return multiprocessing.get_context(force_context)

    env_override = os.environ.get("PBA_MP_CONTEXT")
    if env_override:
        return multiprocessing.get_context(env_override)

    if sys.platform in ("win32", "darwin"):
        return multiprocessing.get_context("spawn")

    return multiprocessing.get_context("fork")


class CoreBudgetPool:
    """Thread-safe dynamic core budget manager with mathematical water-filling makespan balancing.

    Solves the Malleable Task Scheduling Problem (P | malleable | C_max) with non-linear
    Amdahl speedup S(p, L). Loci that would otherwise become critical-path tail stragglers
    are assigned multi-core leases to finish concurrently with the single-core pool,
    targeting near-100% core utilization across arbitrary hardware deployments.
    """

    def __init__(
        self,
        total_cores: int,
        total_tasks: int = 1,
        max_cores_per_task: int | None = None,
        schedule: dict[str, int] | None = None,
    ) -> None:
        self.total_cores = max(1, int(total_cores))
        self._remaining_tasks = max(1, int(total_tasks))
        self.max_cores_per_task = (
            max(1, int(max_cores_per_task))
            if max_cores_per_task is not None
            else self.total_cores
        )
        self._schedule = dict(schedule) if schedule else {}
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
        with self._cond:
            self._remaining_tasks = max(0, self._remaining_tasks - 1)
            self._cond.notify_all()

    @staticmethod
    def solve_water_filling_schedule(
        tasks_meta: dict[str, tuple[int, int]],
        total_cores: int,
        alpha: float = 0.85,
    ) -> dict[str, int]:
        """Calculates optimal core allocation per locus via mathematical makespan balancing.

        Accounts for super-quadratic complexity and cache-miss overhead in ultra-long,
        divergent sequences (L >= 1800 aa), allocating cores to true critical stragglers
        while preserving single-core efficiency for standard loci.
        """
        if not tasks_meta or total_cores <= 1:
            return {k: 1 for k in tasks_meta}

        K = len(tasks_meta)
        C = total_cores

        work: dict[str, float] = {}
        p_max: dict[str, int] = {}

        for k, (n, l) in tasks_meta.items():
            n_eff = max(1, n)
            l_eff = max(1, l)

            # Super-quadratic complexity scaling for ultra-long unanchored loci
            if l_eff >= 1800:
                w = float(n_eff * (l_eff**2.4))
                ceiling = min(8, C)
            elif l_eff >= 1200:
                w = float(n_eff * (l_eff**2.1))
                ceiling = min(4, C)
            elif l_eff >= 600:
                w = float(n_eff * (l_eff**2.0))
                ceiling = min(2, C)
            else:
                w = float(n_eff * (l_eff**2.0))
                ceiling = 1
            work[k] = w
            p_max[k] = max(1, ceiling)

        total_work = sum(work.values())
        if total_work <= 0:
            return {k: 1 for k in tasks_meta}

        # Sparse Workload Regime (K < C)
        if K < C:
            work_scaled = {k: work[k] ** (1.0 / alpha) for k in work}
            sum_scaled = sum(work_scaled.values())
            allocations = {
                k: max(1, min(p_max[k], round((work_scaled[k] / sum_scaled) * C)))
                for k in work
            }
            assigned = sum(allocations.values())
            sorted_keys = sorted(work.keys(), key=lambda x: work[x], reverse=True)
            while assigned < C:
                progress = False
                for k in sorted_keys:
                    if allocations[k] < p_max[k] and assigned < C:
                        allocations[k] += 1
                        assigned += 1
                        progress = True
                if not progress:
                    break
            while assigned > C:
                for k in reversed(sorted_keys):
                    if allocations[k] > 1 and assigned > C:
                        allocations[k] -= 1
                        assigned -= 1
                        if assigned == C:
                            break
            return allocations

        # Dense Workload Regime (K >= C)
        ideal_makespan = total_work / C
        sorted_keys = sorted(work.keys(), key=lambda x: work[x], reverse=True)

        # True critical stragglers must significantly exceed ideal makespan
        outliers = [
            k for k in sorted_keys
            if work[k] > (1.2 * ideal_makespan) and p_max[k] > 1
        ]

        allocations = {k: 1 for k in work}
        if not outliers:
            return allocations

        # Allow critical stragglers to access full node capacity
        cores_remaining = C
        for k in outliers:
            wanted = p_max[k]
            grant = min(wanted, cores_remaining)
            if grant >= 8:
                p = 8
            elif grant >= 4:
                p = 4
            elif grant >= 2:
                p = 2
            else:
                p = 1

            allocations[k] = p
            cores_remaining = max(0, cores_remaining - p)
            if cores_remaining < 2:
                break

        return allocations

    @staticmethod
    def estimate_preferred_cores(file_size_bytes: int) -> int:
        """Fallback static sizing heuristic for standalone unit tests."""
        if file_size_bytes < 2 * 1024 * 1024:
            return 1
        elif file_size_bytes < 10 * 1024 * 1024:
            return 2
        elif file_size_bytes < 25 * 1024 * 1024:
            return 4
        else:
            return 16

    def lease_cores(self, file_size: int = 0, task_id: str | None = None) -> int:
        if task_id is not None and task_id in self._schedule:
            preferred = self._schedule[task_id]
        else:
            preferred = self.estimate_preferred_cores(file_size)

        with self._cond:
            # Solo straggler optimization: lone remaining locus claims all available capacity
            if self._remaining_tasks <= 1:
                allocated = min(self._available_cores, self.max_cores_per_task)
                allocated = max(1, allocated)
                self._available_cores -= allocated
                return allocated

            # Dynamic fair share expands as remaining tasks drop below core count
            fair_share = max(1, self.total_cores // max(1, self._remaining_tasks))
            target = max(fair_share, preferred)

            # Minimum cores we are willing to accept before proceeding:
            # Standard tasks wait only when saturated (0 cores available).
            # Multi-core scheduled tasks wait until their quota (or at least half) is free.
            min_needed = (
                target
                if (preferred > 1 and self._remaining_tasks >= self.total_cores)
                else 1
            )
            while self._available_cores < min_needed:
                if preferred > 1 and self._available_cores >= max(2, target // 2):
                    break
                self._cond.wait()

            allocated = min(self._available_cores, target, self.max_cores_per_task)
            allocated = max(1, allocated)
            self._available_cores -= allocated
            return allocated

    def release_cores(self, count: int) -> None:
        with self._cond:
            self._available_cores = min(
                self.total_cores, self._available_cores + count
            )
            self._cond.notify_all()

    @contextmanager
    def lease(
        self, file_size: int = 0, task_id: str | None = None
    ) -> Iterator[int]:
        allocated = self.lease_cores(file_size=file_size, task_id=task_id)
        try:
            yield allocated
        finally:
            self.release_cores(allocated)
