"""
Ephemeral RAM-Disk Engine for PlastBurstAlign.

Manages high-throughput temporary scratch directories on tmpfs mounts (/dev/shm)
with automated safety margins, fallback ladders, and guaranteed unlinking.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from .logging_ops import logger as log

DEFAULT_MIN_HEADROOM_BYTES = 500 * 1024 * 1024  # 500 MB safe margin


class EphemeralScratchManager:
    """Manages ephemeral memory-backed scratch spaces with guaranteed process isolation."""

    def __init__(
        self,
        custom_scratch_dir: str | Path | None = None,
        min_headroom_bytes: int = DEFAULT_MIN_HEADROOM_BYTES,
    ) -> None:
        self.min_headroom_bytes = min_headroom_bytes
        self.scratch_path: Path | None = None
        self._is_ramdisk = False
        self._cleaned = False

        self._initialize_scratch(custom_scratch_dir)
        atexit.register(self.cleanup)

    def _initialize_scratch(self, custom_scratch_dir: str | Path | None) -> None:
        """Determines the optimal scratch location across memory and disk tiers."""
        unique_suffix = f"pba_{os.getpid()}_{uuid.uuid4().hex[:8]}"

        # Tier 1: User-specified custom scratch path
        if custom_scratch_dir:
            base = Path(custom_scratch_dir).expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            self.scratch_path = base / unique_suffix
            self.scratch_path.mkdir(parents=True, exist_ok=True)
            self._is_ramdisk = self._check_if_tmpfs(self.scratch_path)
            log.info(f"Initialized custom scratch at `{self.scratch_path}` (RAM-disk: {self._is_ramdisk})")
            return

        # Tier 2: System RAM-disk (/dev/shm on Linux)
        shm_path = Path("/dev/shm")
        if shm_path.is_dir() and os.access(shm_path, os.W_OK):
            try:
                usage = shutil.disk_usage(shm_path)
                if usage.free >= self.min_headroom_bytes:
                    self.scratch_path = shm_path / unique_suffix
                    self.scratch_path.mkdir(parents=True, exist_ok=True)
                    self._is_ramdisk = True
                    avail_gb = usage.free / (1024**3)
                    log.info(
                        f"Allocated ephemeral RAM-disk scratch: `{self.scratch_path}` "
                        f"({avail_gb:.1f} GB tmpfs available)"
                    )
                    return
                else:
                    log.warning(
                        f"/dev/shm has insufficient free space ({usage.free // (1024*1024)} MB < "
                        f"{self.min_headroom_bytes // (1024*1024)} MB). Falling back to system temp."
                    )
            except OSError as e:
                log.warning(f"Unable to query /dev/shm disk usage: {e}. Falling back to system temp.")

        # Tier 3: Standard system temporary directory fallback
        system_tmp = Path(tempfile.gettempdir()).resolve()
        self.scratch_path = system_tmp / unique_suffix
        self.scratch_path.mkdir(parents=True, exist_ok=True)
        self._is_ramdisk = False
        log.info(f"Allocated filesystem scratch: `{self.scratch_path}`")

    @staticmethod
    def _check_if_tmpfs(path: Path) -> bool:
        """Checks whether a given path is located on a tmpfs/RAM-disk filesystem."""
        try:
            stat = os.statvfs(path)
            # Standard Linux tmpfs magic value (0x01021994)
            return getattr(stat, "f_type", 0) == 0x01021994 or str(path).startswith("/dev/shm")
        except Exception:
            return str(path).startswith("/dev/shm")

    @property
    def is_ramdisk(self) -> bool:
        """Returns True if the active scratch path resides in RAM."""
        return self._is_ramdisk

    @property
    def path(self) -> Path:
        """Returns the active Path object for scratch operations."""
        if self.scratch_path is None or self._cleaned:
            raise RuntimeError("Scratch directory has not been initialized or was already cleaned up.")
        return self.scratch_path

    def get_subprocess_env(self) -> dict[str, str]:
        """Returns a copy of os.environ with TMPDIR redirected to ephemeral scratch."""
        env = os.environ.copy()
        env["TMPDIR"] = str(self.path)
        return env

    def cleanup(self) -> None:
        """Recursively removes the scratch directory tree."""
        if self._cleaned or self.scratch_path is None:
            return
        try:
            if self.scratch_path.exists():
                shutil.rmtree(self.scratch_path, ignore_errors=True)
                log.debug(f"Successfully unlinked ephemeral scratch `{self.scratch_path}`.")
        except Exception as e:
            log.warning(f"Error cleaning ephemeral scratch `{self.scratch_path}`: {e}")
        finally:
            self._cleaned = True

    def __enter__(self) -> EphemeralScratchManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
