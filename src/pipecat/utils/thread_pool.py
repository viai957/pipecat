#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared thread pool for CPU-bound work across all pipelines.

At scale (3000+ concurrent pipelines), creating a ThreadPoolExecutor per
pipeline exhausts OS thread limits. This module provides a singleton shared
pool that all pipelines use for CPU-bound tasks like image resizing and
VAD inference.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger


class SharedThreadPool:
    """Process-wide singleton ThreadPoolExecutor.

    Uses double-checked locking for thread-safe lazy initialization.
    Pool size is configurable via PIPECAT_THREAD_POOL_WORKERS env var.
    """

    _instance: Optional[ThreadPoolExecutor] = None
    _lock = threading.Lock()

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get the shared ThreadPoolExecutor, creating it on first call.

        Returns:
            The shared ThreadPoolExecutor instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    workers = cls._default_workers()
                    cls._instance = ThreadPoolExecutor(max_workers=workers)
                    logger.debug(f"SharedThreadPool initialized with {workers} workers")
        return cls._instance

    @classmethod
    def shutdown(cls):
        """Shutdown the shared pool. Call only during process teardown."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown(wait=False)
                cls._instance = None
                logger.debug("SharedThreadPool shut down")

    @staticmethod
    def _default_workers() -> int:
        """Determine worker count from env or sensible default."""
        env_val = os.environ.get("PIPECAT_THREAD_POOL_WORKERS")
        if env_val:
            return int(env_val)
        cpu = os.cpu_count() or 4
        return min(32, cpu + 4)
