#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimal FIFO queue (deque + asyncio.Event) for high-throughput frame pipelines.

Replaces asyncio.Queue in per-processor and per-task hot paths to eliminate the
internal bookkeeping overhead of asyncio.Queue:
  - asyncio.Queue.put_nowait: ~85 ns  (_put, _unfinished_tasks, _wakeup_next)
  - asyncio.Queue.get_nowait: ~70 ns  (_get, _wakeup_next on putters)
vs deque operations:
  - deque.append + conditional Event.set: ~35 ns
  - deque.popleft: ~15 ns

For the unbounded, non-blocking (DROP_OLDEST / best-effort) queue use case,
asyncio.Queue's blocking put/get machinery is never exercised, so the overhead
is pure waste.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any


class ProcessQueue:
    """Minimal FIFO queue (deque + asyncio.Event) drop-in for asyncio.Queue.

    Suitable for unbounded, non-blocking use cases (no BLOCK policy needed).
    Thread-safe within a single asyncio event loop (same-loop single-threaded use).

    The get() and put_nowait() / put() are TOCTOU-safe via the clear-then-recheck
    pattern: the event is cleared first, then the deque is checked again before
    suspending, so a concurrent put() that fires between the deque check and the
    event clear is never lost.
    """

    __slots__ = ("_dq", "_event")

    def __init__(self):
        self._dq: deque = deque()
        self._event = asyncio.Event()

    def put_nowait(self, item: Any) -> None:
        was_empty = not self._dq
        self._dq.append(item)
        if was_empty:
            self._event.set()

    async def put(self, item: Any) -> None:
        """Async put — equivalent to put_nowait for unbounded queues."""
        self.put_nowait(item)

    async def get(self) -> Any:
        while True:
            if self._dq:
                return self._dq.popleft()
            self._event.clear()
            if self._dq:  # re-check after clear to avoid TOCTOU race with put_nowait
                self._event.set()
                continue
            await self._event.wait()

    def get_nowait(self) -> Any:
        if self._dq:
            return self._dq.popleft()
        raise asyncio.QueueEmpty()

    def empty(self) -> bool:
        return not self._dq

    def qsize(self) -> int:
        return len(self._dq)

    def task_done(self) -> None:
        pass  # Compatibility — no join() users
