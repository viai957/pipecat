#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Backpressure and bounded queue system for pipeline scalability.

At scale (3000+ concurrent pipelines), unbounded asyncio.Queue instances
lead to memory exhaustion under burst load. This module provides bounded
queues with frame-type-aware drop policies:

- SystemFrame: NEVER dropped (unbounded bypass channel)
- UninterruptibleFrame: NEVER dropped
- DataFrame: Bounded with configurable drop policy (DROP_OLDEST, DROP_NEWEST, BLOCK)
- ControlFrame: Bounded with BLOCK policy

The BoundedFrameQueue splits incoming frames into a system channel (unbounded,
always drained first) and a data channel (bounded deque with drop policy).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from loguru import logger

from pipecat.frames.frames import Frame


class DropPolicy(Enum):
    """Policy for handling full queues."""

    NEVER = "never"
    DROP_OLDEST = "oldest"
    DROP_NEWEST = "newest"
    BLOCK = "block"


@dataclass
class QueueConfig:
    """Configuration for a single bounded queue.

    Args:
        max_size: Maximum number of items in the data channel.
        drop_policy: What to do when the queue is full.
        warn_threshold: Log a warning when queue reaches this fraction of max_size.
    """

    max_size: int = 100
    drop_policy: DropPolicy = DropPolicy.DROP_OLDEST
    warn_threshold: float = 0.8


@dataclass
class BackpressureConfig:
    """Per-queue backpressure configuration for a pipeline.

    When passed as `backpressure` in PipelineParams, these configs are
    propagated to all frame processors and transport queues. When None
    (the default), large defaults are used that behave as effectively unbounded.

    Args:
        input_queue: Config for FrameProcessor input queues.
        process_queue: Config for FrameProcessor process (data) queues.
        push_queue: Config for PipelineTask push queue.
        audio_queue: Config for MediaSender audio queue.
        video_queue: Config for MediaSender video queue.
        observer_queue: Config for TaskObserver proxy queues.
    """

    input_queue: QueueConfig = field(
        default_factory=lambda: QueueConfig(max_size=200, drop_policy=DropPolicy.DROP_OLDEST)
    )
    process_queue: QueueConfig = field(
        default_factory=lambda: QueueConfig(max_size=100, drop_policy=DropPolicy.DROP_OLDEST)
    )
    push_queue: QueueConfig = field(
        default_factory=lambda: QueueConfig(max_size=500, drop_policy=DropPolicy.DROP_OLDEST)
    )
    audio_queue: QueueConfig = field(
        default_factory=lambda: QueueConfig(max_size=50, drop_policy=DropPolicy.DROP_OLDEST)
    )
    video_queue: QueueConfig = field(
        default_factory=lambda: QueueConfig(max_size=10, drop_policy=DropPolicy.DROP_OLDEST)
    )
    observer_queue: QueueConfig = field(
        default_factory=lambda: QueueConfig(max_size=50, drop_policy=DropPolicy.DROP_OLDEST)
    )


# Default config used when backpressure=None (effectively unbounded)
_UNBOUNDED_CONFIG = QueueConfig(max_size=10000, drop_policy=DropPolicy.DROP_OLDEST)


@dataclass
class QueueMetrics:
    """Lightweight per-queue metrics for observability."""

    total_enqueued: int = 0
    total_dropped: int = 0
    high_water_mark: int = 0

    def record_enqueue(self, current_size: int):
        self.total_enqueued += 1
        if current_size > self.high_water_mark:
            self.high_water_mark = current_size

    def record_drop(self):
        self.total_dropped += 1


class BoundedFrameQueue:
    """Bounded queue with SystemFrame bypass channel.

    Architecture:
        - System channel: unbounded deque for SystemFrame — never dropped or blocked.
          (Previously asyncio.Queue; replaced with deque because get() waits on the
          shared _has_items Event, making asyncio.Queue's internal Future/wakeup
          machinery pure overhead — ~150 ns per put + ~150 ns per get saved.)
        - Data channel: bounded deque for DataFrame and ControlFrame — subject to
          drop policy when full.
        - A single asyncio.Event signals when any item is available.

    The get() method prioritizes the system channel, ensuring SystemFrames
    are always processed before data frames.
    """

    def __init__(self, config: Optional[QueueConfig] = None, name: str = ""):
        cfg = config or _UNBOUNDED_CONFIG
        self._max_size = cfg.max_size
        self._drop_policy = cfg.drop_policy
        self._warn_threshold = cfg.warn_threshold
        self._name = name

        # System channel: plain deque (unbounded, never drops, always prioritized).
        # Previously asyncio.Queue, but BoundedFrameQueue.get() waits on the
        # shared _has_items Event rather than asyncio.Queue's internal Future
        # machinery — so the Queue's _wakeup_next / _unfinished_tasks bookkeeping
        # was pure wasted overhead (~150 ns per put + ~150 ns per get).
        self._sys_deque: deque = deque()

        # Data channel: bounded deque
        self._data_deque: deque = deque()
        self._data_event = asyncio.Event()

        # Combined notification: set when either channel has items
        self._has_items = asyncio.Event()

        self._metrics = QueueMetrics()
        self._warned = False
        # Precomputed warn threshold to avoid per-enqueue multiplication.
        self._warn_threshold_abs: int = int(self._max_size * self._warn_threshold)
        # True only for BLOCK policy — guards _data_event.set() calls in get()
        # so DROP_OLDEST/DROP_NEWEST paths skip the Event.set() overhead (~50 ns).
        self._has_block_policy: bool = cfg.drop_policy == DropPolicy.BLOCK

    async def put(self, item: Tuple[Frame, Any, Any]):
        """Enqueue a frame item.

        SystemFrame goes to the system channel (never dropped, always prioritized).
        UninterruptibleFrame goes to the data channel but is protected from dropping.
        All other frames go to the bounded data channel with drop policy applied.
        """
        frame = item[0]

        if frame.is_system_frame:
            self._sys_deque.append(item)
            self._has_items.set()
            self._metrics.record_enqueue(len(self._sys_deque))
            return

        # UninterruptibleFrame goes to data channel (FIFO order preserved)
        # but is never dropped — skip the drop policy for these frames.
        if frame.is_uninterruptible:
            self._data_deque.append(item)
            self._has_items.set()
            self._metrics.record_enqueue(len(self._data_deque))
            return

        # Data channel with backpressure
        if len(self._data_deque) >= self._max_size:
            if self._drop_policy == DropPolicy.DROP_OLDEST:
                dropped = self._data_deque.popleft()
                self._metrics.record_drop()
                if not self._warned:
                    logger.warning(
                        f"BoundedFrameQueue({self._name}): dropping oldest frame "
                        f"(size={self._max_size}, dropped_total={self._metrics.total_dropped})"
                    )
                    self._warned = True
            elif self._drop_policy == DropPolicy.DROP_NEWEST:
                self._metrics.record_drop()
                if not self._warned:
                    logger.warning(
                        f"BoundedFrameQueue({self._name}): dropping newest frame "
                        f"(size={self._max_size}, dropped_total={self._metrics.total_dropped})"
                    )
                    self._warned = True
                return  # Don't enqueue the new item
            elif self._drop_policy == DropPolicy.BLOCK:
                # Wait until space is available
                while len(self._data_deque) >= self._max_size:
                    self._data_event.clear()
                    await self._data_event.wait()
            elif self._drop_policy == DropPolicy.NEVER:
                pass  # Allow unbounded growth (system frames use this implicitly)

        self._data_deque.append(item)
        self._has_items.set()
        current_size = len(self._data_deque)
        self._metrics.record_enqueue(current_size)

        # Threshold warning (once)
        if not self._warned and current_size >= self._max_size * self._warn_threshold:
            logger.warning(
                f"BoundedFrameQueue({self._name}): {current_size}/{self._max_size} "
                f"({current_size / self._max_size:.0%} full)"
            )

    async def get(self) -> Any:
        """Dequeue the next frame. System frames are always prioritized.

        Returns:
            The next (frame, direction, callback) tuple.
        """
        while True:
            # Priority 1: system frames
            if self._sys_deque:
                return self._sys_deque.popleft()

            # Priority 2: data frames
            if self._data_deque:
                item = self._data_deque.popleft()
                if self._has_block_policy:
                    self._data_event.set()  # Signal space available for BLOCK policy
                return item

            # Nothing available — clear event THEN re-check to avoid TOCTOU race.
            # A put() between our checks and clear() would set the event, and
            # clearing after would lose the signal. Re-checking after clear fixes this.
            self._has_items.clear()

            # Re-check after clearing to prevent race with concurrent put()
            if self._sys_deque or self._data_deque:
                self._has_items.set()
                continue

            await self._has_items.wait()

    def get_nowait(self) -> Any:
        """Non-blocking get. Raises if empty."""
        if self._sys_deque:
            return self._sys_deque.popleft()
        if self._data_deque:
            item = self._data_deque.popleft()
            if self._has_block_policy:
                self._data_event.set()
            return item
        raise asyncio.QueueEmpty()

    def enqueue_sync(self, item: Tuple[Frame, Any, Any]):
        """Enqueue a frame synchronously without coroutine allocation overhead.

        Semantically equivalent to ``await put(item)`` for DROP_OLDEST and
        DROP_NEWEST policies — the common case. Saves ~95 ns per call vs
        ``await put()`` by avoiding coroutine creation.

        Do NOT call with BLOCK policy when the queue is at capacity; use
        ``await put()`` instead so the caller can yield until space is free.
        """
        frame = item[0]

        if frame.is_system_frame:
            self._sys_deque.append(item)
            self._has_items.set()
            self._metrics.record_enqueue(len(self._sys_deque))
            return

        if frame.is_uninterruptible:
            self._data_deque.append(item)
            self._has_items.set()
            self._metrics.record_enqueue(len(self._data_deque))
            return

        # DataFrame — apply drop policy when the bounded channel is full.
        # Cache the deque reference to avoid repeated LOAD_ATTR in the hot path.
        dq = self._data_deque
        n = len(dq)
        if n >= self._max_size:
            if self._drop_policy == DropPolicy.DROP_OLDEST:
                dq.popleft()
                self._metrics.record_drop()
                if not self._warned:
                    logger.warning(
                        f"BoundedFrameQueue({self._name}): dropping oldest frame "
                        f"(size={self._max_size}, dropped_total={self._metrics.total_dropped})"
                    )
                    self._warned = True
            elif self._drop_policy == DropPolicy.DROP_NEWEST:
                self._metrics.record_drop()
                if not self._warned:
                    logger.warning(
                        f"BoundedFrameQueue({self._name}): dropping newest frame "
                        f"(size={self._max_size}, dropped_total={self._metrics.total_dropped})"
                    )
                    self._warned = True
                return  # Don't enqueue the new item
            # BLOCK: caller should use async put(); fall through and enqueue anyway
            # to avoid dropping — the caller is responsible for rate-limiting.

        dq.append(item)
        # Only signal when transitioning from empty → non-empty; avoids a Python
        # function-call (~60 ns) on every enqueue when the deque already has items.
        if n == 0:
            self._has_items.set()
        n += 1  # avoid a second len() call — n was captured before the append
        # Inline record_enqueue() to eliminate the method-call overhead (~50 ns).
        m = self._metrics
        m.total_enqueued += 1
        if n > m.high_water_mark:
            m.high_water_mark = n

        # Check the fast int comparison first so the LOAD_ATTR for _warned is
        # short-circuited in the common case (n < threshold).
        if n >= self._warn_threshold_abs and not self._warned:
            logger.warning(
                f"BoundedFrameQueue({self._name}): {n}/{self._max_size} "
                f"({n / self._max_size:.0%} full)"
            )

    def put_nowait(self, item: Tuple[Frame, Any, Any]):
        """Non-blocking put (no drop policy, for internal use like reset)."""
        frame = item[0]
        if frame.is_system_frame:
            self._sys_deque.append(item)
        else:
            self._data_deque.append(item)
        self._has_items.set()

    def empty(self) -> bool:
        """Check if both channels are empty."""
        return not self._sys_deque and not self._data_deque

    def qsize(self) -> int:
        """Total items across both channels."""
        return len(self._sys_deque) + len(self._data_deque)

    def task_done(self):
        """Compatibility with asyncio.Queue task tracking."""
        pass

    @property
    def stats(self) -> dict:
        """Return queue metrics for observability."""
        return {
            "name": self._name,
            "system_size": len(self._sys_deque),
            "data_size": len(self._data_deque),
            "max_size": self._max_size,
            "total_enqueued": self._metrics.total_enqueued,
            "total_dropped": self._metrics.total_dropped,
            "high_water_mark": self._metrics.high_water_mark,
            "drop_policy": self._drop_policy.value,
        }
