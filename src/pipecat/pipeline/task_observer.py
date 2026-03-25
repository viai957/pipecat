#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Task observer for managing pipeline frame observers.

This module provides a proxy observer system that manages multiple observers
for pipeline frame events, ensuring that observer processing doesn't block
the main pipeline execution.
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from attr import dataclass

from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.utils.asyncio.process_queue import ProcessQueue
from pipecat.utils.asyncio.task_manager import BaseTaskManager

# Maximum number of items in each observer proxy queue. Bounded to prevent
# unbounded growth when observers are slower than the pipeline.
_PROXY_QUEUE_MAXSIZE = 50


@dataclass
class Proxy:
    """Proxy data for managing observer tasks and queues.

    This represents is the data received from the main observer that
    is queued for later processing.

    Parameters:
        queue: Queue for frame data awaiting observer processing.
        task: Asyncio task running the observer's frame processing loop.
        observer: The actual observer instance being proxied.
    """

    queue: ProcessQueue
    task: asyncio.Task
    observer: BaseObserver


class TaskObserver(BaseObserver):
    """Proxy observer that manages multiple observers without blocking the pipeline.

    This is a pipeline frame observer that is meant to be used as a proxy to
    the user provided observers. That is, this is the observer that should be
    passed to the frame processors. Then, every time a frame is pushed this
    observer will call all the observers registered to the pipeline task.

    This observer makes sure that passing frames to observers doesn't block the
    pipeline by creating a queue and a task for each user observer. When a frame
    is received, it will be put in a queue for efficiency and later processed by
    each task.
    """

    def __init__(
        self,
        *,
        observers: Optional[List[BaseObserver]] = None,
        task_manager: BaseTaskManager,
        **kwargs,
    ):
        """Initialize the TaskObserver.

        Args:
            observers: List of observers to manage. Defaults to empty list.
            task_manager: Task manager for creating and managing observer tasks.
            **kwargs: Additional arguments passed to the base observer.
        """
        super().__init__(**kwargs)
        self._observers = observers or []
        self._task_manager = task_manager
        self._proxies: Optional[Dict[BaseObserver, Proxy]] = (
            None  # Becomes a dict after start() is called
        )
        # Initialize observer flags. Pre-compute them now if observers were
        # provided at construction time so that FrameProcessor.setup() (which
        # runs before start()) reads the correct values.
        self.has_process_frame_observers: bool = False
        self.has_push_frame_observers: bool = False
        self._any_push_subscribed: Optional[frozenset] = None
        if self._observers:
            self._recompute_observer_flags()

    def add_observer(self, observer: BaseObserver):
        """Add a new observer to the managed list.

        Args:
            observer: The observer to add.
        """
        # Add the observer to the list.
        self._observers.append(observer)
        self._recompute_observer_flags()

        # If we already started, create a new proxy for the observer.
        # Otherwise, it will be created in start().
        if self._proxies:
            proxy = self._create_proxy(observer)
            self._proxies[observer] = proxy

    async def remove_observer(self, observer: BaseObserver):
        """Remove an observer and clean up its resources.

        Args:
            observer: The observer to remove.
        """
        # If the observer has a proxy, remove it.
        if self._proxies and observer in self._proxies:
            proxy = self._proxies[observer]
            # Remove the proxy so it doesn't get called anymore.
            del self._proxies[observer]
            # Cancel the proxy task right away.
            await self._task_manager.cancel_task(proxy.task)

        # Remove the observer from the list.
        if observer in self._observers:
            self._observers.remove(observer)
        self._recompute_observer_flags()

    async def start(self):
        """Start all proxy observer tasks."""
        self._proxies = self._create_proxies(self._observers)
        self._recompute_observer_flags()

    async def stop(self):
        """Stop all proxy observer tasks."""
        if not self._proxies:
            return

        for proxy in self._proxies.values():
            await self._task_manager.cancel_task(proxy.task)

    async def cleanup(self):
        """Cleanup all proxy observers."""
        await super().cleanup()

        if not self._proxies:
            return

        for proxy in self._proxies:
            await proxy.cleanup()

    async def on_process_frame(self, data: FrameProcessed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    async def on_push_frame(self, data: FramePushed):
        """Queue frame data for all managed observers.

        Args:
            data: The frame push event data to distribute to observers.
        """
        await self._send_to_proxy(data)

    def _recompute_observer_flags(self):
        """Recompute has_process_frame_observers / has_push_frame_observers,
        and the per-type subscription union used by is_push_interested().

        Uses method-identity comparison (O(N) over observers) to detect
        overrides. Called on start(), add_observer(), and remove_observer()
        so FrameProcessor hot-path checks stay accurate without rechecking
        on every frame.
        """
        self.has_process_frame_observers = any(
            type(obs).on_process_frame is not BaseObserver.on_process_frame
            for obs in self._observers
        )
        self.has_push_frame_observers = any(
            type(obs).on_push_frame is not BaseObserver.on_push_frame
            for obs in self._observers
        )

        # Compute the union of all push_frame_types subscriptions.
        # If any observer has push_frame_types=None (subscribe to all), the
        # union is None → every frame type must be dispatched.
        union: Optional[set] = set()
        for obs in self._observers:
            ft = obs.push_frame_types
            if ft is None:
                union = None  # At least one observer wants all frames
                break
            union.update(ft)
        self._any_push_subscribed: Optional[frozenset] = frozenset(union) if union is not None else None

    def is_push_interested(self, type_id: int) -> bool:
        """Return True if any proxy observer is interested in this frame type.

        When all proxied observers have declared push_frame_types subscriptions,
        this is an O(1) frozenset lookup. Falls through to True (dispatch) for
        any type_id in the union or when any observer subscribes to all frames.

        Args:
            type_id: The integer value of the frame's FrameType enum.
        """
        return self._any_push_subscribed is None or type_id in self._any_push_subscribed

    def _create_proxy(self, observer: BaseObserver) -> Proxy:
        """Create a proxy for a single observer.

        Uses a ProcessQueue (deque+Event) with manual size-limiting to prevent
        unbounded growth when observers are slower than the pipeline. ProcessQueue
        is ~14x faster than asyncio.Queue for put_nowait (~35ns vs ~490ns).
        """
        queue = ProcessQueue()
        task = self._task_manager.create_task(
            self._proxy_task_handler(queue, observer),
            f"TaskObserver::{observer}::_proxy_task_handler",
        )
        proxy = Proxy(queue=queue, task=task, observer=observer)
        return proxy

    def _create_proxies(self, observers: List[BaseObserver]) -> Dict[BaseObserver, Proxy]:
        """Create proxies for all observers."""
        proxies = {}
        for observer in observers:
            proxy = self._create_proxy(observer)
            proxies[observer] = proxy
        return proxies

    async def _send_to_proxy(self, data: Any):
        for proxy in self._proxies.values():
            q = proxy.queue
            if q.qsize() >= _PROXY_QUEUE_MAXSIZE:
                # Drop oldest to make room — observers should not block the pipeline.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(data)

    async def _proxy_task_handler(self, queue: ProcessQueue, observer: BaseObserver):
        """Handle frame processing for a single observer."""
        on_push_frame_deprecated = False
        signature = inspect.signature(observer.on_push_frame)
        if len(signature.parameters) > 1:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Observer `on_push_frame(source, destination, frame, direction, timestamp)` is deprecated, us `on_push_frame(data: FramePushed)` instead.",
                    DeprecationWarning,
                )

            on_push_frame_deprecated = True

        while True:
            data = await queue.get()

            if isinstance(data, FramePushed):
                if on_push_frame_deprecated:
                    await observer.on_push_frame(
                        data.source, data.destination, data.frame, data.direction, data.timestamp
                    )
                else:
                    await observer.on_push_frame(data)
            elif isinstance(data, FrameProcessed):
                await observer.on_process_frame(data)
