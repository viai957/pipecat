#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processing pipeline infrastructure for Pipecat.

This module provides the core frame processing system that enables building
audio/video processing pipelines. It includes frame processors, pipeline
management, and frame flow control mechanisms.
"""

import asyncio
import dataclasses
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from loguru import logger

# Set to True to enable per-frame push trace logging. Off by default because
# loguru evaluates currentframe() on every logger.trace() call (~50 ns each),
# adding ~300 µs per 1000 frames (≈3.8% overhead) in the push_frame hot path.
_TRACE_FRAME_PUSHES: bool = False

# Sentinel: observer subscribes to ALL frame types (push_frame_types=None).
# Stored in _push_subscribed to distinguish "subscribe to all" from "subscribe
# to a specific frozenset of type_ids".
_PUSH_ALL: object = object()

from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy
from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frame_types import FrameType
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    FrameProcessorPauseFrame,
    FrameProcessorPauseUrgentFrame,
    FrameProcessorResumeFrame,
    FrameProcessorResumeUrgentFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
    StartFrame,
    SystemFrame,
    UninterruptibleFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.backpressure import BoundedFrameQueue, DropPolicy, QueueConfig
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.process_queue import ProcessQueue
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class FrameDirection(Enum):
    """Direction of frame flow in the processing pipeline.

    Parameters:
        DOWNSTREAM: Frames flowing from input to output.
        UPSTREAM: Frames flowing back from output to input.
    """

    DOWNSTREAM = 1
    UPSTREAM = 2


FrameCallback = Callable[["FrameProcessor", Frame, FrameDirection], Awaitable[None]]


@dataclass
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.

    Parameters:
        clock: The clock instance for timing operations.
        task_manager: The task manager for handling async operations.
        observer: Optional observer for monitoring frame processing events.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    observer: Optional[BaseObserver] = None


class FrameProcessorQueue(BoundedFrameQueue):
    """A bounded priority queue for system frames and other frames.

    Wraps BoundedFrameQueue to provide backward-compatible interface.
    SystemFrames are routed to an unbounded bypass channel (never dropped).
    DataFrames go to a bounded deque with configurable drop policy.
    """

    def __init__(self, config: Optional[QueueConfig] = None, name: str = "input"):
        """Initialize the FrameProcessorQueue.

        Args:
            config: Optional queue configuration. If None, uses large defaults
                    that behave as effectively unbounded (backward compatible).
            name: Name for logging/metrics identification.
        """
        super().__init__(config=config, name=name)


# Per-processor process queue: deque + asyncio.Event instead of asyncio.Queue.
# See ProcessQueue docstring for performance rationale.
_ProcessQueue = ProcessQueue


# Timeout in seconds for cancelling the input frame processing task.
# This prevents hanging if a library swallows asyncio.CancelledError.
INPUT_TASK_CANCEL_TIMEOUT_SECS = 3

# Timeout in seconds for cancelling the non-system (process) frame task.
# Without a timeout, __cancel_process_task() can hang indefinitely if a
# processor's task (e.g. LLM streaming, TTS WebSocket) is slow to cancel,
# which blocks InterruptionFrame propagation and causes the repeated
# "InterruptionFrame has not completed" warning loop.
PROCESS_TASK_CANCEL_TIMEOUT_SECS = 3


class FrameProcessor(BaseObject):
    """Base class for all frame processors in the pipeline.

    Frame processors are the building blocks of Pipecat pipelines, they can be
    linked to form complex processing pipelines. They receive frames, process
    them, and pass them to the next or previous processor in the chain.  Each
    frame processor guarantees frame ordering and processes frames in its own
    task. System frames are also processed in a separate task which guarantees
    frame priority.

    Event handlers available:

    - on_before_process_frame: Called before a frame is processed
    - on_after_process_frame: Called after a frame is processed
    - on_before_push_frame: Called before a frame is pushed
    - on_after_push_frame: Called after a frame is pushed
    - on_error: Called when an error is raised in the frame processing.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        enable_direct_mode: bool = False,
        metrics: Optional[FrameProcessorMetrics] = None,
        **kwargs,
    ):
        """Initialize the frame processor.

        Args:
            name: Optional name for this processor instance.
            enable_direct_mode: Whether to process frames immediately or use internal queues.
            metrics: Optional metrics collector for this processor.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self._prev: Optional["FrameProcessor"] = None
        self._next: Optional["FrameProcessor"] = None

        # Enable direct mode to skip queues and process frames right away.
        self._enable_direct_mode = enable_direct_mode

        # Clock
        self._clock: Optional[BaseClock] = None

        # Task Manager
        self._task_manager: Optional[BaseTaskManager] = None

        # Observer
        self._observer: Optional[BaseObserver] = None
        # Cached push subscription: None = no observer interested (ultra-fast path),
        # _PUSH_ALL sentinel = observer subscribes to all types, frozenset = specific types.
        # Updated once in setup() — avoids 3 attribute lookups + 1 method call per push_frame.
        self._push_subscribed: Any = None

        # Other properties
        self._enable_metrics = False
        self._enable_usage_metrics = False
        self._report_only_initial_ttfb = False
        # Other properties (deprecated)
        self._allow_interruptions = False
        self._interruption_strategies: List[BaseInterruptionStrategy] = []
        self._deprecated_openaillmcontext = False

        # Indicates whether we have received the StartFrame.
        self.__started = False

        # Cancellation is done through CancelFrame (a system frame). This could
        # cause other events being triggered (e.g. closing a transport) which
        # could also cause other frames to be pushed from other tasks
        # (e.g. EndFrame). So, when we are cancelling we don't want anything
        # else to be pushed.
        self._cancelling = False

        # Metrics
        self._metrics = metrics or FrameProcessorMetrics()
        self._metrics.set_processor_name(self.name)

        # Processors have an input priority queue which stores any type of
        # frames in order. System frames have higher priority than any other
        # frames, so they will be returned first from the queue.
        #
        # If a system frame is obtained it will be processed immediately any
        # other type of frame (data and control) will be put in a separate queue
        # for later processing. This guarantees that each frame processor will
        # always process system frames before any other frame in the queue.

        # The input task that handles all types of frames. It processes system
        # frames right away and queues non-system frames for later processing.
        self.__should_block_system_frames = False
        self.__input_queue = FrameProcessorQueue(name=f"{self.name}::input")
        self.__input_event: Optional[asyncio.Event] = None
        self.__input_frame_task: Optional[asyncio.Task] = None

        # The process task processes non-system frames.  Non-system frames will
        # be processed as soon as they are received by the processing task
        # (default) or they will block if `pause_processing_frames()` is
        # called. To resume processing frames we need to call
        # `resume_processing_frames()` which will wake up the event.
        self.__should_block_frames = False
        self.__process_queue = _ProcessQueue()
        self.__process_event: Optional[asyncio.Event] = None
        self.__process_frame_task: Optional[asyncio.Task] = None
        self.__process_current_frame: Optional[Frame] = None

        # Set while awaiting push_interruption_task_frame_and_wait() so that
        # _start_interruption() knows not to cancel the process task.
        self._wait_for_interruption = False

        # Frame processor events.
        self._register_event_handler("on_before_process_frame", sync=True)
        self._register_event_handler("on_after_process_frame", sync=True)
        self._register_event_handler("on_before_push_frame", sync=True)
        self._register_event_handler("on_after_push_frame", sync=True)
        self._register_event_handler("on_error", sync=True)

        # Cached flags for fast-path event handler skipping. When False (common
        # case), the hot paths skip _has_handlers() dict lookups (~83ns each).
        self._has_push_frame_handlers = False
        self._has_process_frame_handlers = False
        # Precomputed fast-path gate: True when started=True, no push handlers,
        # and no observer push subscription. Collapses 3 LOAD_ATTRs (~78 ns) in
        # push_frame to a single flag check (~26 ns), saving ~52 ns per call.
        # Updated in __start(), setup(), and add_event_handler().
        self._push_fast_ready: bool = False
        # Precomputed enqueue gate: True when not cancelling, not waiting for
        # interruption, and not in direct mode — the common case for regular
        # data-frame routing. Collapses 3 LOAD_ATTRs + 3 comparisons (~64 ns)
        # in _queue_frame_sync to 1 flag check (~13 ns), saving ~51 ns per hop.
        # Updated at the 4 sites that change the underlying flags.
        self._fast_enqueue_ready: bool = not enable_direct_mode

    _PUSH_FRAME_EVENTS = frozenset({"on_before_push_frame", "on_after_push_frame"})
    _PROCESS_FRAME_EVENTS = frozenset({"on_before_process_frame", "on_after_process_frame"})

    # When True, this processor is purely transparent for DOWNSTREAM data frames
    # (it only calls push_frame(frame, direction) and does no other work on data
    # frames).  _compute_fast_queue_target() will recurse through transparent
    # processors to find the first processor that actually owns a task queue,
    # allowing the fast-enqueue path to skip the transparent processor's task
    # entirely.  Non-data frames (control, transport, etc.) still flow through
    # the normal process_frame chain.
    _transparent_for_data_frames: ClassVar[bool] = False

    def add_event_handler(self, event_name: str, handler):
        """Add an event handler and update fast-path caches.

        Args:
            event_name: The name of the event to handle.
            handler: The function to call when the event occurs.
        """
        super().add_event_handler(event_name, handler)
        if event_name in self._PUSH_FRAME_EVENTS:
            self._has_push_frame_handlers = True
            self._push_fast_ready = False  # handlers registered; fast path disabled
        elif event_name in self._PROCESS_FRAME_EVENTS:
            self._has_process_frame_handlers = True

    @property
    def id(self) -> int:
        """Get the unique identifier for this processor.

        Returns:
            The unique integer ID of this processor.
        """
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this processor.

        Returns:
            The name of this processor instance.
        """
        return self._name

    @property
    def processors(self) -> List["FrameProcessor"]:
        """Return the list of sub-processors contained within this processor.

        Only compound processors (e.g. pipelines and parallel pipelines) have
        sub-processors. Non-compound processors will return an empty list.

        Returns:
            The list of sub-processors if this is a compound processor.
        """
        return []

    @property
    def entry_processors(self) -> List["FrameProcessor"]:
        """Return the list of entry processors for this processor.

        Entry processors are the first processors in a compound processor
        (e.g. pipelines, parallel pipelines). Note that pipelines can also be an
        entry processor as pipelines are processors themselves. Non-compound
        processors will simply return an empty list.

        Returns:
            The list of entry processors.
        """
        return []

    @property
    def next(self) -> Optional["FrameProcessor"]:
        """Get the next processor.

        Returns:
            The next processor, or None if there's no next processor.
        """
        return self._next

    @property
    def previous(self) -> Optional["FrameProcessor"]:
        """Get the previous processor.

        Returns:
            The previous processor, or None if there's no previous processor.
        """
        return self._prev

    @property
    def interruptions_allowed(self):
        """Check if interruptions are allowed for this processor.

        .. deprecated:: 0.0.99
            Use  `LLMUserAggregator`'s new `user_mute_strategies` parameter instead.

        Returns:
            True if interruptions are allowed.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`FrameProcessor.interruptions_allowed` is deprecated. "
                "Use  `LLMUserAggregator`'s new `user_mute_strategies` parameter instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        return self._allow_interruptions

    @property
    def metrics_enabled(self):
        """Check if metrics collection is enabled.

        Returns:
            True if metrics collection is enabled.
        """
        return self._enable_metrics

    @property
    def usage_metrics_enabled(self):
        """Check if usage metrics collection is enabled.

        Returns:
            True if usage metrics collection is enabled.
        """
        return self._enable_usage_metrics

    @property
    def report_only_initial_ttfb(self):
        """Check if only initial TTFB should be reported.

        Returns:
            True if only initial time-to-first-byte should be reported.
        """
        return self._report_only_initial_ttfb

    @property
    def interruption_strategies(self) -> Sequence[BaseInterruptionStrategy]:
        """Get the interruption strategies for this processor.

        .. deprecated:: 0.0.99
            This function is deprecated, use the new user and bot turn start
            strategies insted.

        Returns:
            Sequence of interruption strategies.
        """
        return self._interruption_strategies

    @property
    def task_manager(self) -> BaseTaskManager:
        """Get the task manager for this processor.

        Returns:
            The task manager instance.

        Raises:
            Exception: If the task manager is not initialized.
        """
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        return self._task_manager

    def processors_with_metrics(self):
        """Return processors that can generate metrics.

        Recursively collects all processors that support metrics generation,
        including those from nested processors.

        Returns:
            List of frame processors that can generate metrics.
        """
        return []

    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False

    def set_core_metrics_data(self, data: MetricsData):
        """Set core metrics data for this processor.

        Args:
            data: The metrics data to set.
        """
        self._metrics.set_core_metrics_data(data)

    async def start_ttfb_metrics(self, *, start_time: Optional[float] = None):
        """Start time-to-first-byte metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(
                start_time=start_time, report_only_initial_ttfb=self._report_only_initial_ttfb
            )

    async def stop_ttfb_metrics(self, *, end_time: Optional[float] = None):
        """Stop time-to-first-byte metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)

    async def start_processing_metrics(self, *, start_time: Optional[float] = None):
        """Start processing metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_processing_metrics(start_time=start_time)

    async def stop_processing_metrics(self, *, end_time: Optional[float] = None):
        """Stop processing metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_processing_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Start LLM usage metrics collection.

        Args:
            tokens: Token usage information for the LLM.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_llm_usage_metrics(tokens)
            if frame:
                await self.push_frame(frame)

    async def start_tts_usage_metrics(self, text: str):
        """Start TTS usage metrics collection.

        Args:
            text: The text being processed by TTS.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_tts_usage_metrics(text)
            if frame:
                await self.push_frame(frame)

    async def stop_all_metrics(self):
        """Stop all active metrics collection."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    def create_task(self, coroutine: Coroutine, name: Optional[str] = None) -> asyncio.Task:
        """Create a new task managed by this processor.

        Args:
            coroutine: The coroutine to run in the task.
            name: Optional name for the task.

        Returns:
            The created asyncio task.
        """
        if name:
            name = f"{self}::{name}"
        else:
            name = f"{self}::{coroutine.cr_code.co_name}"
        return self.task_manager.create_task(coroutine, name)

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = 1.0):
        """Cancel a task managed by this processor.

        A default timeout if 1 second is used in order to avoid potential
        freezes caused by certain libraries that swallow
        `asyncio.CancelledError`.

        Args:
            task: The task to cancel.
            timeout: Optional timeout for task cancellation.
        """
        await self.task_manager.cancel_task(task, timeout)

    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Wait for a task to complete.

        .. deprecated:: 0.0.81
            This function is deprecated, use `await task` or
            `await asyncio.wait_for(task, timeout)` instead.

        Args:
            task: The task to wait for.
            timeout: Optional timeout for waiting.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`FrameProcessor.wait_for_task()` is deprecated. "
                "Use `await task` or `await asyncio.wait_for(task, timeout)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if timeout:
            await asyncio.wait_for(task, timeout)
        else:
            await task

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._clock = setup.clock
        self._task_manager = setup.task_manager
        self._observer = setup.observer

        # Cache the observer's push-type subscription to avoid 3 attribute
        # lookups + 1 method call per push_frame. Updated once here at setup;
        # dynamic observer changes would require re-calling this.
        obs = self._observer
        if obs is not None and obs.has_push_frame_observers:
            any_sub = obs._any_push_subscribed
            self._push_subscribed = _PUSH_ALL if any_sub is None else any_sub
        # else: _push_subscribed stays None → ultra-fast path always taken

        # Update fast-path flag (handles the rare case where setup() is called
        # after __start(), e.g. dynamic pipeline reconfiguration).
        self._push_fast_ready = (
            self.__started and not self._has_push_frame_handlers and self._push_subscribed is None
        )

        # Create processing tasks.
        self.__create_input_task()

        if self._metrics is not None:
            await self._metrics.setup(self._task_manager)

    async def cleanup(self):
        """Clean up processor resources."""
        await super().cleanup()
        await self.__cancel_input_task()
        await self.__cancel_process_task()
        if self._metrics is not None:
            await self._metrics.cleanup()

    def link(self, processor: "FrameProcessor"):
        """Link this processor to the next processor in the pipeline.

        Args:
            processor: The processor to link to.
        """
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_clock(self) -> BaseClock:
        """Get the clock used by this processor.

        Returns:
            The clock instance.

        Raises:
            Exception: If the clock is not initialized.
        """
        if not self._clock:
            raise Exception(f"{self} Clock is still not initialized.")
        return self._clock

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this processor.

        Returns:
            The asyncio event loop.
        """
        return self.task_manager.get_event_loop()

    def _queue_frame_sync(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> bool:
        """Synchronously enqueue a frame without coroutine allocation overhead.

        This is the hot-path entry point called by push_frame's ultra-fast
        path. It saves ~190 ns per hop vs ``await queue_frame()`` by
        eliminating two coroutine allocations (queue_frame + put).

        Returns:
            True  — frame was handled; caller needs no further action.
            False — rare edge case (direct mode, interruption bypass); caller
                    must fall back to ``await queue_frame()``.
        """
        # _fast_enqueue_ready is True when not cancelling, not waiting for
        # an interruption, and not in direct mode — the common case.
        # Collapses 3 LOAD_ATTRs + 3 comparisons (~64 ns) into 1 flag check
        # (~13 ns), saving ~51 ns per hop in the data-frame hot path.
        if not self._fast_enqueue_ready:
            if self._cancelling:
                return True  # Handled (dropped)

            # InterruptionFrame during wait_for_interruption must bypass queues
            # and be processed immediately via async __process_frame.
            if self._wait_for_interruption and (
                frame.type_id == FrameType.CTRL_INTERRUPT
                or frame.type_id == FrameType.CTRL_START_INTERRUPT
            ):
                return False  # Fall back to async queue_frame

            if self._enable_direct_mode:
                return False  # Fall back to async queue_frame

        # Common path: enqueue synchronously.
        self.__input_queue.enqueue_sync((frame, direction, None))
        return True

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: Optional[FrameCallback] = None,
    ):
        """Queue a frame for processing.

        Args:
            frame: The frame to queue.
            direction: The direction of frame flow.
            callback: Optional callback to call after processing.
        """
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        # If we are waiting for an interruption, bypass all queued system frames
        # and process the frame right away. This is because a previous system
        # frame might be waiting for the interruption frame blocking the input
        # task, so this InterruptionFrame would never be dequeued and we'd
        # deadlock.
        if self._wait_for_interruption and (
            frame.type_id == FrameType.CTRL_INTERRUPT
            or frame.type_id == FrameType.CTRL_START_INTERRUPT
        ):
            await self.__process_frame(frame, direction, callback)
            return

        if self._enable_direct_mode:
            # Inline __process_frame for direct-mode processors (Pipeline,
            # PipelineSource, PipelineSink) to eliminate one coroutine
            # allocation per hop through the pipeline chain (~90 ns saved).
            try:
                if self._has_process_frame_handlers:
                    if self._has_handlers("on_before_process_frame"):
                        await self._call_event_handler("on_before_process_frame", frame)
                    await self.process_frame(frame, direction)
                    if callback:
                        await callback(self, frame, direction)
                    if self._has_handlers("on_after_process_frame"):
                        await self._call_event_handler("on_after_process_frame", frame)
                else:
                    await self.process_frame(frame, direction)
                    if callback:
                        await callback(self, frame, direction)
            except Exception as e:
                await self.push_error(error_msg=f"Error processing frame: {e}", exception=e)
        elif self.__input_queue._drop_policy == DropPolicy.BLOCK:
            # BLOCK policy needs back-pressure: fall through to the async put()
            # so the caller can be suspended until space is available.
            await self.__input_queue.put((frame, direction, callback))
        else:
            # Hot path: synchronous enqueue, no coroutine overhead (~95ns saved).
            self.__input_queue.enqueue_sync((frame, direction, callback))

    async def pause_processing_frames(self):
        """Pause processing of queued frames."""
        logger.trace(f"{self}: pausing frame processing")
        self.__should_block_frames = True
        if self.__process_event:
            self.__process_event.clear()

    async def pause_processing_system_frames(self):
        """Pause processing of queued system frames."""
        logger.trace(f"{self}: pausing system frame processing")
        self.__should_block_system_frames = True
        if self.__input_event:
            self.__input_event.clear()

    async def resume_processing_frames(self):
        """Resume processing of queued frames."""
        logger.trace(f"{self}: resuming frame processing")
        if self.__process_event:
            self.__process_event.set()

    async def resume_processing_system_frames(self):
        """Resume processing of queued system frames."""
        logger.trace(f"{self}: resuming system frame processing")
        if self.__input_event:
            self.__input_event.set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        if self._observer and self._observer.has_process_frame_observers:
            timestamp = self._clock.get_time() if self._clock else 0
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=timestamp,
            )
            await self._observer.on_process_frame(data)

        # Type-id integer dispatch: ~4-14ns vs ~80ns isinstance chain for data frames.
        # StartInterruptionFrame (deprecated) has CTRL_START_INTERRUPT, not CTRL_INTERRUPT,
        # so both are checked to preserve backwards compatibility.
        _tid = frame.type_id
        # Fast exit for non-CONTROL-category frames (AUDIO, TEXT, IMAGE, LLM, …).
        # FrameCategory.CONTROL == 0x08; the high byte of type_id encodes the category.
        # _tid >> 8 always yields a small int (category ≤ 0x16 < 256), so CPython uses
        # its small-int cache and the comparison is a pointer check — ~10 ns total.
        # Saves 8 FrameType attribute lookups (~50 ns each) per non-control frame.
        if _tid >> 8 != 0x08:
            return
        if _tid == FrameType.CTRL_START:
            await self.__start(frame)
        elif _tid == FrameType.CTRL_INTERRUPT or _tid == FrameType.CTRL_START_INTERRUPT:
            await self._start_interruption()
            await self.stop_all_metrics()
        elif _tid == FrameType.CTRL_CANCEL:
            await self.__cancel(frame)
        elif _tid == FrameType.CTRL_PAUSE or _tid == FrameType.CTRL_PAUSE_URGENT:
            await self.__pause(frame)
        elif _tid == FrameType.CTRL_RESUME or _tid == FrameType.CTRL_RESUME_URGENT:
            await self.__resume(frame)

    async def push_error(
        self,
        error_msg: str,
        exception: Optional[Exception] = None,
        fatal: bool = False,
    ):
        """Creates and pushes an ErrorFrame upstream.

        Creates and pushes an ErrorFrame upstream to notify other processors in the
        pipeline about an error condition. The error frame will include context about
        which processor generated the error.

        Args:
            error_msg: Descriptive message explaining the error condition.
            exception: Optional exception object that caused the error, if available.
                This provides additional context for debugging and error handling.
            fatal: Whether this error should be considered fatal to the pipeline.
                Fatal errors typically cause the entire pipeline to stop processing.
                Defaults to False for non-fatal errors.

        Example::

            ```python
            # Non-fatal error
            await self.push_error("Failed to process audio chunk, skipping")

            # Fatal error with exception context
            try:
                result = some_critical_operation()
            except Exception as e:
                await self.push_error("Critical operation failed", exception=e, fatal=True)
            ```
        """
        error_frame = ErrorFrame(error=error_msg, fatal=fatal, exception=exception, processor=self)
        await self.push_error_frame(error=error_frame)

    async def push_error_frame(self, error: ErrorFrame):
        """Push an error frame upstream.

        Args:
            error: The error frame to push.
        """
        if not error.processor:
            error.processor = self
        await self._call_event_handler("on_error", error)

        if error.exception:
            tb = traceback.extract_tb(error.exception.__traceback__)
            last = tb[-1]
            error_message = (
                f"{error.processor} exception ({last.filename}:{last.lineno}): {error.error}"
            )
        else:
            error_message = f"{error.processor} error: {error.error}"

        logger.error(error_message)
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        # _push_fast_ready is True when started=True, no push handlers, and no
        # observer subscription — the common case. Collapses 3 LOAD_ATTRs
        # (~78 ns) into a single flag check (~26 ns), saving ~52 ns per call.
        if not self._push_fast_ready:
            # Slow/error/observer path: check each condition explicitly.
            if not self.__started:
                logger.error(
                    "Processor {} trying to push {} but StartFrame not received yet",
                    self._name,
                    type(frame).__name__,
                )
                return

            if self._has_push_frame_handlers:
                # Slow path: event handlers registered — full handler scaffolding
                if self._has_handlers("on_before_push_frame"):
                    await self._call_event_handler("on_before_push_frame", frame)

                await self.__internal_push_frame(frame, direction)

                if self._has_handlers("on_after_push_frame"):
                    await self._call_event_handler("on_after_push_frame", frame)
                return

            elif (
                # Use the pre-cached _push_subscribed (set in setup()) to avoid 3
                # attribute lookups + 1 Python method call per push_frame.
                # _push_subscribed is None when no observer subscribes to push events;
                # _PUSH_ALL when any observer wants all frame types; frozenset otherwise.
                (ps := self._push_subscribed) is not None
                and (ps is _PUSH_ALL or frame.type_id in ps)
            ):
                # Observer path: needs timestamp + FramePushed creation
                await self.__internal_push_frame(frame, direction)
                return
            # else: fall through to ultra-fast path (started, no handlers, no observer)

        # Ultra-fast path: no handlers, no observer — inline the routing
        # and use the synchronous enqueue fast path to avoid ~190 ns of
        # coroutine allocation overhead (queue_frame + put) per hop.
        # Falls back to await queue_frame() only for rare edge cases
        # (direct mode, interruption bypass).
        try:
            if direction == FrameDirection.DOWNSTREAM:
                if self._next:
                    if _TRACE_FRAME_PUSHES:
                        logger.trace(
                            "Pushing {} from {} to {}", frame, self, self._next
                        )
                    if not self._next._queue_frame_sync(frame, direction):
                        await self._next.queue_frame(frame, direction)
            elif self._prev:
                if _TRACE_FRAME_PUSHES:
                    logger.trace(
                        "Pushing {} upstream from {} to {}", frame, self, self._prev
                    )
                if not self._prev._queue_frame_sync(frame, direction):
                    await self._prev.queue_frame(frame, direction)
        except Exception as e:
            await self.push_error(error_msg=f"Uncaught exception: {e}", exception=e)

    async def push_interruption_task_frame_and_wait(self, *, timeout: float = 5.0):
        """Push an interruption task frame upstream and wait for the interruption.

        This function sends an `InterruptionTaskFrame` upstream to the
        pipeline task. The task creates a corresponding `InterruptionFrame`
        and sends it downstream through the pipeline. An `asyncio.Event` is
        attached to both frames so the caller can wait until the interruption
        has fully traversed the pipeline. The event is set when the
        `InterruptionFrame` reaches the pipeline sink. If the frame does
        not complete within the given timeout, a warning is logged and the
        event is forcibly set so the caller is unblocked.

        Args:
            timeout: Maximum seconds to wait for the interruption to complete.
        """
        self._wait_for_interruption = True
        self._fast_enqueue_ready = False  # disable fast path while waiting

        event = asyncio.Event()

        await self.push_frame(InterruptionTaskFrame(event=event), FrameDirection.UPSTREAM)

        # Wait for the `InterruptionFrame` to complete and log a warning if it
        # takes too long. If it does take too long make sure we unblock it,
        # otherwise we will hang here forever.
        while not event.is_set():
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self}: InterruptionFrame has not completed after"
                    f" {timeout}s. Make sure InterruptionFrame.complete()"
                    " is being called (e.g. if the frame is being blocked"
                    " or consumed before reaching the pipeline sink)."
                )
                event.set()

        self._wait_for_interruption = False
        # Restore fast-enqueue only if not cancelling and not in direct mode.
        self._fast_enqueue_ready = not self._enable_direct_mode and not self._cancelling

    async def broadcast_frame(self, frame_cls: Type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    async def broadcast_frame_instance(self, frame: Frame):
        """Broadcasts a frame instance upstream and downstream.

        This method creates two new frame instances shallow-copying all fields
        from the original frame except `id` and `name`, which get fresh values.

        Args:
            frame: The frame instance to broadcast.

        Note:
            Prefer using `broadcast_frame()` when possible, as it is more
            efficient. This method should only be used when you are not the
            creator of the frame and need to broadcast an existing instance.
        """
        frame_cls = type(frame)
        init_fields = {f.name: getattr(frame, f.name) for f in dataclasses.fields(frame) if f.init}
        extra_fields = {
            f.name: getattr(frame, f.name)
            for f in dataclasses.fields(frame)
            if not f.init and f.name not in ("id", "name")
        }

        downstream_frame = frame_cls(**init_fields)
        for k, v in extra_fields.items():
            setattr(downstream_frame, k, v)
        # Preserve metadata if it was set on the original frame (lazy property).
        if frame._metadata is not None:
            downstream_frame._metadata = frame._metadata

        upstream_frame = frame_cls(**init_fields)
        for k, v in extra_fields.items():
            setattr(upstream_frame, k, v)
        if frame._metadata is not None:
            upstream_frame._metadata = frame._metadata

        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    async def __start(self, frame: StartFrame):
        """Handle the start frame to initialize processor state.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self.__started = True
        # Activate push fast-path gate now that we are started.
        # setup() has already run so _push_subscribed and _has_push_frame_handlers
        # reflect their final values in the common case.
        self._push_fast_ready = (
            not self._has_push_frame_handlers and self._push_subscribed is None
        )
        self._allow_interruptions = frame.allow_interruptions
        self._enable_metrics = frame.enable_metrics
        self._enable_usage_metrics = frame.enable_usage_metrics
        self._interruption_strategies = frame.interruption_strategies
        self._report_only_initial_ttfb = frame.report_only_initial_ttfb

        # NOTE(aleix): Remove when OpenAILLMContext/LLMUserContextAggregator is removed.
        self._deprecated_openaillmcontext = "deprecated_openaillmcontext" in frame.metadata

        self.__create_process_task()

    async def __cancel(self, frame: CancelFrame):
        """Handle the cancel frame to stop processor operation.

        Args:
            frame: The cancel frame.
        """
        self._cancelling = True
        self._fast_enqueue_ready = False  # drop all subsequent frames
        await self.__cancel_process_task()

    async def __pause(self, frame: FrameProcessorPauseFrame | FrameProcessorPauseUrgentFrame):
        """Handle pause frame to pause processor operation.

        Args:
            frame: The pause frame.
        """
        if frame.processor.name == self.name:
            await self.pause_processing_frames()

    async def __resume(self, frame: FrameProcessorResumeFrame | FrameProcessorResumeUrgentFrame):
        """Handle resume frame to resume processor operation.

        Args:
            frame: The resume frame.
        """
        if frame.processor.name == self.name:
            await self.resume_processing_frames()

    #
    # Handle interruptions
    #

    async def _start_interruption(self):
        """Start handling an interruption by cancelling current tasks."""
        try:
            if self._wait_for_interruption:
                # If we get here we know the process task was just waiting for
                # an interruption (push_interruption_task_frame_and_wait()), so
                # we can't cancel the task because it might still need to do
                # more things (e.g. pushing a frame after the
                # interruption). Instead we just drain the queue because this is
                # an interruption.
                self.__reset_process_task()
            elif isinstance(self.__process_current_frame, UninterruptibleFrame):
                # We don't want to cancel UninterruptibleFrame, so we simply
                # cleanup the queue.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task.
                await self.__cancel_process_task()
                self.__create_process_task()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception handling _start_interruption: {e}",
                exception=e,
            )

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        try:
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(
                    "Pushing {} from {} to {}", frame, self, self._next
                )

                if (
                    self._observer
                    and self._observer.has_push_frame_observers
                    and self._observer.is_push_interested(frame.type_id)
                ):
                    timestamp = self._clock.get_time() if self._clock else 0
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await self._observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(
                    "Pushing {} upstream from {} to {}", frame, self, self._prev
                )
                if (
                    self._observer
                    and self._observer.has_push_frame_observers
                    and self._observer.is_push_interested(frame.type_id)
                ):
                    timestamp = self._clock.get_time() if self._clock else 0
                    data = FramePushed(
                        source=self,
                        destination=self._prev,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await self._observer.on_push_frame(data)
                await self._prev.queue_frame(frame, direction)
        except Exception as e:
            await self.push_error(error_msg=f"Uncaught exception: {e}", exception=e)

    def _check_started(self, frame: Frame):
        """Check if the processor has been started.

        Args:
            frame: The frame being processed.

        Returns:
            True if the processor has been started.
        """
        if not self.__started:
            logger.error(f"{self} Trying to process {frame} but StartFrame not received yet")
        return self.__started

    def __create_input_task(self):
        """Create the frame input processing task."""
        if self._enable_direct_mode:
            return

        if not self.__input_frame_task:
            self.__input_event = asyncio.Event()
            self.__input_frame_task = self.create_task(self.__input_frame_task_handler())

    async def __cancel_input_task(self):
        """Cancel the frame input processing task."""
        if self.__input_frame_task:
            # Apply a timeout as a safeguard: if a library swallows asyncio.CancelledError,
            # the task would otherwise never be cancelled. With a timeout, we can detect this
            # situation and surface it in the logs instead of hanging indefinitely.
            await self.cancel_task(self.__input_frame_task, INPUT_TASK_CANCEL_TIMEOUT_SECS)
            self.__input_frame_task = None

    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())

    def __reset_process_task(self):
        """Reset non-system frame processing task."""
        if self._enable_direct_mode:
            return

        self.__should_block_frames = False
        self.__process_event = asyncio.Event()
        self.__reset_process_queue()

    def __reset_process_queue(self):
        """Reset non-system frame processing queue."""
        # Collect UninterruptibleFrame items, discard the rest.
        uninterruptible = []
        while not self.__process_queue.empty():
            item = self.__process_queue.get_nowait()
            if isinstance(item[0], UninterruptibleFrame):
                uninterruptible.append(item)

        # Put back UninterruptibleFrame frames into our process queue.
        for item in uninterruptible:
            self.__process_queue.put_nowait(item)

    async def __cancel_process_task(self):
        """Cancel the non-system frame processing task."""
        if self.__process_frame_task:
            await self.cancel_task(self.__process_frame_task, PROCESS_TASK_CANCEL_TIMEOUT_SECS)
            self.__process_frame_task = None

    async def __process_frame(
        self, frame: Frame, direction: FrameDirection, callback: Optional[FrameCallback]
    ):
        try:
            if self._has_process_frame_handlers:
                # Slow path: event handlers registered
                if self._has_handlers("on_before_process_frame"):
                    await self._call_event_handler("on_before_process_frame", frame)

                await self.process_frame(frame, direction)
                if callback:
                    await callback(self, frame, direction)

                if self._has_handlers("on_after_process_frame"):
                    await self._call_event_handler("on_after_process_frame", frame)
            else:
                # Fast path: no handlers — skip two 83ns dict lookups per call
                await self.process_frame(frame, direction)
                if callback:
                    await callback(self, frame, direction)
        except Exception as e:
            await self.push_error(error_msg=f"Error processing frame: {e}", exception=e)

    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queued for another
        task to execute.

        Batch drain: after the initial await get(), drains all immediately
        available DataFrame items via get_nowait() in the same loop iteration,
        avoiding one asyncio task wakeup per frame. Falls back to await get()
        once the queue is empty, preserving full interrupt semantics.
        System frames break the batch and are handled immediately (they may
        yield inside __process_frame).
        """
        while True:
            # Keep the queue item as a tuple; unpack only `frame` for the
            # system-frame check, deferring direction/callback unpacking to
            # the rare system-frame and error paths. This lets the inner data-
            # frame batch drain pass the original item tuple directly to the
            # process queue, eliminating one 3-tuple allocation per frame
            # (~50 ns saved per hop; ~1.3 ms per 27k-call benchmark run).
            item = await self.__input_queue.get()
            frame = item[0]

            # Unblock system frame processing if paused (very rare path).
            while self.__should_block_system_frames and self.__input_event:
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if frame.is_system_frame:
                await self.__process_frame(frame, item[1], item[2])
                # SystemFrame processing may yield; start a fresh batch.
                continue

            pq = self.__process_queue
            if pq is None:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

            # Data frame batch drain.
            # Pass the original item tuple directly to pq to avoid repacking
            # (frame, direction, callback) into a new tuple on every iteration.
            # Saves ~50 ns per hop vs pq.put_nowait((frame, direction, callback)).
            # Cache queue references as locals so the inner loop uses LOAD_FAST
            # (~5 ns) instead of repeated LOAD_ATTR on name-mangled attributes
            # (~25 ns each).
            iq = self.__input_queue
            while True:
                pq.put_nowait(item)  # pass original tuple — no repacking
                # task_done() is a no-op on BoundedFrameQueue / ProcessQueue
                # (no join() caller exists); skip the ~120 ns call overhead.
                try:
                    item = iq.get_nowait()
                    frame = item[0]  # only unpack frame for system-frame check
                except asyncio.QueueEmpty:
                    break
                if frame.is_system_frame:
                    # System frame arrived mid-batch; handle immediately.
                    while self.__should_block_system_frames and self.__input_event:
                        logger.trace(f"{self}: system frame processing paused")
                        await self.__input_event.wait()
                        self.__input_event.clear()
                        self.__should_block_system_frames = False
                        logger.trace(f"{self}: system frame processing resumed")
                    await self.__process_frame(frame, item[1], item[2])
                    break  # End data batch; outer loop starts fresh.

    async def __process_frame_task_handler(self):
        """Handle non-system frames from the process queue.

        Drains all immediately available frames per wakeup (batch drain) to
        amortize asyncio task scheduling overhead: one task wakeup per burst
        instead of one wakeup per frame. Falls back to await get() when the
        queue is empty, preserving full interrupt/pause/cancellation semantics.

        The __process_frame() body is inlined here for DataFrames to eliminate
        one coroutine allocation per frame per processor (~90 ns saved).
        SystemFrames still go through __process_frame() via __input_frame_task_handler.
        """
        while True:
            self.__process_current_frame = None

            (frame, direction, callback) = await self.__process_queue.get()

            while True:
                self.__process_current_frame = frame

                if self.__should_block_frames and self.__process_event:
                    logger.trace(f"{self}: frame processing paused")
                    await self.__process_event.wait()
                    self.__process_event.clear()
                    self.__should_block_frames = False
                    logger.trace(f"{self}: frame processing resumed")

                # Inline __process_frame for DataFrames: saves one coroutine
                # allocation (~90 ns) per frame per processor in the hot path.
                # The slow path (event handlers) is preserved verbatim.
                try:
                    if self._has_process_frame_handlers:
                        if self._has_handlers("on_before_process_frame"):
                            await self._call_event_handler("on_before_process_frame", frame)
                        await self.process_frame(frame, direction)
                        if callback:
                            await callback(self, frame, direction)
                        if self._has_handlers("on_after_process_frame"):
                            await self._call_event_handler("on_after_process_frame", frame)
                    else:
                        await self.process_frame(frame, direction)
                        if callback:
                            await callback(self, frame, direction)
                except Exception as e:
                    await self.push_error(error_msg=f"Error processing frame: {e}", exception=e)

                # task_done() is a no-op on ProcessQueue (no join() caller);
                # skip the ~140 ns overhead in the per-frame hot path.

                # Batch drain: grab the next frame synchronously if one is
                # already queued, avoiding a task scheduling round-trip.
                self.__process_current_frame = None
                try:
                    (frame, direction, callback) = self.__process_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
