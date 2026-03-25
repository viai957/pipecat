#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline implementation for connecting and managing frame processors.

This module provides the main Pipeline class that connects frame processors
in sequence and manages frame flow between them, along with helper classes
for pipeline source and sink operations.

When the ``pipecat-ai-native`` package is installed, the Pipeline transparently
activates the Rust pipeline engine for improved performance. Set
``PIPECAT_NATIVE=0`` to disable.
"""

from typing import Callable, Coroutine, List, Optional

from loguru import logger

from pipecat._native_status import is_native_engine_enabled
from pipecat.frames.frame_types import FrameCategory
from pipecat.frames.frames import Frame
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup


class PipelineSource(FrameProcessor):
    """Source processor that forwards frames to an upstream handler.

    This processor acts as the entry point for a pipeline, forwarding
    downstream frames to the next processor and upstream frames to a
    provided upstream handler function.
    """

    def __init__(self, upstream_push_frame: Callable[[Frame, FrameDirection], Coroutine], **kwargs):
        """Initialize the pipeline source.

        Args:
            upstream_push_frame: Coroutine function to handle upstream frames.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(enable_direct_mode=True, **kwargs)
        self._upstream_push_frame = upstream_push_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and route them based on direction.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        # Skip super() for data frames with no observer: base class has nothing
        # to do, saving ~90 ns of coroutine creation per hop. Control frames
        # (category 0x08) and observer-notified frames always call super().
        if (self._observer and self._observer.has_process_frame_observers) or (
            frame.type_id >> 8 == FrameCategory.CONTROL
        ):
            await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._upstream_push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                # Inline push_frame ultra-fast path: saves ~90 ns coroutine
                # creation per hop. Falls back to push_frame if push handlers
                # or observer subscription are active.
                if (
                    not self._has_push_frame_handlers
                    and not (self._observer and self._observer.has_push_frame_observers)
                    and self._next
                ):
                    if not self._next._queue_frame_sync(frame, direction):
                        await self._next.queue_frame(frame, direction)
                else:
                    await self.push_frame(frame, direction)


class PipelineSink(FrameProcessor):
    """Sink processor that forwards frames to a downstream handler.

    This processor acts as the exit point for a pipeline, forwarding
    upstream frames to the previous processor and downstream frames to a
    provided downstream handler function.
    """

    def __init__(
        self, downstream_push_frame: Callable[[Frame, FrameDirection], Coroutine], **kwargs
    ):
        """Initialize the pipeline sink.

        Args:
            downstream_push_frame: Coroutine function to handle downstream frames.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(enable_direct_mode=True, **kwargs)
        self._downstream_push_frame = downstream_push_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and route them based on direction.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        # Skip super() for data frames with no observer: base class has nothing
        # to do, saving ~90 ns of coroutine creation per hop. Control frames
        # (category 0x08) and observer-notified frames always call super().
        if (self._observer and self._observer.has_process_frame_observers) or (
            frame.type_id >> 8 == FrameCategory.CONTROL
        ):
            await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self.push_frame(frame, direction)
            case FrameDirection.DOWNSTREAM:
                await self._downstream_push_frame(frame, direction)


class Pipeline(BasePipeline):
    """Main pipeline implementation that connects frame processors in sequence.

    Creates a linear chain of frame processors with automatic source and sink
    processors for external frame handling. Manages processor lifecycle and
    provides metrics collection from contained processors.
    """

    def __init__(
        self,
        processors: List[FrameProcessor],
        *,
        source: Optional[FrameProcessor] = None,
        sink: Optional[FrameProcessor] = None,
    ):
        """Initialize the pipeline with a list of processors.

        Args:
            processors: List of frame processors to connect in sequence.
            source: An optional pipeline source processor.
            sink: An optional pipeline sink processor.
        """
        super().__init__(enable_direct_mode=True)

        # Add a source and a sink queue so we can forward frames upstream and
        # downstream outside of the pipeline.
        self._source = source or PipelineSource(self.push_frame, name=f"{self}::Source")
        self._sink = sink or PipelineSink(self.push_frame, name=f"{self}::Sink")
        self._processors: List[FrameProcessor] = [self._source] + processors + [self._sink]

        self._link_processors()

        # Precompute the first non-direct-mode processor in the downstream
        # chain. Used by the fast enqueue path in process_frame to bypass
        # the PipelineSource/nested-Pipeline routing coroutine chain for data
        # frames when no observer is attached.
        # Also records whether any transparent processor was skipped (needed
        # to gate push_frame observer safety in setup()).
        self._fast_path_skips_transparent: bool = False
        self._fast_queue_target: Optional[FrameProcessor] = self._compute_fast_queue_target()

        # Precomputed gate for the pipeline-level fast-enqueue path:
        # True when _fast_queue_target exists AND no on_process_frame observer is
        # active. When True, _queue_frame_sync() and _try_fast_enqueue() bypass
        # this pipeline's own task chain (saving 2 asyncio event wakeups per
        # batch, ~6 µs), enqueuing directly to the inner fast_queue_target.
        # Refined in setup() once the observer is known; initially assumes none.
        self._pipeline_fast_enqueue_ready: bool = bool(self._fast_queue_target)

        # Transparent native engine activation (disable via PIPECAT_NATIVE=0)
        self._native_pipeline = None
        if is_native_engine_enabled():
            try:
                from pipecat.engine.rust_engine import NativePipeline

                self._native_pipeline = NativePipeline(processors)
            except Exception:
                logger.warning("NativePipeline init failed, falling back to pure Python")

    #
    # Frame processor
    #

    @property
    def processors(self):
        """Return the list of sub-processors contained within this processor.

        Only compound processors (e.g. pipelines and parallel pipelines) have
        sub-processors. Non-compound processors will return an empty list.

        Returns:
            The list of sub-processors if this is a compound processor.
        """
        return self._processors

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
        return [self._source]

    def processors_with_metrics(self):
        """Return processors that can generate metrics.

        Recursively collects all processors that support metrics generation,
        including those from nested pipelines.

        Returns:
            List of frame processors that can generate metrics.
        """
        services = []
        for p in self.processors:
            if p.can_generate_metrics():
                services.append(p)
            services.extend(p.processors_with_metrics())
        return services

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the pipeline and all contained processors.

        Args:
            setup: Configuration for frame processor setup.
        """
        await super().setup(setup)
        # Refine _pipeline_fast_enqueue_ready now that self._observer is known.
        # The bypass skips Pipeline.process_frame — disable if on_process_frame
        # observers are active.
        # has_push_frame_observers is normally NOT checked: that flag concerns
        # frames EXITING via PipelineSink → Pipeline.push_frame (outgoing path).
        # EXCEPTION: when _fast_path_skips_transparent is True (e.g. the task
        # pipeline bypasses RTVIProcessor), the transparent processor's push_frame
        # calls are also skipped. Push_frame observers (e.g. IdleFrameObserver)
        # that depend on seeing those calls must disable the bypass.
        self._pipeline_fast_enqueue_ready = bool(self._fast_queue_target) and not (
            self._observer
            and (
                self._observer.has_process_frame_observers
                or (self._fast_path_skips_transparent and self._observer.has_push_frame_observers)
            )
        )
        await self._setup_processors(setup)

    async def cleanup(self):
        """Clean up the pipeline and all contained processors."""
        await super().cleanup()
        await self._cleanup_processors()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames by routing them through the pipeline.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        # Compute once; used in both the super() guard and the routing branch.
        is_control = frame.type_id >> 8 == FrameCategory.CONTROL

        # Skip super() for data frames with no observer: base class has nothing
        # to do, saving ~90 ns of coroutine creation per hop. Control frames
        # (category 0x08) and observer-notified frames always call super().
        # When _pipeline_fast_enqueue_ready is True, we know neither observer
        # flag is set, so only the control-frame check is needed.
        if is_control or (
            not self._pipeline_fast_enqueue_ready
            and self._observer
            and self._observer.has_process_frame_observers
        ):
            await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            # Fast path: for data frames with no observer, skip the entire
            # PipelineSource → nested-Pipeline → ... routing chain and enqueue
            # directly to the first non-direct-mode processor. Saves 6+
            # coroutine allocations (~1 μs) per frame for typical pipelines.
            if self._pipeline_fast_enqueue_ready and not is_control:
                if not self._fast_queue_target._queue_frame_sync(frame, direction):
                    await self._fast_queue_target.queue_frame(frame, direction)
            else:
                await self._source.queue_frame(frame, FrameDirection.DOWNSTREAM)
        elif direction == FrameDirection.UPSTREAM:
            await self._sink.queue_frame(frame, FrameDirection.UPSTREAM)

    async def _setup_processors(self, setup: FrameProcessorSetup):
        """Set up all processors in the pipeline."""
        for p in self._processors:
            await p.setup(setup)

    async def _cleanup_processors(self):
        """Clean up all processors in the pipeline."""
        for p in self._processors:
            await p.cleanup()

    def _link_processors(self):
        """Link all processors in sequence and set their parent."""
        prev = self._processors[0]
        for curr in self._processors[1:]:
            prev.link(curr)
            prev = curr

    def _queue_frame_sync(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> bool:
        """Bypass this pipeline's task chain for downstream data frames.

        When a processor upstream of this pipeline calls ``push_frame`` and
        hits the ultra-fast path, it invokes ``_queue_frame_sync`` on its
        ``_next`` (which may be a Pipeline).  The base-class implementation
        always returns False for direct-mode processors, forcing the caller to
        fall back to ``await queue_frame()``, which adds two asyncio task
        wakeups (input task + process task) just to route the frame through
        ``Pipeline.process_frame`` to ``_fast_queue_target``.

        This override short-circuits that: for downstream data frames when no
        observer requires per-frame events, it enqueues directly to the inner
        ``_fast_queue_target``, saving ~2 asyncio event wakeups per batch
        (~6 µs for a typical 200-frame batch).

        Returns True if the frame was synchronously forwarded, False when the
        caller must fall back to ``await queue_frame()``.
        """
        if (
            direction == FrameDirection.DOWNSTREAM
            and self._pipeline_fast_enqueue_ready
            and frame.type_id >> 8 != FrameCategory.CONTROL
        ):
            return self._fast_queue_target._queue_frame_sync(frame, direction)
        return False

    def _try_fast_enqueue(self, frame: Frame) -> bool:
        """Synchronously enqueue a data frame to the fast-path target.

        Bypasses the ``queue_frame`` → ``process_frame`` coroutine chain for
        data frames when no observer requires per-frame notifications.  Returns
        True when the frame was enqueued synchronously, False when the caller
        must fall back to ``await queue_frame()`` (control frame, observer
        active, or the target's sync queue is blocked).
        """
        if self._pipeline_fast_enqueue_ready and frame.type_id >> 8 != FrameCategory.CONTROL:
            return self._fast_queue_target._queue_frame_sync(frame, FrameDirection.DOWNSTREAM)
        return False

    def _compute_fast_queue_target(self) -> Optional[FrameProcessor]:
        """Find the first non-direct-mode processor for the fast enqueue path.

        Traverses the processor chain starting after the source, recursing into
        nested Pipelines and transparent processors, to find the first processor
        that owns a task queue.  Returns None if no safe target exists.

        Safety rules:
        - Non-Pipeline direct-mode processors (e.g. FunctionFilter) MUST run
          their process_frame logic and cannot be skipped → return None.
        - Transparent processors (e.g. RTVIProcessor, which only calls
          push_frame for data frames) are safe to bypass → traverse past them
          to find the real fast target.
        """
        current = self._source._next
        while current is not None and current is not self._sink:
            if isinstance(current, Pipeline):
                target = current._compute_fast_queue_target()
                if target is not None:
                    # Propagate the transparent-bypass flag from the nested pipeline.
                    self._fast_path_skips_transparent = (
                        self._fast_path_skips_transparent or current._fast_path_skips_transparent
                    )
                    return target
            elif current._transparent_for_data_frames:
                # Transparent processor: data frames pass through unchanged.
                # Safe to bypass for the fast-enqueue path; continue searching.
                # Record that we're skipping a transparent processor's push_frame
                # calls — setup() uses this to disable the fast path when a
                # push_frame observer (e.g. IdleFrameObserver) is active.
                self._fast_path_skips_transparent = True
            elif not current._enable_direct_mode:
                return current
            else:
                # Non-Pipeline, non-transparent direct-mode processor.
                # Its process_frame must run — no safe fast target for this chain.
                return None
            current = current._next
        return None
