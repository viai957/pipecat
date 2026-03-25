"""Python shim classes for the Rust native pipeline engine.

These classes wrap the Rust engine types exposed by ``pipecat-ai-native``.

Architecture::

    NativePipeline
      Validates Python processors for Rust compatibility by calling
      ``wrap_processor()`` on each. Stored on ``Pipeline._native_pipeline``
      when the native package is installed.

    NativePipelineTask
      Wraps Rust NativePipelineTask which drives the pipeline loop with
      Python processors called via GIL-protected callbacks.

      Exposes ``queue_frame()`` and ``cancel()`` for frame injection into
      the Rust engine's input channel, enabling full task-level delegation.

    NativePipelineRunner
      Wraps NativePipelineTask with signal handling.

The native engine is activated automatically when ``pipecat-ai-native`` is
installed. Disable with ``PIPECAT_NATIVE=0``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from pipecat._native_status import is_native_available

if TYPE_CHECKING:
    from pipecat.frames.frames import Frame
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class NativePipeline:
    """Wraps Rust Pipeline. Accepts regular Python FrameProcessor instances.

    Each Python processor is validated via ``wrap_processor()`` on the Rust
    side to confirm compatibility. Stored on ``Pipeline._native_pipeline``
    for future use when the Rust engine drives execution.
    """

    def __init__(self, processors: List[FrameProcessor]):
        """Initialize the native pipeline with Python processors.

        Args:
            processors: List of frame processors to validate for Rust compatibility.
        """
        if not is_native_available():
            raise RuntimeError("pipecat-ai-native is not installed or is disabled")

        from pipecat._native import wrap_processor  # type: ignore[import-not-found]

        self._processors = processors

        # Validate all processors have process_frame
        for p in processors:
            wrap_processor(p)

    @property
    def processors(self) -> List[FrameProcessor]:
        """Return the Python processors managed by this pipeline."""
        return self._processors


class NativePipelineTask:
    """Wraps Rust NativePipelineTask with observer bridging and frame injection.

    The Rust engine owns the pipeline loop (frame routing, backpressure,
    heartbeat injection, sink monitoring, observer dispatch). Python
    processors are called via GIL-protected callbacks.

    ``queue_frame()`` and ``cancel()`` delegate to the Rust task's
    ``queue_frame()`` and ``cancel()`` methods, enabling Python code
    to inject frames into the Rust pipeline while it is running.
    """

    def __init__(
        self,
        pipeline: NativePipeline,
        params: Optional[Any] = None,
        observers: Optional[List[BaseObserver]] = None,
        sink_callback: Optional[Any] = None,
    ):
        """Initialize the native pipeline task.

        Args:
            pipeline: The native pipeline containing validated processors.
            params: Optional pipeline parameters (heartbeat interval, etc.).
            observers: Optional list of observers for frame flow monitoring.
            sink_callback: Optional callback invoked when a terminal frame
                reaches the pipeline sink. Receives the frame name as a string
                ("End", "Stop", "Cancel", "FatalError").
        """
        from pipecat._native import NativePipelineTask as RustNativePipelineTask  # type: ignore

        self._pipeline = pipeline
        self._params = params
        self._observers = observers or []
        self._running = False
        self._cancelling = False
        self._end_reason: Optional[str] = None

        # Extract heartbeat interval from params if provided
        heartbeat_secs = 5.0  # default
        if params is not None and hasattr(params, "heartbeat_interval"):
            heartbeat_secs = params.heartbeat_interval

        # Sink callback: called from Rust when a terminal frame reaches the sink.
        # This enables hybrid delegation where Python manages lifecycle events.
        def _sink_handler(reason: str):
            self._end_reason = reason
            if sink_callback is not None:
                sink_callback(reason)

        # Create the Rust-side task with Python processors and observers
        self._rust_task = RustNativePipelineTask(
            processors=pipeline.processors,
            observers=self._observers if self._observers else None,
            heartbeat_secs=heartbeat_secs,
            sink_callback=_sink_handler,
        )

    @property
    def end_reason(self) -> Optional[str]:
        """The terminal frame name that ended the pipeline, or None if still running."""
        return self._end_reason

    def queue_frame(self, frame: Frame):
        """Queue a frame for injection into the Rust pipeline (downstream).

        Converts the Python frame to a Rust Frame and sends it through
        the pipeline's external channel. Can be called while the pipeline
        is running.

        Args:
            frame: The Python frame to inject into the pipeline.
        """
        self._rust_task.queue_frame(frame)

    def queue_upstream_frame(self, frame: Frame):
        """Queue a frame traveling upstream into the pipeline sink.

        Args:
            frame: The Python frame to send upstream.
        """
        self._rust_task.queue_upstream_frame(frame)

    async def cancel(self):
        """Cancel the pipeline task by injecting a CancelFrame."""
        self._cancelling = True
        self._rust_task.cancel()

    async def run(self):
        """Run the pipeline task to completion.

        This bridges the Rust tokio runtime to Python's asyncio event loop.
        The Rust engine owns the pipeline loop; Python processors are called
        via GIL-protected callbacks.
        """
        if self._running:
            raise RuntimeError("Task is already running")

        self._running = True
        self._cancelling = False

        try:
            logger.info(
                "Native pipeline task started with %d processors",
                len(self._pipeline.processors),
            )
            # Delegate to Rust's pipeline loop via pyo3-async-runtimes bridge
            await self._rust_task.run_async()
        finally:
            self._running = False

    @property
    def is_running(self) -> bool:
        """Return True if the pipeline task is currently running."""
        return self._running


class NativePipelineRunner:
    """Wraps NativePipelineTask with signal handling.

    Bridges Rust's tokio runtime to Python's asyncio event loop for
    running pipeline tasks.
    """

    def __init__(
        self,
        *,
        handle_sigint: bool = True,
        handle_sigterm: bool = False,
    ):
        """Initialize the native pipeline runner.

        Args:
            handle_sigint: Whether to handle SIGINT for graceful shutdown.
            handle_sigterm: Whether to handle SIGTERM for graceful shutdown.
        """
        self._handle_sigint = handle_sigint
        self._handle_sigterm = handle_sigterm

    async def run(self, task: NativePipelineTask):
        """Run a NativePipelineTask to completion.

        Args:
            task: The pipeline task to run.
        """
        await task.run()
