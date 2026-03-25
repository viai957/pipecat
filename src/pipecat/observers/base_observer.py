#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base observer classes for monitoring frame flow in the Pipecat pipeline.

This module provides the foundation for observing frame transfers between
processors without modifying the pipeline structure. Observers can be used
for logging, debugging, analytics, and monitoring pipeline behavior.
"""

from dataclasses import dataclass
from typing import Optional

from typing_extensions import TYPE_CHECKING

from pipecat.frames.frames import Frame
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class FrameProcessed:
    """Event data for frame processing in the pipeline.

    Represents an event where a frame is being processed by a processor. This
    data structure is typically used by observers to track the flow of frames
    through the pipeline for logging, debugging, or analytics purposes.

    Parameters:
        processor: The processor processing the frame.
        frame: The frame being processed.
        direction: The direction of the frame (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.

    """

    processor: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


@dataclass
class FramePushed:
    """Event data for frame transfers between processors in the pipeline.

    Represents an event where a frame is pushed from one processor to another
    within the pipeline. This data structure is typically used by observers
    to track the flow of frames through the pipeline for logging, debugging,
    or analytics purposes.

    Parameters:
        source: The processor sending the frame.
        destination: The processor receiving the frame.
        frame: The frame being transferred.
        direction: The direction of the transfer (e.g., downstream or upstream).
        timestamp: The time when the frame was pushed, based on the pipeline clock.
    """

    source: "FrameProcessor"
    destination: "FrameProcessor"
    frame: Frame
    direction: "FrameDirection"
    timestamp: int


class BaseObserver(BaseObject):
    """Base class for pipeline frame observers.

    Observers can view all frames that flow through the pipeline without
    needing to inject processors into the pipeline structure. This enables
    non-intrusive monitoring capabilities such as frame logging, debugging,
    performance analysis, and analytics collection.

    Class Attributes:
        has_process_frame_observers: Whether this observer (or any proxied
            observer) overrides on_process_frame. Used as a fast-path guard
            to skip FrameProcessed creation when no observer needs it.
        has_push_frame_observers: Whether this observer (or any proxied
            observer) overrides on_push_frame.
    """

    # Default True so unknown/custom observer subclasses always receive events.
    has_process_frame_observers: bool = True
    has_push_frame_observers: bool = True

    # Declare the frame types this observer actually handles in on_push_frame.
    # None means "subscribe to all frame types" (safe default for user observers).
    # A frozenset[int] restricts dispatching to only those type_id values, enabling
    # TaskObserver to skip FramePushed creation for unsubscribed frame types.
    push_frame_types: Optional[frozenset] = None

    def is_push_interested(self, type_id: int) -> bool:
        """Return True if this observer wants to receive a push event for type_id.

        The default implementation always returns True (conservative — safe for
        any observer that doesn't declare push_frame_types). TaskObserver
        overrides this to check against the union of all proxy subscriptions.

        Args:
            type_id: The integer value of the frame's FrameType enum.
        """
        return True

    async def on_process_frame(self, data: FrameProcessed):
        """Handle the event when a frame is being processed by a processor.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        being processed by a processor.

        Args:
            data: The event data containing details about the frame processing.
        """
        pass

    async def on_push_frame(self, data: FramePushed):
        """Handle the event when a frame is pushed from one processor to another.

        This method should be implemented by subclasses to define specific
        behavior (e.g., logging, monitoring, debugging) when a frame is
        transferred through the pipeline.

        Args:
            data: The event data containing details about the frame transfer.
        """
        pass
