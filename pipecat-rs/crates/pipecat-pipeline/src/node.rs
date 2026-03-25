//! Internal runtime wrapper that runs a single [`FrameProcessor`] as a tokio task.
//!
//! Each [`ProcessorNode`] owns a processor, an input queue, and the wiring to
//! its neighbours. The run loop pulls frames from the input queue (system
//! frames first), handles interruption draining, and dispatches to the
//! processor's `process_frame` method.

use tokio_util::sync::CancellationToken;

use pipecat_core::clock::PipelineClock;
use pipecat_core::{Frame, Result};

use crate::backpressure::{BoundedFrameQueue, FrameEnvelope, FrameQueueSender};
#[cfg(test)]
use crate::backpressure::QueueConfig;
use crate::processor::{FrameProcessor, ProcessorContext};

/// Internal node wrapping a single processor and its input queue.
///
/// Created by [`Pipeline`](crate::pipeline::Pipeline) during assembly.
/// Each node is spawned as a tokio task via [`run`](ProcessorNode::run).
pub(crate) struct ProcessorNode {
    processor: Box<dyn FrameProcessor>,
    input_queue: BoundedFrameQueue,
    ctx: ProcessorContext,
    cancellation: CancellationToken,
    /// Cached processor name to avoid per-frame allocation.
    proc_name: String,
    /// Cached from `processor.is_sync()` at construction time.
    is_sync: bool,
}

impl ProcessorNode {
    /// Create a new node.
    ///
    /// - `processor`: the user-provided processor implementation.
    /// - `config`: queue configuration for the input queue.
    /// - `downstream`: sender to the next node (or sink).
    /// - `upstream`: sender to the previous node (or source).
    /// - `clock`: shared pipeline clock.
    /// - `cancellation`: token to signal shutdown.
    ///
    /// Returns `(input_sender, node)`. The `input_sender` is used by the
    /// pipeline to wire the previous node's output to this node's input.
    #[cfg(test)]
    pub(crate) fn new(
        processor: Box<dyn FrameProcessor>,
        config: QueueConfig,
        downstream: FrameQueueSender,
        upstream: FrameQueueSender,
        clock: PipelineClock,
        cancellation: CancellationToken,
    ) -> (FrameQueueSender, Self) {
        let name = processor.name().to_string();
        let is_sync = processor.is_sync();
        let (input_sender, input_queue) = BoundedFrameQueue::new(config, &name);
        let node = Self {
            processor,
            input_queue,
            ctx: ProcessorContext::new(downstream, upstream, clock),
            cancellation,
            proc_name: name,
            is_sync,
        };
        (input_sender, node)
    }

    /// Create a node with an externally-provided input queue.
    ///
    /// Used by [`Pipeline::start()`] which pre-creates all input queues to
    /// allow direct cross-node wiring without forwarder tasks.
    pub(crate) fn new_with_queue(
        processor: Box<dyn FrameProcessor>,
        input_queue: BoundedFrameQueue,
        downstream: FrameQueueSender,
        upstream: FrameQueueSender,
        clock: PipelineClock,
        cancellation: CancellationToken,
    ) -> Self {
        let proc_name = processor.name().to_string();
        let is_sync = processor.is_sync();
        Self {
            processor,
            input_queue,
            ctx: ProcessorContext::new(downstream, upstream, clock),
            cancellation,
            proc_name,
            is_sync,
        }
    }

    /// Run the processor loop until cancellation or an `End`/`Cancel` frame.
    ///
    /// The outer loop uses `tokio::select!` to wait for either cancellation or
    /// a frame from the input queue. After processing the first frame, an inner
    /// batch loop drains up to [`BATCH_DRAIN_LIMIT`] additional frames via
    /// `get_nowait()` before yielding back to the scheduler. This amortises the
    /// per-iteration overhead of `tokio::select!` (cancellation future + async
    /// queue get) across multiple frames during bursts.
    pub(crate) async fn run(mut self) -> Result<()> {
        /// Maximum extra frames to drain in a tight loop after the initial
        /// async dequeue before re-entering `tokio::select!`.
        const BATCH_DRAIN_LIMIT: usize = 15;

        tracing::debug!(processor = %self.proc_name, "node run loop started");

        'outer: loop {
            // ── Async wait for the first frame (or cancellation) ────────
            tokio::select! {
                biased;

                _ = self.cancellation.cancelled() => {
                    tracing::debug!(processor = %self.proc_name, "node cancelled via token");
                    break;
                }

                maybe_env = self.input_queue.get() => {
                    let Some(envelope) = maybe_env else {
                        tracing::debug!(processor = %self.proc_name, "input queue closed");
                        break;
                    };

                    if self.dispatch_one(envelope).await? {
                        break;
                    }
                }
            }

            // ── Batch drain: process up to N more without re-polling ────
            for _ in 0..BATCH_DRAIN_LIMIT {
                if self.cancellation.is_cancelled() {
                    break 'outer;
                }
                match self.input_queue.get_nowait() {
                    Some(envelope) => {
                        if self.dispatch_one(envelope).await? {
                            break 'outer;
                        }
                    }
                    None => break,
                }
            }
        }

        self.processor.cleanup().await?;
        tracing::debug!(processor = %self.proc_name, "node cleanup complete");
        Ok(())
    }

    /// Dispatch a single frame envelope to the processor.
    ///
    /// Returns `Ok(true)` when the node should shut down (End/Cancel received),
    /// `Ok(false)` to continue processing.
    async fn dispatch_one(&mut self, envelope: FrameEnvelope) -> Result<bool> {
        let FrameEnvelope { frame, direction } = envelope;

        match &frame {
            Frame::Interruption(_) => {
                self.input_queue.drain_keeping_uninterruptible();
                tracing::debug!(processor = %self.proc_name, "interruption: drained data queue");
                self.processor.process_frame(frame, direction, &self.ctx).await?;
            }

            Frame::Start(_) => {
                self.processor.setup(&self.ctx).await?;
                self.processor.process_frame(frame, direction, &self.ctx).await?;
            }

            Frame::Cancel(_) => {
                self.processor.process_frame(frame, direction, &self.ctx).await?;
                tracing::debug!(processor = %self.proc_name, "received Cancel, shutting down");
                return Ok(true);
            }

            Frame::End(_) => {
                self.processor.process_frame(frame, direction, &self.ctx).await?;
                tracing::debug!(processor = %self.proc_name, "received End, shutting down");
                return Ok(true);
            }

            _ => {
                if self.is_sync {
                    self.processor.process_frame_sync(frame, direction, &self.ctx)?;
                } else {
                    self.processor.process_frame(frame, direction, &self.ctx).await?;
                }
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::PassthroughProcessor;
    use pipecat_core::{FrameDirection, FrameHeader, TextData};

    /// Helper: create a FrameQueueSender backed by a dummy queue (for test
    /// output channels where we just collect what comes out).
    fn dummy_output_sender(name: &str) -> (FrameQueueSender, BoundedFrameQueue) {
        BoundedFrameQueue::new(
            QueueConfig {
                max_size: 1000,
                ..Default::default()
            },
            name,
        )
    }

    #[tokio::test]
    async fn node_processes_start_and_end() {
        let (ds_sender, mut ds_queue) = dummy_output_sender("ds");
        let (us_sender, _us_queue) = dummy_output_sender("us");
        let cancel = CancellationToken::new();

        let processor = PassthroughProcessor::new("test_node");
        let (input_sender, node) = ProcessorNode::new(
            Box::new(processor),
            QueueConfig::default(),
            ds_sender,
            us_sender,
            PipelineClock::new(),
            cancel.clone(),
        );

        // Feed Start then End to trigger full lifecycle.
        input_sender
            .send(FrameEnvelope {
                frame: Frame::Start(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();
        input_sender
            .send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        node.run().await.unwrap();

        // Both frames should appear on the downstream queue.
        let env1 = ds_queue.get_nowait().unwrap();
        assert_eq!(env1.frame.name(), "Start");
        let env2 = ds_queue.get_nowait().unwrap();
        assert_eq!(env2.frame.name(), "End");
    }

    #[tokio::test]
    async fn node_drains_on_interruption() {
        let (ds_sender, mut ds_queue) = dummy_output_sender("ds");
        let (us_sender, _us_queue) = dummy_output_sender("us");
        let cancel = CancellationToken::new();

        let processor = PassthroughProcessor::new("drain_test");
        let (input_sender, node) = ProcessorNode::new(
            Box::new(processor),
            QueueConfig {
                max_size: 100,
                ..Default::default()
            },
            ds_sender,
            us_sender,
            PipelineClock::new(),
            cancel.clone(),
        );

        // Feed: data, data, interruption, end
        input_sender
            .send(FrameEnvelope {
                frame: Frame::Text {
                    header: FrameHeader::new(),
                    data: TextData {
                        text: "should_be_drained".into(),
                    },
                },
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();
        input_sender
            .send(FrameEnvelope {
                frame: Frame::Text {
                    header: FrameHeader::new(),
                    data: TextData {
                        text: "also_drained".into(),
                    },
                },
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        // Interruption is a system frame, dequeued with priority.
        input_sender
            .send(FrameEnvelope {
                frame: Frame::Interruption(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        // End goes to data channel (Control class). After drain, it survives
        // as uninterruptible.
        input_sender
            .send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        node.run().await.unwrap();

        // Collect all frames that came out downstream.
        let mut names = Vec::new();
        while let Some(env) = ds_queue.get_nowait() {
            names.push(env.frame.name().to_string());
        }

        // Interruption is processed first (system priority), draining Text frames.
        // Then End is processed, causing shutdown.
        assert!(names.contains(&"Interruption".to_string()));
        assert!(names.contains(&"End".to_string()));
        assert!(!names.contains(&"Text".to_string()));
    }

    #[tokio::test]
    async fn node_cancellation_token() {
        let (ds_sender, _ds_queue) = dummy_output_sender("ds");
        let (us_sender, _us_queue) = dummy_output_sender("us");
        let cancel = CancellationToken::new();

        let processor = PassthroughProcessor::new("cancel_test");
        let (_input_sender, node) = ProcessorNode::new(
            Box::new(processor),
            QueueConfig::default(),
            ds_sender,
            us_sender,
            PipelineClock::new(),
            cancel.clone(),
        );

        // Cancel immediately.
        cancel.cancel();
        node.run().await.unwrap();
    }
}
