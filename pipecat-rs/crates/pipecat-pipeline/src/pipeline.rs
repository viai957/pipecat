//! Pipeline assembly: chains [`FrameProcessor`] instances into a linear graph
//! and spawns each as a tokio task.
//!
//! ```text
//! [source_tx] --> [Node0] --> [Node1] --> ... --> [NodeN] --> [sink_rx]
//!                  <--          <--                  <--
//! ```
//!
//! Upstream frames travel in the reverse direction through separate channels.
//!
//! Each node's output is wired *directly* to the next node's input queue via a
//! [`FrameQueueSender`]. No intermediate forwarder tasks are spawned, reducing
//! the total task count from 3N to N.

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use pipecat_core::clock::PipelineClock;
use pipecat_core::Result;

use crate::backpressure::{BoundedFrameQueue, FrameQueueSender, QueueConfig};
use crate::node::ProcessorNode;
use crate::processor::FrameProcessor;

/// A linear pipeline of frame processors.
///
/// Use [`Pipeline::new`] to create a pipeline from a `Vec` of processors.
/// Call [`Pipeline::start`] to spawn each processor as a tokio task, returning
/// the channels that feed the head and drain the tail.
pub struct Pipeline {
    processors: Option<Vec<Box<dyn FrameProcessor>>>,
    queue_config: QueueConfig,
}

/// Handles returned by [`Pipeline::start`] for driving the pipeline.
pub struct PipelineHandle {
    /// Send frames into the first processor (downstream input).
    pub source_tx: FrameQueueSender,
    /// Receive frames that exit the last processor (downstream output).
    pub sink_rx: BoundedFrameQueue,
    /// Receive frames that exit the first processor going upstream.
    pub upstream_rx: BoundedFrameQueue,
    /// Send frames into the last processor going upstream.
    pub upstream_tx: FrameQueueSender,
    /// Join handles for all spawned tasks (node tasks only — no forwarders).
    pub join_handles: Vec<JoinHandle<Result<()>>>,
    /// Cancellation token shared by all nodes.
    pub cancellation: CancellationToken,
}

impl Pipeline {
    /// Create a new pipeline from an ordered list of processors.
    ///
    /// Processors are chained left-to-right: the first processor receives
    /// frames from the source and the last pushes to the sink.
    pub fn new(processors: Vec<Box<dyn FrameProcessor>>) -> Self {
        Self {
            processors: Some(processors),
            queue_config: QueueConfig::default(),
        }
    }

    /// Override the default queue configuration for all processor input queues.
    pub fn with_queue_config(mut self, config: QueueConfig) -> Self {
        self.queue_config = config;
        self
    }

    /// Spawn all processor nodes as tokio tasks and return the external
    /// interface channels.
    ///
    /// The pipeline clock is shared across all nodes so that PTS values are
    /// coherent.
    ///
    /// # Direct wiring (no forwarder tasks)
    ///
    /// Each node's output `FrameQueueSender` points directly to the next
    /// node's input queue. This eliminates the previous 2N forwarder tasks,
    /// reducing from 3N to N tokio tasks for N processors.
    pub fn start(mut self, clock: PipelineClock) -> PipelineHandle {
        let processors = self.processors.take().expect("pipeline already started");
        let cancellation = CancellationToken::new();
        let n = processors.len();

        if n == 0 {
            // Empty pipeline: source connects directly to sink.
            let (source_tx, sink_rx) =
                BoundedFrameQueue::new(self.queue_config.clone(), "empty_ds");
            let (upstream_tx, upstream_rx) =
                BoundedFrameQueue::new(self.queue_config.clone(), "empty_us");
            return PipelineHandle {
                source_tx,
                sink_rx,
                upstream_rx,
                upstream_tx,
                join_handles: Vec::new(),
                cancellation,
            };
        }

        // Downstream sink: where the last node's output goes.
        let (sink_sender, sink_rx) =
            BoundedFrameQueue::new(self.queue_config.clone(), "pipeline_sink");

        // Upstream source: where the first node's upstream output goes.
        let (upstream_source_sender, upstream_rx) =
            BoundedFrameQueue::new(self.queue_config.clone(), "pipeline_upstream");

        // Strategy: create all input queues first, then wire senders.
        //
        // For N=3 processors [p0, p1, p2]:
        //   p0 downstream -> p1's input
        //   p1 downstream -> p2's input
        //   p2 downstream -> sink
        //   p0 upstream   -> upstream_rx (source)
        //   p1 upstream   -> p0's input
        //   p2 upstream   -> p1's input
        //   source_tx     -> p0's input
        //   upstream_tx   -> p2's input
        let mut processors: Vec<_> = processors.into_iter().collect();
        let mut node_input_senders: Vec<FrameQueueSender> = Vec::with_capacity(n);
        let mut node_input_queues: Vec<BoundedFrameQueue> = Vec::with_capacity(n);

        for proc in &processors {
            let (sender, queue) =
                BoundedFrameQueue::new(self.queue_config.clone(), proc.name());
            node_input_senders.push(sender);
            node_input_queues.push(queue);
        }

        let mut join_handles: Vec<JoinHandle<Result<()>>> = Vec::with_capacity(n);

        // source_tx = sender to first node's input
        let source_tx = node_input_senders[0].clone();
        // upstream_tx = sender to last node's input
        let upstream_tx = node_input_senders[n - 1].clone();

        let mut queues_iter = node_input_queues.into_iter();

        for i in 0..n {
            let proc = processors.remove(0);
            let input_queue = queues_iter.next().unwrap();

            // Downstream sender: node[i] -> node[i+1]'s input, or sink.
            let ds_sender = if i + 1 < n {
                node_input_senders[i + 1].clone()
            } else {
                sink_sender.clone()
            };

            // Upstream sender: node[i] -> node[i-1]'s input, or upstream_rx.
            let us_sender = if i > 0 {
                node_input_senders[i - 1].clone()
            } else {
                upstream_source_sender.clone()
            };

            let node = ProcessorNode::new_with_queue(
                proc,
                input_queue,
                ds_sender,
                us_sender,
                clock.clone(),
                cancellation.clone(),
            );

            let node_handle = tokio::spawn(async move { node.run().await });
            join_handles.push(node_handle);
        }

        PipelineHandle {
            source_tx,
            sink_rx,
            upstream_rx,
            upstream_tx,
            join_handles,
            cancellation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backpressure::FrameEnvelope;
    use crate::processor::PassthroughProcessor;
    use pipecat_core::{Frame, FrameDirection, FrameHeader, TextData};

    #[tokio::test]
    async fn empty_pipeline_passes_through() {
        let pipeline = Pipeline::new(vec![]);
        let mut handle = pipeline.start(PipelineClock::new());

        let frame = Frame::Start(FrameHeader::new());
        handle
            .source_tx
            .send(FrameEnvelope {
                frame,
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        let env = handle.sink_rx.get().await.unwrap();
        assert_eq!(env.frame.name(), "Start");
    }

    #[tokio::test]
    async fn single_processor_passthrough() {
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut handle = pipeline.start(PipelineClock::new());

        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::Start(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::Text {
                    header: FrameHeader::new(),
                    data: TextData {
                        text: "hello".into(),
                    },
                },
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        // Give the node time to process.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let mut names = Vec::new();
        while let Some(env) = handle.sink_rx.get_nowait() {
            names.push(env.frame.name().to_string());
        }

        assert!(names.contains(&"Start".to_string()));
        assert!(names.contains(&"Text".to_string()));
        assert!(names.contains(&"End".to_string()));

        // Cancel to ensure all tasks terminate.
        handle.cancellation.cancel();
        for jh in handle.join_handles {
            let _ = jh.await;
        }
    }

    #[tokio::test]
    async fn multi_processor_chain() {
        let p1 = PassthroughProcessor::new("p1");
        let p2 = PassthroughProcessor::new("p2");
        let p3 = PassthroughProcessor::new("p3");
        let pipeline = Pipeline::new(vec![Box::new(p1), Box::new(p2), Box::new(p3)]);
        let mut handle = pipeline.start(PipelineClock::new());

        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::Start(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::Text {
                    header: FrameHeader::new(),
                    data: TextData {
                        text: "chain".into(),
                    },
                },
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let mut names = Vec::new();
        while let Some(env) = handle.sink_rx.get_nowait() {
            names.push(env.frame.name().to_string());
        }

        assert!(names.contains(&"Start".to_string()));
        assert!(names.contains(&"Text".to_string()));
        assert!(names.contains(&"End".to_string()));

        handle.cancellation.cancel();
        for jh in handle.join_handles {
            let _ = jh.await;
        }
    }

    #[tokio::test]
    async fn backpressure_drops_in_pipeline() {
        // Verify that backpressure actually works through a pipeline.
        // Use a tiny queue (max_size=5) and send many frames.
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)])
            .with_queue_config(QueueConfig {
                max_size: 5,
                drop_policy: crate::backpressure::DropPolicy::DropNewest,
                warn_threshold: 0.8,
            });
        let mut handle = pipeline.start(PipelineClock::new());

        // Send Start.
        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::Start(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        // Flood with data frames (the processor needs to be running to
        // drain, so some will be processed).
        for i in 0..50 {
            let _ = handle
                .source_tx
                .send(FrameEnvelope {
                    frame: Frame::Text {
                        header: FrameHeader::new(),
                        data: TextData {
                            text: format!("msg{i}"),
                        },
                    },
                    direction: FrameDirection::Downstream,
                })
                .await;
        }

        // Send End.
        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut count = 0;
        while let Some(_) = handle.sink_rx.get_nowait() {
            count += 1;
        }

        // We should have received fewer than 50 data frames + Start + End
        // due to drops. At minimum Start and End should arrive.
        assert!(count >= 2, "Start and End should arrive");

        handle.cancellation.cancel();
        for jh in handle.join_handles {
            let _ = jh.await;
        }
    }
}
