//! Pipeline task: wraps a [`Pipeline`] with lifecycle management, heartbeat
//! monitoring, idle detection, and external frame injection.
//!
//! A [`PipelineTask`] is the primary entry point for running a pipeline. It
//! injects `StartFrame` at the beginning and listens for `EndFrame`/`StopFrame`
//! at the sink to know when the pipeline has finished.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;

use pipecat_core::clock::PipelineClock;
use pipecat_core::{Frame, FrameDirection, FrameHeader, PipecatError, Result};

use crate::backpressure::FrameEnvelope;
use crate::observer::Observer;
use crate::pipeline::Pipeline;

/// How the pipeline terminated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineEndReason {
    /// Pipeline received an EndFrame at the sink (graceful end).
    End,
    /// Pipeline received a StopFrame at the sink (graceful stop, no cleanup).
    Stop,
    /// Pipeline received a CancelFrame at the sink (immediate cancellation).
    Cancel,
    /// Pipeline encountered a fatal error at the sink.
    FatalError,
    /// The sink channel closed unexpectedly.
    SinkClosed,
    /// The pipeline was cancelled externally (CancellationToken).
    ExternalCancel,
}

impl PipelineEndReason {
    /// Human-readable name for the end reason (matches Frame variant names).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::End => "End",
            Self::Stop => "Stop",
            Self::Cancel => "Cancel",
            Self::FatalError => "FatalError",
            Self::SinkClosed => "SinkClosed",
            Self::ExternalCancel => "ExternalCancel",
        }
    }
}

/// Parameters for configuring a [`PipelineTask`].
#[derive(Debug, Clone)]
pub struct PipelineTaskParams {
    /// Interval between heartbeat frames. Set to `None` to disable.
    pub heartbeat_interval: Option<Duration>,
    /// Maximum time the pipeline can be idle (no frames processed at the sink)
    /// before it is considered hung. Set to `None` to disable.
    pub idle_timeout: Option<Duration>,
}

impl Default for PipelineTaskParams {
    fn default() -> Self {
        Self {
            heartbeat_interval: Some(Duration::from_secs(5)),
            idle_timeout: None,
        }
    }
}

/// Wraps a [`Pipeline`] with full lifecycle management.
///
/// The task:
/// 1. Starts the pipeline (spawns processor tasks).
/// 2. Sends a `StartFrame` downstream.
/// 3. Optionally starts a heartbeat task.
/// 4. Monitors the sink for `EndFrame`/`StopFrame` to know when to shut down.
/// 5. Provides `queue_frame` for external code to inject frames.
pub struct PipelineTask {
    pipeline: Option<Pipeline>,
    params: PipelineTaskParams,
    clock: PipelineClock,
    observers: Vec<Box<dyn Observer>>,
    /// External queue for injecting frames into the pipeline source.
    external_tx: mpsc::UnboundedSender<FrameEnvelope>,
    external_rx: Option<mpsc::UnboundedReceiver<FrameEnvelope>>,
    /// Signalled when the pipeline reaches its end.
    end_notify: Arc<Notify>,
}

impl PipelineTask {
    /// Create a new task wrapping the given pipeline.
    pub fn new(pipeline: Pipeline, params: PipelineTaskParams) -> Self {
        let (external_tx, external_rx) = mpsc::unbounded_channel();
        Self {
            pipeline: Some(pipeline),
            params,
            clock: PipelineClock::new(),
            observers: Vec::new(),
            external_tx,
            external_rx: Some(external_rx),
            end_notify: Arc::new(Notify::new()),
        }
    }

    /// Add an observer that will be notified of frame flow events.
    pub fn add_observer(&mut self, observer: Box<dyn Observer>) {
        self.observers.push(observer);
    }

    /// Queue a frame for injection into the pipeline source.
    ///
    /// This can be called from any task/thread. The frame will be delivered
    /// to the first processor in the pipeline.
    pub fn queue_frame(&self, frame: Frame) -> Result<()> {
        self.external_tx
            .send(FrameEnvelope {
                frame,
                direction: FrameDirection::Downstream,
            })
            .map_err(|e| PipecatError::Pipeline(format!("queue_frame failed: {e}")))?;
        Ok(())
    }

    /// Queue a frame traveling upstream into the pipeline sink.
    pub fn queue_upstream_frame(&self, frame: Frame) -> Result<()> {
        self.external_tx
            .send(FrameEnvelope {
                frame,
                direction: FrameDirection::Upstream,
            })
            .map_err(|e| PipecatError::Pipeline(format!("queue_upstream_frame failed: {e}")))?;
        Ok(())
    }

    /// Get a clone of the external frame sender for use in other tasks.
    pub fn frame_sender(&self) -> mpsc::UnboundedSender<FrameEnvelope> {
        self.external_tx.clone()
    }

    /// Cancel the pipeline by sending a `CancelFrame`.
    pub fn cancel(&self) -> Result<()> {
        self.queue_frame(Frame::Cancel(FrameHeader::new()))
    }

    /// Run the pipeline to completion.
    ///
    /// This method:
    /// 1. Starts the pipeline (spawns all processor nodes).
    /// 2. Sends `StartFrame` downstream.
    /// 3. Starts the heartbeat task (if configured).
    /// 4. Forwards externally queued frames into the pipeline.
    /// 5. Monitors the sink for `EndFrame` or `StopFrame`.
    /// 6. Returns when the pipeline ends or is cancelled.
    pub async fn run(&mut self) -> Result<PipelineEndReason> {
        let pipeline = self
            .pipeline
            .take()
            .ok_or_else(|| PipecatError::Pipeline("pipeline already consumed".into()))?;

        let mut external_rx = self
            .external_rx
            .take()
            .ok_or_else(|| PipecatError::Pipeline("task already running".into()))?;

        let clock = self.clock.clone();
        let mut handle = pipeline.start(clock.clone());

        // Send StartFrame
        handle
            .source_tx
            .send(FrameEnvelope {
                frame: Frame::Start(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            })
            .await
            .map_err(|_| PipecatError::Pipeline("failed to send StartFrame".into()))?;

        tracing::debug!("pipeline task: StartFrame sent");

        // Heartbeat task
        let mut heartbeat_handle: Option<JoinHandle<()>> = None;
        if let Some(interval) = self.params.heartbeat_interval {
            let source_tx = handle.source_tx.clone();
            let cancel = handle.cancellation.clone();
            heartbeat_handle = Some(tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                ticker.tick().await; // skip first immediate tick
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => break,
                        _ = ticker.tick() => {
                            let env = FrameEnvelope {
                                frame: Frame::Heartbeat(FrameHeader::new()),
                                direction: FrameDirection::Downstream,
                            };
                            if source_tx.send(env).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }));
        }

        // External frame forwarder task
        let source_tx_ext = handle.source_tx.clone();
        let upstream_tx_ext = handle.upstream_tx.clone();
        let cancel_ext = handle.cancellation.clone();
        let ext_forwarder = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = cancel_ext.cancelled() => break,
                    maybe_env = external_rx.recv() => {
                        match maybe_env {
                            Some(env) => {
                                match env.direction {
                                    FrameDirection::Downstream => {
                                        if source_tx_ext.send(env).await.is_err() {
                                            break;
                                        }
                                    }
                                    FrameDirection::Upstream => {
                                        if upstream_tx_ext.send(env).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
        });

        // Sink monitor: watch for End/Stop frames
        let end_notify = Arc::clone(&self.end_notify);
        let cancellation = handle.cancellation.clone();
        let end_reason;

        loop {
            tokio::select! {
                biased;

                _ = cancellation.cancelled() => {
                    tracing::debug!("pipeline task: cancellation received");
                    end_reason = PipelineEndReason::ExternalCancel;
                    break;
                }

                maybe_env = handle.sink_rx.get() => {
                    let Some(env) = maybe_env else {
                        tracing::debug!("pipeline task: sink queue closed");
                        end_reason = PipelineEndReason::SinkClosed;
                        break;
                    };
                    match &env.frame {
                        Frame::End(_) => {
                            tracing::debug!("pipeline task: EndFrame reached sink");
                            end_notify.notify_waiters();
                            end_reason = PipelineEndReason::End;
                            break;
                        }
                        Frame::Stop(_) => {
                            tracing::debug!("pipeline task: StopFrame reached sink");
                            end_notify.notify_waiters();
                            end_reason = PipelineEndReason::Stop;
                            break;
                        }
                        Frame::Cancel(_) => {
                            tracing::debug!("pipeline task: CancelFrame reached sink");
                            end_notify.notify_waiters();
                            end_reason = PipelineEndReason::Cancel;
                            break;
                        }
                        Frame::Error { error, .. } => {
                            if error.fatal {
                                tracing::error!(
                                    msg = %error.message,
                                    "pipeline task: fatal error at sink"
                                );
                                end_notify.notify_waiters();
                                end_reason = PipelineEndReason::FatalError;
                                break;
                            } else {
                                tracing::warn!(
                                    msg = %error.message,
                                    "pipeline task: non-fatal error at sink"
                                );
                            }
                        }
                        Frame::Heartbeat(_) => {
                            tracing::trace!("pipeline task: heartbeat reached sink");
                        }
                        _ => {
                            tracing::trace!(
                                frame = env.frame.name(),
                                "pipeline task: frame reached sink"
                            );
                        }
                    }
                }

                maybe_env = handle.upstream_rx.get() => {
                    let Some(env) = maybe_env else {
                        tracing::debug!("pipeline task: upstream queue closed");
                        continue;
                    };
                    match &env.frame {
                        Frame::InterruptionTask(_) => {
                            // Convert to InterruptionFrame and push downstream.
                            let interrupt_env = FrameEnvelope {
                                frame: Frame::Interruption(FrameHeader::new()),
                                direction: FrameDirection::Downstream,
                            };
                            let _ = handle.source_tx.send(interrupt_env).await;
                            tracing::debug!("pipeline task: InterruptionTaskFrame -> InterruptionFrame");
                        }
                        Frame::EndTask(_) => {
                            // Request graceful pipeline closure: convert to EndFrame downstream.
                            let end_env = FrameEnvelope {
                                frame: Frame::End(FrameHeader::new()),
                                direction: FrameDirection::Downstream,
                            };
                            let _ = handle.source_tx.send(end_env).await;
                            tracing::debug!("pipeline task: EndTaskFrame -> EndFrame");
                        }
                        Frame::CancelTask(_) => {
                            // Request immediate pipeline cancellation: convert to CancelFrame downstream.
                            let cancel_env = FrameEnvelope {
                                frame: Frame::Cancel(FrameHeader::new()),
                                direction: FrameDirection::Downstream,
                            };
                            let _ = handle.source_tx.send(cancel_env).await;
                            tracing::debug!("pipeline task: CancelTaskFrame -> CancelFrame");
                        }
                        Frame::StopTask(_) => {
                            // Request graceful pipeline stop: convert to StopFrame downstream.
                            let stop_env = FrameEnvelope {
                                frame: Frame::Stop(FrameHeader::new()),
                                direction: FrameDirection::Downstream,
                            };
                            let _ = handle.source_tx.send(stop_env).await;
                            tracing::debug!("pipeline task: StopTaskFrame -> StopFrame");
                        }
                        Frame::Error { error, .. } => {
                            tracing::warn!(
                                msg = %error.message,
                                "pipeline task: error received upstream"
                            );
                        }
                        _ => {
                            tracing::trace!(
                                frame = env.frame.name(),
                                "pipeline task: upstream frame at source"
                            );
                        }
                    }
                }
            }
        }

        // Cleanup
        handle.cancellation.cancel();

        if let Some(hb) = heartbeat_handle {
            let _ = hb.await;
        }
        ext_forwarder.abort();

        // Wait for all node tasks to finish (with a timeout).
        for jh in handle.join_handles {
            let _ = tokio::time::timeout(Duration::from_secs(5), jh).await;
        }

        tracing::debug!(reason = end_reason.as_str(), "pipeline task: shutdown complete");
        Ok(end_reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Pipeline;
    use crate::processor::PassthroughProcessor;
    use pipecat_core::TextData;

    #[tokio::test]
    async fn task_lifecycle_start_end() {
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut task = PipelineTask::new(pipeline, PipelineTaskParams::default());

        // Schedule an EndFrame after a short delay.
        let sender = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = sender.send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            });
        });

        task.run().await.unwrap();
    }

    #[tokio::test]
    async fn task_cancel() {
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                heartbeat_interval: None,
                idle_timeout: None,
            },
        );

        let sender = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = sender.send(FrameEnvelope {
                frame: Frame::Cancel(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            });
        });

        task.run().await.unwrap();
    }

    #[tokio::test]
    async fn task_queue_frame() {
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                heartbeat_interval: None,
                idle_timeout: None,
            },
        );

        // Queue a text frame then an end frame from another task.
        let tx = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            let _ = tx.send(FrameEnvelope {
                frame: Frame::Text {
                    header: FrameHeader::new(),
                    data: TextData {
                        text: "injected".into(),
                    },
                },
                direction: FrameDirection::Downstream,
            });
            tokio::time::sleep(Duration::from_millis(30)).await;
            let _ = tx.send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            });
        });

        task.run().await.unwrap();
    }

    #[tokio::test]
    async fn task_multi_processor() {
        let p1 = PassthroughProcessor::new("p1");
        let p2 = PassthroughProcessor::new("p2");
        let pipeline = Pipeline::new(vec![Box::new(p1), Box::new(p2)]);
        let mut task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                heartbeat_interval: None,
                idle_timeout: None,
            },
        );

        let tx = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(FrameEnvelope {
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            });
        });

        task.run().await.unwrap();
    }

    #[tokio::test]
    async fn task_end_task_upstream_converts_to_end() {
        // EndTaskFrame sent upstream should be converted to EndFrame downstream,
        // causing the pipeline to shut down gracefully.
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                heartbeat_interval: None,
                idle_timeout: None,
            },
        );

        let tx = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            // Send EndTask UPSTREAM — should be converted to End DOWNSTREAM
            let _ = tx.send(FrameEnvelope {
                frame: Frame::EndTask(FrameHeader::new()),
                direction: FrameDirection::Upstream,
            });
        });

        task.run().await.unwrap();
    }

    #[tokio::test]
    async fn task_cancel_task_upstream_converts_to_cancel() {
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                heartbeat_interval: None,
                idle_timeout: None,
            },
        );

        let tx = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(FrameEnvelope {
                frame: Frame::CancelTask(FrameHeader::new()),
                direction: FrameDirection::Upstream,
            });
        });

        task.run().await.unwrap();
    }

    #[tokio::test]
    async fn task_stop_task_upstream_converts_to_stop() {
        let proc = PassthroughProcessor::new("p1");
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let mut task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                heartbeat_interval: None,
                idle_timeout: None,
            },
        );

        let tx = task.frame_sender();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send(FrameEnvelope {
                frame: Frame::StopTask(FrameHeader::new()),
                direction: FrameDirection::Upstream,
            });
        });

        task.run().await.unwrap();
    }
}
