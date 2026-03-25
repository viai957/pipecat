//! High-level entry point for running pipeline tasks.
//!
//! [`PipelineRunner`] handles signal management (Ctrl-C) and provides a clean
//! API for executing one or more pipeline tasks.

use pipecat_core::{Frame, FrameHeader, Result};

use crate::task::PipelineTask;

/// High-level runner that executes a [`PipelineTask`] with signal handling.
///
/// On receiving `SIGINT` (Ctrl-C), the runner sends a `CancelFrame` to the
/// task, triggering graceful shutdown.
pub struct PipelineRunner;

impl PipelineRunner {
    /// Run a single pipeline task to completion.
    ///
    /// Installs a Ctrl-C handler that cancels the task on the first signal.
    /// Returns when the task finishes (either naturally via `EndFrame` or via
    /// cancellation).
    pub async fn run(task: &mut PipelineTask) -> Result<crate::task::PipelineEndReason> {
        let sender = task.frame_sender();

        // Spawn Ctrl-C handler.
        let ctrlc_handle = tokio::spawn(async move {
            match tokio::signal::ctrl_c().await {
                Ok(()) => {
                    tracing::info!("pipeline runner: Ctrl-C received, sending CancelFrame");
                    let env = crate::backpressure::FrameEnvelope {
                        frame: Frame::Cancel(FrameHeader::new()),
                        direction: pipecat_core::FrameDirection::Downstream,
                    };
                    let _ = sender.send(env);
                }
                Err(e) => {
                    tracing::warn!("pipeline runner: failed to listen for Ctrl-C: {e}");
                }
            }
        });

        let result = task.run().await;

        // Abort the Ctrl-C handler if the task finished before a signal.
        ctrlc_handle.abort();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backpressure::FrameEnvelope;
    use crate::pipeline::Pipeline;
    use crate::processor::PassthroughProcessor;
    use crate::task::PipelineTaskParams;
    use pipecat_core::FrameDirection;
    use std::time::Duration;

    #[tokio::test]
    async fn runner_basic_lifecycle() {
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
                frame: Frame::End(FrameHeader::new()),
                direction: FrameDirection::Downstream,
            });
        });

        PipelineRunner::run(&mut task).await.unwrap();
    }
}
