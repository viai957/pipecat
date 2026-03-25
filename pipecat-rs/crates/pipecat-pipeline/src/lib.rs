//! # pipecat-pipeline
//!
//! Pipeline engine for Pipecat real-time AI pipelines.
//!
//! This crate provides the runtime that chains [`FrameProcessor`] instances
//! into a linear pipeline, spawns each as a tokio task, and manages the full
//! lifecycle from `StartFrame` to `EndFrame`.
//!
//! ## Key types
//!
//! - [`FrameProcessor`] -- trait implemented by every pipeline stage.
//! - [`ProcessorContext`] -- context passed to processors for pushing frames.
//! - [`Pipeline`] -- assembles processors into a chain and spawns tasks.
//! - [`PipelineTask`] -- wraps a pipeline with lifecycle management, heartbeat,
//!   and external frame injection.
//! - [`PipelineRunner`] -- high-level entry point with signal handling.
//! - [`BoundedFrameQueue`] -- bounded queue with system-frame bypass.
//! - [`Observer`] -- trait for monitoring frame flow.

pub mod backpressure;
mod node;
pub mod observer;
pub mod pipeline;
pub mod processor;
pub mod runner;
pub mod task;

// Re-export the most commonly used types at the crate root.
pub use backpressure::{
    BackpressureConfig, BoundedFrameQueue, DropPolicy, FrameEnvelope, FrameQueueSender, QueueConfig,
    QueueMetrics,
};
pub use observer::{FrameProcessed, FramePushed, NullObserver, Observer};
pub use pipeline::{Pipeline, PipelineHandle};
pub use processor::{FrameProcessor, PassthroughProcessor, ProcessorContext};
pub use runner::PipelineRunner;
pub use task::{PipelineEndReason, PipelineTask, PipelineTaskParams};
