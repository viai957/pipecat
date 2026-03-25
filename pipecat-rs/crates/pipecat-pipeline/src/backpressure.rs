//! Bounded frame queue with system-frame bypass and configurable drop policies.
//!
//! The queue is split into a **sender** ([`FrameQueueSender`]) and a **consumer**
//! ([`BoundedFrameQueue`]). System frames (heartbeats, interruptions, errors,
//! etc.) are routed through an unbounded channel so they are never dropped. Data
//! and control frames go through a bounded `mpsc` channel that enforces a
//! configurable [`DropPolicy`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;

use pipecat_core::{Frame, FrameClass, FrameDirection};

/// Policy applied when the bounded data queue reaches its capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    /// Never drop frames -- `send` will block until space is available.
    Never,
    /// Drop the oldest frame in the queue to make room.
    DropOldest,
    /// Drop the incoming (newest) frame.
    DropNewest,
    /// Block the caller until space is available (same as `Never` but semantically
    /// signals intent).
    Block,
}

/// Configuration for a single bounded queue.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Maximum number of data/control frames in the bounded portion.
    pub max_size: usize,
    /// What to do when the queue is full.
    pub drop_policy: DropPolicy,
    /// Fraction (0.0 .. 1.0) of `max_size` at which a warning is logged.
    pub warn_threshold: f32,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_size: 100,
            drop_policy: DropPolicy::DropOldest,
            warn_threshold: 0.8,
        }
    }
}

/// Aggregate backpressure configuration for all queues in a processor.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    pub input_queue: QueueConfig,
    pub process_queue: QueueConfig,
    pub push_queue: QueueConfig,
    pub audio_queue: QueueConfig,
    pub video_queue: QueueConfig,
    pub observer_queue: QueueConfig,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            input_queue: QueueConfig {
                max_size: 200,
                ..Default::default()
            },
            process_queue: QueueConfig {
                max_size: 100,
                ..Default::default()
            },
            push_queue: QueueConfig {
                max_size: 500,
                ..Default::default()
            },
            audio_queue: QueueConfig {
                max_size: 50,
                ..Default::default()
            },
            video_queue: QueueConfig {
                max_size: 10,
                ..Default::default()
            },
            observer_queue: QueueConfig {
                max_size: 50,
                ..Default::default()
            },
        }
    }
}

/// A frame packaged with its intended travel direction.
pub struct FrameEnvelope {
    pub frame: Frame,
    pub direction: FrameDirection,
}

impl std::fmt::Debug for FrameEnvelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameEnvelope")
            .field("frame", &self.frame.name())
            .field("direction", &self.direction)
            .finish()
    }
}

/// Cumulative statistics for a [`BoundedFrameQueue`].
#[derive(Debug, Default)]
pub struct QueueMetrics {
    /// Total number of frames successfully enqueued (system + data).
    pub total_enqueued: u64,
    /// Total number of data frames dropped due to drop policy.
    pub total_dropped: u64,
    /// Maximum observed occupancy of the data portion.
    pub high_water_mark: usize,
}

/// Shared drop counter between sender and consumer.
struct SharedMetrics {
    total_dropped: AtomicU64,
}

/// Producer half of a bounded frame queue.
///
/// Routes system frames through an unbounded channel and data/control frames
/// through a bounded `mpsc` channel with the configured [`DropPolicy`].
///
/// Multiple clones of a `FrameQueueSender` can exist (it is `Clone`), but in
/// practice each inter-node link is SPSC.
#[derive(Clone)]
pub struct FrameQueueSender {
    name: Arc<str>,
    system_tx: mpsc::UnboundedSender<FrameEnvelope>,
    data_tx: mpsc::Sender<FrameEnvelope>,
    config: QueueConfig,
    shared_metrics: Arc<SharedMetrics>,
}

impl FrameQueueSender {
    /// Send a frame, routing it to the system or data channel based on
    /// classification.
    ///
    /// - **System** frames bypass backpressure entirely (unbounded channel).
    /// - **Uninterruptible** frames (End, Stop, Cancel) go through the bounded
    ///   channel to preserve ordering with data frames, but they block-wait
    ///   instead of being dropped — pipeline lifecycle signals must never be lost.
    /// - **Data** and other **Control** frames go through the bounded channel
    ///   with the configured [`DropPolicy`].
    pub async fn send(&self, envelope: FrameEnvelope) -> Result<(), FrameEnvelope> {
        if envelope.frame.classification() == FrameClass::System {
            self.system_tx.send(envelope).map_err(|e| e.0)?;
            return Ok(());
        }

        // Uninterruptible frames (End, Stop, Cancel) must never be dropped.
        // Send them through the bounded channel (preserves ordering with data)
        // but block-wait if full instead of applying drop policy.
        if envelope.frame.is_uninterruptible() {
            self.data_tx.send(envelope).await.map_err(|e| e.0)?;
            return Ok(());
        }

        // Data/control frame — apply drop policy on the bounded channel.
        match self.config.drop_policy {
            DropPolicy::DropNewest => match self.data_tx.try_send(envelope) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_env)) => {
                    self.shared_metrics
                        .total_dropped
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        queue = %self.name,
                        "queue full, dropped newest frame"
                    );
                }
                Err(mpsc::error::TrySendError::Closed(env)) => return Err(env),
            },
            DropPolicy::DropOldest => match self.data_tx.try_send(envelope) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(env)) => {
                    self.shared_metrics
                        .total_dropped
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        queue = %self.name,
                        "queue full, dropped frame (bounded channel)"
                    );
                    drop(env);
                }
                Err(mpsc::error::TrySendError::Closed(env)) => return Err(env),
            },
            DropPolicy::Never | DropPolicy::Block => {
                // Blocking send — waits until space is available.
                self.data_tx.send(envelope).await.map_err(|e| e.0)?;
            }
        }

        Ok(())
    }

    /// Non-blocking send with drop-policy handling.
    ///
    /// Behaves like [`send`](Self::send) but never awaits. For `DropNewest`
    /// and `DropOldest` policies this is always non-blocking. For `Block`/`Never`
    /// it falls back to `try_send` and returns `Err` if the channel is full,
    /// so callers that need blocking semantics should use the async [`send`].
    #[allow(clippy::result_large_err)]
    pub fn try_send(&self, envelope: FrameEnvelope) -> Result<(), FrameEnvelope> {
        if envelope.frame.classification() == FrameClass::System {
            self.system_tx.send(envelope).map_err(|e| e.0)?;
            return Ok(());
        }
        match self.config.drop_policy {
            DropPolicy::DropNewest => match self.data_tx.try_send(envelope) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_env)) => {
                    self.shared_metrics
                        .total_dropped
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(()) // silently drop newest
                }
                Err(mpsc::error::TrySendError::Closed(env)) => Err(env),
            },
            DropPolicy::DropOldest => match self.data_tx.try_send(envelope) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(env)) => {
                    self.shared_metrics
                        .total_dropped
                        .fetch_add(1, Ordering::Relaxed);
                    drop(env);
                    Ok(())
                }
                Err(mpsc::error::TrySendError::Closed(env)) => Err(env),
            },
            DropPolicy::Never | DropPolicy::Block => self
                .data_tx
                .try_send(envelope)
                .map_err(|e| match e {
                    mpsc::error::TrySendError::Full(env)
                    | mpsc::error::TrySendError::Closed(env) => env,
                }),
        }
    }
}

/// Consumer half of a bounded frame queue.
///
/// Gives system frames absolute priority over data/control frames.
pub struct BoundedFrameQueue {
    #[allow(dead_code)]
    name: String,
    system_rx: mpsc::UnboundedReceiver<FrameEnvelope>,
    data_rx: mpsc::Receiver<FrameEnvelope>,
    /// Frames retained after an interruption drain (uninterruptible frames).
    retained: Vec<FrameEnvelope>,
    metrics: QueueMetrics,
    shared_metrics: Arc<SharedMetrics>,
}

impl BoundedFrameQueue {
    /// Create a new queue pair with the given configuration and name.
    ///
    /// Returns `(sender, consumer)`. The sender can be cloned and given to
    /// producers; the consumer is used by the node's run loop.
    pub fn new(config: QueueConfig, name: &str) -> (FrameQueueSender, Self) {
        let (system_tx, system_rx) = mpsc::unbounded_channel();
        let (data_tx, data_rx) = mpsc::channel(config.max_size.max(1));

        let shared_metrics = Arc::new(SharedMetrics {
            total_dropped: AtomicU64::new(0),
        });

        let sender = FrameQueueSender {
            name: Arc::from(name),
            system_tx,
            data_tx,
            config: config.clone(),
            shared_metrics: Arc::clone(&shared_metrics),
        };

        let consumer = Self {
            name: name.to_string(),
            system_rx,
            data_rx,
            retained: Vec::new(),
            metrics: QueueMetrics::default(),
            shared_metrics,
        };

        (sender, consumer)
    }

    /// Dequeue the next frame, blocking if the queue is empty.
    ///
    /// Priority order: retained (uninterruptible survivors) > system > data.
    ///
    /// When one channel is closed (all senders dropped), the method falls back
    /// to polling only the remaining open channel. When both are closed and no
    /// retained frames remain, returns `None`.
    pub async fn get(&mut self) -> Option<FrameEnvelope> {
        // Priority 0: retained frames from interruption drain.
        if !self.retained.is_empty() {
            self.metrics.total_enqueued += 1;
            return Some(self.retained.remove(0));
        }

        // Fast path: check system channel first, then data channel.
        if let Ok(envelope) = self.system_rx.try_recv() {
            self.metrics.total_enqueued += 1;
            return Some(envelope);
        }
        if let Ok(envelope) = self.data_rx.try_recv() {
            self.metrics.total_enqueued += 1;
            return Some(envelope);
        }

        // All empty — async wait with system priority.
        // Track closed channels so we don't spin on a closed system_rx.
        let mut system_open = true;
        let mut data_open = true;

        loop {
            if !system_open && !data_open {
                return None;
            }

            // Both open: use biased select for system priority.
            if system_open && data_open {
                tokio::select! {
                    biased;
                    result = self.system_rx.recv() => {
                        match result {
                            Some(envelope) => {
                                self.metrics.total_enqueued += 1;
                                return Some(envelope);
                            }
                            None => {
                                system_open = false;
                                // Don't loop — immediately try data_rx.
                            }
                        }
                    }
                    result = self.data_rx.recv() => {
                        match result {
                            Some(envelope) => {
                                self.metrics.total_enqueued += 1;
                                return Some(envelope);
                            }
                            None => {
                                data_open = false;
                            }
                        }
                    }
                }
            } else if system_open {
                match self.system_rx.recv().await {
                    Some(envelope) => {
                        self.metrics.total_enqueued += 1;
                        return Some(envelope);
                    }
                    None => return None,
                }
            } else {
                // data_open only
                match self.data_rx.recv().await {
                    Some(envelope) => {
                        self.metrics.total_enqueued += 1;
                        return Some(envelope);
                    }
                    None => return None,
                }
            }
        }
    }

    /// Try to dequeue a frame without blocking. Returns `None` if both channels
    /// are empty.
    pub fn get_nowait(&mut self) -> Option<FrameEnvelope> {
        if !self.retained.is_empty() {
            self.metrics.total_enqueued += 1;
            return Some(self.retained.remove(0));
        }
        if let Ok(envelope) = self.system_rx.try_recv() {
            self.metrics.total_enqueued += 1;
            return Some(envelope);
        }
        if let Ok(envelope) = self.data_rx.try_recv() {
            self.metrics.total_enqueued += 1;
            return Some(envelope);
        }
        None
    }

    /// Returns `true` if both the system channel and data channel are empty.
    pub fn is_empty(&self) -> bool {
        self.system_rx.is_empty() && self.data_rx.is_empty()
    }

    /// Combined length of system channel and data channel.
    pub fn len(&self) -> usize {
        self.system_rx.len() + self.data_rx.len()
    }

    /// Drain data/control frames, keeping only those that are uninterruptible
    /// (e.g. `End`, `Stop`, `Cancel`). System frames are left untouched.
    ///
    /// Uninterruptible frames are moved to an internal retained buffer and will
    /// be returned by subsequent `get()` / `get_nowait()` calls with highest
    /// priority.
    pub fn drain_keeping_uninterruptible(&mut self) {
        while let Ok(env) = self.data_rx.try_recv() {
            if env.frame.is_uninterruptible() {
                self.retained.push(env);
            }
        }
    }

    /// Read-only access to queue metrics.
    pub fn metrics(&self) -> QueueMetrics {
        QueueMetrics {
            total_enqueued: self.metrics.total_enqueued,
            total_dropped: self.shared_metrics.total_dropped.load(Ordering::Relaxed),
            high_water_mark: self.metrics.high_water_mark,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_core::{Frame, FrameDirection, FrameHeader};

    fn system_frame() -> FrameEnvelope {
        FrameEnvelope {
            frame: Frame::Start(FrameHeader::new()),
            direction: FrameDirection::Downstream,
        }
    }

    fn data_frame(text: &str) -> FrameEnvelope {
        FrameEnvelope {
            frame: Frame::Text {
                header: FrameHeader::new(),
                data: pipecat_core::TextData {
                    text: text.to_string(),
                },
            },
            direction: FrameDirection::Downstream,
        }
    }

    fn end_frame() -> FrameEnvelope {
        FrameEnvelope {
            frame: Frame::End(FrameHeader::new()),
            direction: FrameDirection::Downstream,
        }
    }

    #[tokio::test]
    async fn system_frames_have_priority() {
        let config = QueueConfig {
            max_size: 10,
            ..Default::default()
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "test");

        sender.send(data_frame("first")).await.unwrap();
        sender.send(system_frame()).await.unwrap();
        sender.send(data_frame("second")).await.unwrap();

        // System frame comes out first despite being enqueued second.
        let env = q.get().await.unwrap();
        assert_eq!(env.frame.name(), "Start");

        let env = q.get().await.unwrap();
        assert_eq!(env.frame.name(), "Text");
    }

    #[tokio::test]
    async fn drop_newest_policy() {
        let config = QueueConfig {
            max_size: 2,
            drop_policy: DropPolicy::DropNewest,
            warn_threshold: 0.8,
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "test");

        sender.send(data_frame("a")).await.unwrap();
        sender.send(data_frame("b")).await.unwrap();
        sender.send(data_frame("c")).await.unwrap(); // dropped

        let metrics = q.metrics();
        assert_eq!(metrics.total_dropped, 1);

        let env = q.get().await.unwrap();
        if let Frame::Text { data, .. } = &env.frame {
            assert_eq!(data.text, "a");
        } else {
            panic!("expected Text frame");
        }
    }

    #[tokio::test]
    async fn drop_oldest_policy_drops_when_full() {
        let config = QueueConfig {
            max_size: 2,
            drop_policy: DropPolicy::DropOldest,
            warn_threshold: 0.8,
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "test");

        sender.send(data_frame("a")).await.unwrap();
        sender.send(data_frame("b")).await.unwrap();
        // Queue is full, this should drop (bounded channel rejects).
        sender.send(data_frame("c")).await.unwrap();

        let metrics = q.metrics();
        assert_eq!(metrics.total_dropped, 1);

        let env = q.get().await.unwrap();
        if let Frame::Text { data, .. } = &env.frame {
            assert_eq!(data.text, "a");
        } else {
            panic!("expected Text frame");
        }
    }

    #[tokio::test]
    async fn block_policy_does_not_deadlock() {
        let config = QueueConfig {
            max_size: 2,
            drop_policy: DropPolicy::Block,
            warn_threshold: 0.8,
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "test");

        // Fill the queue.
        sender.send(data_frame("a")).await.unwrap();
        sender.send(data_frame("b")).await.unwrap();

        // Send from a separate task (will block until space available).
        let sender_clone = sender.clone();
        let send_handle = tokio::spawn(async move {
            sender_clone.send(data_frame("c")).await.unwrap();
        });

        // Drain one frame to make space.
        let env = q.get().await.unwrap();
        assert_eq!(env.frame.name(), "Text");

        // The blocked send should now complete.
        tokio::time::timeout(std::time::Duration::from_secs(1), send_handle)
            .await
            .expect("send should complete within 1s")
            .expect("send task should succeed");
    }

    #[tokio::test]
    async fn drain_keeps_uninterruptible() {
        let config = QueueConfig {
            max_size: 10,
            ..Default::default()
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "test");

        sender.send(data_frame("a")).await.unwrap();
        sender.send(end_frame()).await.unwrap();
        sender.send(data_frame("b")).await.unwrap();

        // Small delay to let messages arrive.
        tokio::task::yield_now().await;

        q.drain_keeping_uninterruptible();

        let env = q.get().await.unwrap();
        assert_eq!(env.frame.name(), "End");
    }

    #[tokio::test]
    async fn get_nowait_returns_none_when_empty() {
        let config = QueueConfig::default();
        let (_sender, mut q) = BoundedFrameQueue::new(config, "test");
        assert!(q.get_nowait().is_none());
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[tokio::test]
    async fn metrics_tracking() {
        let config = QueueConfig {
            max_size: 10,
            ..Default::default()
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "test");

        sender.send(data_frame("a")).await.unwrap();
        sender.send(data_frame("b")).await.unwrap();
        sender.send(system_frame()).await.unwrap();

        // Drain all to update consumer metrics.
        q.get().await.unwrap();
        q.get().await.unwrap();
        q.get().await.unwrap();

        assert_eq!(q.metrics().total_enqueued, 3);
        assert_eq!(q.metrics().total_dropped, 0);
    }

    #[tokio::test]
    async fn backpressure_drops_under_load() {
        // Verify that sending 200+ data frames through a queue with max_size=10
        // actually drops frames.
        let config = QueueConfig {
            max_size: 10,
            drop_policy: DropPolicy::DropNewest,
            warn_threshold: 0.8,
        };
        let (sender, mut q) = BoundedFrameQueue::new(config, "load_test");

        let total_sent = 200;
        for i in 0..total_sent {
            let _ = sender.send(data_frame(&format!("msg{i}"))).await;
        }

        // Drain whatever is left.
        let mut received = 0;
        while q.get_nowait().is_some() {
            received += 1;
        }

        let metrics = q.metrics();
        assert!(received <= 10, "should have at most max_size frames");
        assert!(metrics.total_dropped > 0, "drops should have occurred");
        assert_eq!(
            received as u64 + metrics.total_dropped,
            total_sent,
            "received + dropped should equal sent"
        );
    }
}
