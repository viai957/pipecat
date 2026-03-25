//! Async bridge between tokio and asyncio.
//!
//! Handles driving Python coroutines from Rust's tokio runtime and routing
//! frames back through Rust channels when Python calls `push_frame()`.
