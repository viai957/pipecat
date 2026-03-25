# Changelog

All notable changes to the pipecat-rs Rust engine will be documented in this file.

## [0.1.0] - 2026-03-17

### Added

**Core Engine**
- Frame enum with 96 variants covering all Pipecat frame types
- 140 frame type constants matching Python FrameType registry
- FrameProcessor trait with sync fast-path (`process_frame_sync`)
- PipelineEndReason enum for lifecycle coordination
- Lazy-allocated frame headers (hot fields inline, cold fields boxed)

**Pipeline**
- Linear pipeline with direct inter-processor wiring (N tasks, not 3N)
- PipelineTask with StartFrame injection, heartbeat, sink monitoring
- Upstream task frame handling (EndTask/CancelTask/StopTask conversion)
- Backpressure system with dual-channel design (system bypass + bounded data)
- Batch drain optimization (up to 15 frames per select! iteration)
- Observer trait for frame flow monitoring

**Python Bridge (pipecat-python)**
- PyO3 bindings for NativePipelineTask with queue_frame/cancel
- Bidirectional frame conversion preserving frame ID and metadata
- Recursive coroutine driver for Python async processor callbacks
- RustPushBridge injection for processor-to-processor routing
- Observer bridge with proper async coroutine awaiting
- InterruptionFrame asyncio.Event sidecar storage
- Sink completion callback for Python lifecycle coordination

**Audio (pipecat-audio)**
- AudioBufferPool: pre-allocated O(1) acquire/release buffer pool
- AudioRingBuffer: lock-free SPSC ring buffer via rtrb
- G.711 codec (u-law/A-law encode/decode)
- Rubato-based resampling (SIMD-accelerated)
- Volume calculation, silence detection, audio mixing

**Services (pipecat-service-traits)**
- STTService, TTSService, LLMService traits
- RateLimited and Traced composable decorators
- Setting<T> delta-update system for runtime config changes

**Infrastructure**
- CI pipeline: clippy gate, test gate, Criterion benchmarks
- 5-platform wheel builds (Linux x86/arm, macOS x86/arm, Windows)
- Python 3.10-3.13 support
- PyPI release on native-v* tags

### Performance
- Frame creation: ~10ns (Rust enum) vs ~500ns (Python dataclass)
- Channel send: ~50ns (mpsc) vs ~5-20us (asyncio queue)
- Frame routing verified end-to-end with Python processors
- 353 Rust tests, 28 native integration tests, 690 Python tests pass
