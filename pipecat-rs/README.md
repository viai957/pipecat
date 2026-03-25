# Pipecat Rust Engine (pipecat-rs)

Rust implementation of Pipecat's frame processing pipeline. Provides
high-performance frame routing, audio processing, and service integration for
the [pipecat-ai](https://github.com/pipecat-ai/pipecat) Python framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Frontend Layer                         │
│  (Configuration, Service credentials, User-defined processors)  │
│                                                                 │
│  Pipeline([stt, llm, tts])  →  PyO3 FFI  →  Rust Engine        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                 Rust Engine (pipecat-core)                       │
│                                                                 │
│  ┌──────────┐    mpsc     ┌──────────┐    mpsc    ┌──────────┐ │
│  │Processor │───channel──▶│Processor │───channel─▶│Processor │ │
│  │  Node 1  │             │  Node 2  │            │  Node 3  │ │
│  └──────────┘             └──────────┘            └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

Frames flow between processor nodes over bounded `mpsc` channels. The Python
layer configures the pipeline graph and provides service credentials; the Rust
engine owns the hot loop, frame dispatch, and audio DSP.

## Crate Structure

| Crate | Path | Description |
|-------|------|-------------|
| **pipecat-core** | `crates/pipecat-core` | Frame enum (96 variants), type IDs, `FrameProcessor` trait, errors |
| **pipecat-pipeline** | `crates/pipecat-pipeline` | Pipeline engine, `PipelineTask`, backpressure, observer |
| **pipecat-audio** | `crates/pipecat-audio` | Audio DSP -- buffer pool, ring buffer, resampling, codecs, VAD |
| **pipecat-python** | `crates/pipecat-python` | PyO3 bindings -- `NativePipelineTask`, frame conversion, observer bridge |
| **pipecat-transport-traits** | `crates/pipecat-transport-traits` | Abstract transport interfaces |
| **pipecat-transport-local** | `crates/pipecat-transport-local` | Local audio I/O via cpal |
| **pipecat-service-traits** | `crates/pipecat-service-traits` | STT/TTS/LLM provider traits, decorators |
| **pipecat-service-deepgram** | `crates/pipecat-service-deepgram` | Deepgram STT via WebSocket |
| **pipecat-vad** | `crates/pipecat-vad` | Voice Activity Detection |
| **pipecat-metrics** | `crates/pipecat-metrics` | Pipeline metrics collection |
| **pipecat-bench** | `crates/pipecat-bench` | Criterion benchmarks |

Dependency graph (simplified):

```
pipecat-python
  └── pipecat-pipeline
        ├── pipecat-core
        ├── pipecat-audio
        ├── pipecat-metrics
        └── pipecat-vad
  └── pipecat-service-deepgram
        └── pipecat-service-traits
  └── pipecat-transport-local
        └── pipecat-transport-traits
```

## Building

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace --exclude pipecat-python

# Run clippy
cargo clippy --workspace --exclude pipecat-python -- -D warnings

# Build Python wheel
cd crates/pipecat-python && maturin develop --release

# Run benchmarks
cargo bench -p pipecat-bench
```

The `pipecat-python` crate is excluded from normal test/clippy runs because it
requires a Python interpreter and PyO3 build environment. Use `maturin` to
build and test it.

## Development

- **Rust edition**: 2021
- **MSRV**: stable
- **Release profile**: thin LTO, single codegen unit, `opt-level = 3`
- **Tests**: 353 tests across all crates
- **CI**: GitHub Actions with clippy gate, test gate, 5-platform wheel builds

### Key dependencies

| Dependency | Purpose |
|------------|---------|
| `tokio` | Async runtime |
| `bytes` | Zero-copy buffer management |
| `rubato` / `dasp` | Audio resampling and DSP |
| `cpal` | Cross-platform audio I/O |
| `rtrb` | Real-time safe ring buffer |
| `ort` | ONNX Runtime for ML inference (VAD) |
| `pyo3` | Python ↔ Rust FFI |
| `criterion` | Benchmarking |
| `tokio-tungstenite` | WebSocket client (service integrations) |
| `thiserror` | Typed error enums |

### Conventions

- All public types implement `Debug`.
- Errors use `thiserror` enums, not `anyhow` / boxed errors.
- Async code uses `tokio`; blocking I/O is offloaded with `spawn_blocking`.
- Frame types are `#[non_exhaustive]` to allow additions without breaking
  changes.
- Benchmarks live in `pipecat-bench` and use Criterion with HTML reports.

### Running a single crate's tests

```bash
cargo test -p pipecat-core
cargo test -p pipecat-audio
cargo test -p pipecat-pipeline
```

### Profiling

The bench profile inherits from release but enables debug symbols, so
`perf`/`Instruments` traces resolve correctly:

```bash
cargo bench -p pipecat-bench -- --profile-time=10
```

## Integration with Python

The `pipecat-python` crate exposes a `pipecat._native` extension module via
PyO3. The Python side imports it as:

```python
from pipecat._native import NativePipelineTask
```

Key bridge types:

- `NativePipelineTask` -- wraps the Rust `PipelineTask` and drives it from
  Python's asyncio event loop.
- Frame conversion functions translate between Python `Frame` dataclasses and
  Rust `Frame` enum variants.
- Observer bridge forwards Rust-side frame events to Python observer callbacks.

Build the wheel for local development:

```bash
cd crates/pipecat-python
maturin develop --release
```

## License

BSD-2-Clause. See [LICENSE](../LICENSE) in the repository root.
