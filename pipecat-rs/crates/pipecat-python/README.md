# pipecat-ai-native

High-performance Rust engine for [pipecat-ai](https://github.com/pipecat-ai/pipecat) voice pipelines.

## What it does

`pipecat-ai-native` provides a Rust-based frame routing engine that accelerates Pipecat's pipeline hot path. When installed alongside `pipecat-ai`, it automatically activates to route frames through Rust's mpsc channels instead of Python's asyncio queues.

Your Python processors work unchanged -- Rust handles the frame routing between them.

## Installation

```bash
pip install pipecat-ai-native
```

Or install together with pipecat-ai:

```bash
pip install pipecat-ai pipecat-ai-native
```

## How it works

- **Auto-detected** -- When imported, `pipecat-ai` detects the native engine and activates it. Zero configuration required.
- **Drop-in acceleration** -- Existing pipelines get faster frame routing with no code changes.
- **Escape hatch** -- Set `PIPECAT_NATIVE=0` to disable the native engine and fall back to pure Python.

## Verification

```python
from pipecat._native_status import is_native_available

print(f"Native engine: {is_native_available()}")
```

## Requirements

- Python >= 3.10
- pipecat-ai >= 0.0.70

## Platform support

| OS      | Architectures          |
|---------|------------------------|
| Linux   | x86_64, aarch64        |
| macOS   | x86_64, Apple Silicon  |
| Windows | x86_64                 |

Pre-built wheels are published for all supported platforms.

## License

BSD-2-Clause

## Links

- [GitHub](https://github.com/pipecat-ai/pipecat)
- [Documentation](https://docs.pipecat.ai)
- [pipecat-ai on PyPI](https://pypi.org/project/pipecat-ai/)
