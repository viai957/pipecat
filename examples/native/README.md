# Rust Native Engine Examples

These examples demonstrate how to use Pipecat's Rust native engine for
high-performance frame routing.

## Prerequisites

Install the Rust native engine:

```bash
# From the repository root:
make dev-native

# Or manually:
cd pipecat-rs/crates/pipecat-python && pip install -e .
```

Verify installation:

```bash
python -c "from pipecat._native_status import is_native_available; print(f'Native: {is_native_available()}')"
```

## Examples

| File | Description |
|------|-------------|
| `01-verify-native-engine.py` | Verify the native engine is installed and check capabilities |
| `02-native-pipeline-passthrough.py` | Run a pipeline directly on the Rust engine |
| `03-native-vs-python-benchmark.py` | Benchmark comparing Rust vs Python frame routing |
| `04-existing-pipeline-with-native.py` | Standard Pipecat pipeline that auto-accelerates with native engine |

## Key Concepts

### Zero-config acceleration

Any existing Pipecat pipeline automatically benefits from the native engine
when installed. No code changes needed:

```python
# This pipeline runs faster when pipecat-ai-native is installed
pipeline = Pipeline([stt, llm, tts])
task = PipelineTask(pipeline, params)
await runner.run(task)
```

### Direct Rust pipeline (advanced)

For maximum performance, use `NativePipelineTask` directly to bypass Python's
pipeline wrapper:

```python
from pipecat._native import NativePipelineTask

task = NativePipelineTask(processors=[proc1, proc2])
await task.run_async()
```

### Escape hatch

Disable the native engine at any time:

```bash
PIPECAT_NATIVE=0 python my_bot.py
```
