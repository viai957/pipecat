.PHONY: dev dev-native build-native test-native test-native-python clean-native

# Standard Python-only dev setup
dev:
	uv sync --group dev --all-extras --no-extra gstreamer --no-extra krisp

# Full dev setup including Rust native engine.
# Builds the Rust extension and symlinks it into the Python source tree
# so that `uv run` finds it without needing `pip install`.
dev-native: dev
	@echo "Building Rust native engine..."
	cd pipecat-rs/crates/pipecat-python && \
		$(shell uv run python -c "import sys; print(sys.executable)") -m pip install -e .
	@# Symlink the .so into src/pipecat/ so editable installs find it
	@rm -f src/pipecat/_native.cpython-*.so
	@ln -sf $$(ls pipecat-rs/crates/pipecat-python/python/pipecat/_native.cpython-*.so 2>/dev/null | head -1) \
		src/pipecat/ 2>/dev/null || true
	@echo "Native engine installed. Verify with:"
	@echo "  uv run python -c 'from pipecat._native_status import is_native_available; print(is_native_available())'"

# Build a release wheel for the Rust extension (for distribution)
build-native:
	cd pipecat-rs/crates/pipecat-python && maturin build --release --out dist

# Run the Rust test suite
test-native:
	cd pipecat-rs && cargo test --workspace --exclude pipecat-python
	cd pipecat-rs && cargo clippy --workspace --exclude pipecat-python -- -D warnings

# Run Python integration tests for the native engine
test-native-python:
	uv run pytest tests/test_native_integration.py -v

# Remove native engine build artifacts
clean-native:
	rm -f src/pipecat/_native.cpython-*.so
	cd pipecat-rs && cargo clean
