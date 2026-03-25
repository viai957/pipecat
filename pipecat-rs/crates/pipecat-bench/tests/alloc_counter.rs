//! Hot-path allocation counter test.
//!
//! Uses a custom global allocator to count heap allocations during audio frame
//! passthrough. This serves as a regression guard — if someone introduces a
//! new allocation in the hot path, this test fails.
//!
//! Run with: cargo test -p pipecat-bench --test alloc_counter -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;

use pipecat_core::clock::PipelineClock;
use pipecat_core::{AudioData, Frame, FrameDirection, FrameHeader};
use pipecat_pipeline::{FrameEnvelope, FrameProcessor, PassthroughProcessor, Pipeline};

// ─── Counting Allocator ─────────────────────────────────────────────────────

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static COUNTING_ENABLED: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING_ENABLED.load(Ordering::Relaxed) != 0 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING_ENABLED.load(Ordering::Relaxed) != 0 {
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_counts() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn start_counting() {
    reset_counts();
    COUNTING_ENABLED.store(1, Ordering::SeqCst);
}

fn stop_counting() -> (u64, u64) {
    COUNTING_ENABLED.store(0, Ordering::SeqCst);
    (
        ALLOC_COUNT.load(Ordering::SeqCst),
        DEALLOC_COUNT.load(Ordering::SeqCst),
    )
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn audio_envelope(buf: &Bytes) -> FrameEnvelope {
    FrameEnvelope {
        frame: Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: buf.clone(),
                sample_rate: 16000,
                num_channels: 1,
            },
        },
        direction: FrameDirection::Downstream,
    }
}

fn start_envelope() -> FrameEnvelope {
    FrameEnvelope {
        frame: Frame::Start(FrameHeader::new()),
        direction: FrameDirection::Downstream,
    }
}

fn end_envelope() -> FrameEnvelope {
    FrameEnvelope {
        frame: Frame::End(FrameHeader::new()),
        direction: FrameDirection::Downstream,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

/// Verify that audio frame passthrough through an 8-processor pipeline
/// allocates at most a small fixed number of times per frame.
///
/// With the sync fast path (L5), passthrough processors use `try_push_frame`
/// which avoids the boxed future from `#[async_trait]`. Combined with the
/// FrameHeader hot/cold split (L3A) and variant boxing (L3B), the hot path
/// should be nearly allocation-free.
#[tokio::test]
async fn hot_path_allocation_count() {
    let n_frames = 500u64;
    let n_processors = 8usize;

    let processors: Vec<Box<dyn FrameProcessor>> = (0..n_processors)
        .map(|i| Box::new(PassthroughProcessor::new(&format!("p{i}"))) as Box<dyn FrameProcessor>)
        .collect();

    let pipeline = Pipeline::new(processors);
    let mut handle = pipeline.start(PipelineClock::new());

    // Pre-allocate the audio buffer outside the counting window.
    let audio_buf = Bytes::from(vec![0u8; 640]); // 20ms @ 16kHz mono

    // Warm up: send Start and let it propagate.
    handle.source_tx.send(start_envelope()).await.unwrap();
    handle.sink_rx.get().await.unwrap();

    // ── Start counting ──────────────────────────────────────────────────
    start_counting();

    for _ in 0..n_frames {
        handle
            .source_tx
            .send(audio_envelope(&audio_buf))
            .await
            .unwrap();
        handle.sink_rx.get().await.unwrap();
    }

    let (allocs, _deallocs) = stop_counting();
    // ── Stop counting ───────────────────────────────────────────────────

    // Teardown
    handle.source_tx.send(end_envelope()).await.unwrap();
    let _ = handle.sink_rx.get().await.unwrap();
    handle.cancellation.cancel();
    drop(handle.source_tx);
    drop(handle.upstream_tx);
    for jh in handle.join_handles {
        let _ = jh.await;
    }

    let allocs_per_frame = allocs as f64 / n_frames as f64;

    println!("\n============================================================");
    println!("  HOT-PATH ALLOCATION COUNTER ({n_processors}-proc, {n_frames} audio frames)");
    println!("============================================================");
    println!("  Total allocations:  {allocs}");
    println!("  Allocs per frame:   {allocs_per_frame:.2}");
    println!("============================================================\n");

    // With the sync fast path, we expect very few allocations per frame.
    // The main remaining sources are:
    // - FrameId::next() is allocation-free (atomic counter)
    // - Bytes::clone() is allocation-free (refcount bump)
    // - tokio mpsc channel internals (slot allocation)
    //
    // Allow up to 4 allocs per frame as headroom. If this fails, someone
    // introduced an allocation in the hot path — investigate before raising
    // the limit.
    assert!(
        allocs_per_frame <= 4.0,
        "Hot-path regression: {allocs_per_frame:.2} allocs/frame exceeds 4.0 limit. \
         Total: {allocs} allocs over {n_frames} frames through {n_processors} processors."
    );
}
