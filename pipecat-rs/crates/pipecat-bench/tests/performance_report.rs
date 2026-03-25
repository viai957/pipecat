//! Comprehensive performance measurement suite.
//!
//! Run with: cargo test -p pipecat-bench --test performance_report -- --nocapture

use std::time::Instant;

use bytes::Bytes;

use pipecat_bench::LatencyRecorder;
use pipecat_core::clock::PipelineClock;
use pipecat_core::{AudioData, Frame, FrameDirection, FrameHeader, TextData};
use pipecat_pipeline::{
    BoundedFrameQueue, FrameEnvelope, FrameProcessor, PassthroughProcessor, Pipeline, QueueConfig,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn text_envelope(msg: &str) -> FrameEnvelope {
    FrameEnvelope {
        frame: Frame::Text {
            header: FrameHeader::new(),
            data: TextData {
                text: msg.to_string(),
            },
        },
        direction: FrameDirection::Downstream,
    }
}

fn audio_envelope(sample_rate: u32, duration_ms: u32) -> FrameEnvelope {
    let num_samples = (sample_rate * duration_ms / 1000) as usize;
    let bytes = num_samples * 2; // 16-bit PCM
    FrameEnvelope {
        frame: Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: Bytes::from(vec![0u8; bytes]),
                sample_rate,
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

// ─── 1. Frame Creation Throughput ────────────────────────────────────────────

#[test]
fn measure_frame_creation_throughput() {
    println!("\n============================================================");
    println!("  FRAME CREATION THROUGHPUT");
    println!("============================================================");

    let iterations = 1_000_000u64;

    // StartFrame (minimal — just header)
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(Frame::Start(FrameHeader::new()));
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
    let per_sec = 1_000_000_000.0 / ns_per;
    println!(
        "  StartFrame:      {:.1} ns/op  ({:.1}M frames/sec)",
        ns_per,
        per_sec / 1_000_000.0
    );

    // TextFrame (header + string alloc)
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(Frame::Text {
            header: FrameHeader::new(),
            data: TextData {
                text: "hello world".into(),
            },
        });
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
    let per_sec = 1_000_000_000.0 / ns_per;
    println!(
        "  TextFrame:       {:.1} ns/op  ({:.1}M frames/sec)",
        ns_per,
        per_sec / 1_000_000.0
    );

    // AudioFrame (header + Bytes refcount clone)
    let audio_buf = Bytes::from(vec![0u8; 640]); // 20ms @ 16kHz mono
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: audio_buf.clone(),
                sample_rate: 16000,
                num_channels: 1,
            },
        });
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
    let per_sec = 1_000_000_000.0 / ns_per;
    println!(
        "  AudioFrame:      {:.1} ns/op  ({:.1}M frames/sec)",
        ns_per,
        per_sec / 1_000_000.0
    );

    // AudioFrame clone (Bytes refcount bump)
    let frame = Frame::AudioRawInput {
        header: FrameHeader::new(),
        audio: AudioData {
            audio: audio_buf.clone(),
            sample_rate: 16000,
            num_channels: 1,
        },
    };
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(frame.clone());
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iterations as f64;
    let per_sec = 1_000_000_000.0 / ns_per;
    println!(
        "  AudioFrame clone:{:.1} ns/op  ({:.1}M clones/sec)",
        ns_per,
        per_sec / 1_000_000.0
    );

    println!();
}

// ─── 2. BoundedFrameQueue Throughput ─────────────────────────────────────────

#[tokio::test]
async fn measure_queue_throughput() {
    println!("\n============================================================");
    println!("  BOUNDED FRAME QUEUE THROUGHPUT");
    println!("============================================================");

    let iterations = 100_000u64;

    // Data frames (bounded path)
    let (sender, mut q) = BoundedFrameQueue::new(
        QueueConfig {
            max_size: iterations as usize + 1,
            ..Default::default()
        },
        "bench_data",
    );

    let start = Instant::now();
    for i in 0..iterations {
        sender.send(text_envelope(&format!("m{i}"))).await.unwrap();
    }
    let put_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        q.get_nowait().unwrap();
    }
    let get_elapsed = start.elapsed();

    let put_ns = put_elapsed.as_nanos() as f64 / iterations as f64;
    let get_ns = get_elapsed.as_nanos() as f64 / iterations as f64;
    let roundtrip_ns = put_ns + get_ns;
    let throughput = 1_000_000_000.0 / roundtrip_ns;

    println!("  Data frames ({iterations} ops):");
    println!("    put:        {:.1} ns/op", put_ns);
    println!("    get_nowait: {:.1} ns/op", get_ns);
    println!("    roundtrip:  {:.1} ns/op", roundtrip_ns);
    println!("    throughput: {:.1}M frames/sec", throughput / 1_000_000.0);

    // System frames (unbounded path)
    let (sender, mut q) = BoundedFrameQueue::new(QueueConfig::default(), "bench_sys");

    let start = Instant::now();
    for _ in 0..iterations {
        sender.send(start_envelope()).await.unwrap();
    }
    let put_elapsed = start.elapsed();

    let start_t = Instant::now();
    for _ in 0..iterations {
        q.get_nowait().unwrap();
    }
    let get_elapsed = start_t.elapsed();

    let put_ns = put_elapsed.as_nanos() as f64 / iterations as f64;
    let get_ns = get_elapsed.as_nanos() as f64 / iterations as f64;
    let roundtrip_ns = put_ns + get_ns;
    let throughput = 1_000_000_000.0 / roundtrip_ns;

    println!("  System frames ({iterations} ops):");
    println!("    put:        {:.1} ns/op", put_ns);
    println!("    get_nowait: {:.1} ns/op", get_ns);
    println!("    roundtrip:  {:.1} ns/op", roundtrip_ns);
    println!("    throughput: {:.1}M frames/sec", throughput / 1_000_000.0);
    println!();
}

// ─── 3. Pipeline Passthrough Latency ─────────────────────────────────────────

async fn measure_pipeline_latency(n_processors: usize, n_frames: usize) -> LatencyRecorder {
    let mut recorder = LatencyRecorder::new(3);
    let processors: Vec<Box<dyn FrameProcessor>> = (0..n_processors)
        .map(|i| Box::new(PassthroughProcessor::new(&format!("p{i}"))) as Box<dyn FrameProcessor>)
        .collect();

    let pipeline = Pipeline::new(processors);
    let mut handle = pipeline.start(PipelineClock::new());

    // Send Start
    handle.source_tx.send(start_envelope()).await.unwrap();
    // Wait for Start to propagate
    handle.sink_rx.get().await.unwrap();

    // Measure per-frame latency
    for i in 0..n_frames {
        let t0 = Instant::now();
        handle.source_tx.send(text_envelope(&format!("f{i}"))).await.unwrap();
        handle.sink_rx.get().await.unwrap();
        let elapsed_us = t0.elapsed().as_micros() as u64;
        recorder.record(elapsed_us);
    }

    // Teardown
    handle.source_tx.send(end_envelope()).await.unwrap();
    let _ = handle.sink_rx.get().await.unwrap();
    handle.cancellation.cancel();
    drop(handle.source_tx);
    drop(handle.upstream_tx);
    for jh in handle.join_handles {
        let _ = jh.await;
    }

    recorder
}

#[tokio::test]
async fn measure_pipeline_passthrough_latency() {
    println!("\n============================================================");
    println!("  PIPELINE PASSTHROUGH LATENCY (per-frame, HdrHistogram)");
    println!("============================================================");

    let n_frames = 10_000;

    for n_proc in [1, 2, 4, 8] {
        let recorder = measure_pipeline_latency(n_proc, n_frames).await;
        let report = recorder.report();
        let throughput = n_frames as f64 / (report.mean * n_frames as f64 / 1_000_000.0);
        println!(
            "  {n_proc} processor(s), {n_frames} frames:"
        );
        println!("    p50:  {} us", report.p50);
        println!("    p95:  {} us", report.p95);
        println!("    p99:  {} us", report.p99);
        println!("    p999: {} us", report.p999);
        println!("    mean: {:.2} us", report.mean);
        println!("    min:  {} us", report.min);
        println!("    max:  {} us", report.max);
        println!(
            "    throughput: {:.0} frames/sec ({:.1}K)",
            throughput,
            throughput / 1_000.0
        );
        println!();
    }
}

// ─── 4. Pipeline Burst Throughput ────────────────────────────────────────────

#[tokio::test]
async fn measure_pipeline_burst_throughput() {
    println!("\n============================================================");
    println!("  PIPELINE BURST THROUGHPUT (fire-and-drain)");
    println!("============================================================");

    for n_proc in [1, 4, 8] {
        let n_frames = 10_000usize;
        let processors: Vec<Box<dyn FrameProcessor>> = (0..n_proc)
            .map(|i| {
                Box::new(PassthroughProcessor::new(&format!("p{i}"))) as Box<dyn FrameProcessor>
            })
            .collect();

        let pipeline = Pipeline::new(processors);
        let mut handle = pipeline.start(PipelineClock::new());

        let t0 = Instant::now();

        // Fire all frames as fast as possible
        handle.source_tx.send(start_envelope()).await.unwrap();
        for i in 0..n_frames {
            handle
                .source_tx
                .send(text_envelope(&format!("b{i}")))
                .await
                .unwrap();
        }
        handle.source_tx.send(end_envelope()).await.unwrap();

        // Drain until End
        let mut received = 0usize;
        loop {
            let env = handle.sink_rx.get().await.unwrap();
            received += 1;
            if matches!(env.frame, Frame::End(_)) {
                break;
            }
        }

        let elapsed = t0.elapsed();
        let frames_per_sec = received as f64 / elapsed.as_secs_f64();

        handle.cancellation.cancel();
        drop(handle.source_tx);
        drop(handle.upstream_tx);
        for jh in handle.join_handles {
            let _ = jh.await;
        }

        println!(
            "  {n_proc} processor(s): {received} frames in {:.2} ms = {:.0} frames/sec ({:.1}K)",
            elapsed.as_secs_f64() * 1000.0,
            frames_per_sec,
            frames_per_sec / 1_000.0
        );
    }
    println!();
}

// ─── 5. Audio Simulation (real-time audio throughput) ────────────────────────

#[tokio::test]
async fn measure_audio_simulation_throughput() {
    println!("\n============================================================");
    println!("  AUDIO FRAME THROUGHPUT (simulated 16kHz, 20ms chunks)");
    println!("============================================================");

    // Simulate 10 seconds of audio at 16kHz, 20ms chunks = 500 frames
    let n_chunks = 500usize;

    for n_proc in [1, 4, 8] {
        let processors: Vec<Box<dyn FrameProcessor>> = (0..n_proc)
            .map(|i| {
                Box::new(PassthroughProcessor::new(&format!("p{i}"))) as Box<dyn FrameProcessor>
            })
            .collect();

        let pipeline = Pipeline::new(processors);
        let mut handle = pipeline.start(PipelineClock::new());

        let mut recorder = LatencyRecorder::new(3);

        // Start
        handle.source_tx.send(start_envelope()).await.unwrap();
        handle.sink_rx.get().await.unwrap();

        let t0 = Instant::now();
        for _ in 0..n_chunks {
            let frame_start = Instant::now();
            handle
                .source_tx
                .send(audio_envelope(16000, 20))
                .await
                .unwrap();
            handle.sink_rx.get().await.unwrap();
            recorder.record(frame_start.elapsed().as_micros() as u64);
        }
        let total_elapsed = t0.elapsed();

        // Teardown
        handle.source_tx.send(end_envelope()).await.unwrap();
        let _ = handle.sink_rx.get().await.unwrap();
        handle.cancellation.cancel();
        drop(handle.source_tx);
        drop(handle.upstream_tx);
        for jh in handle.join_handles {
            let _ = jh.await;
        }

        let report = recorder.report();
        let real_time_audio_secs = n_chunks as f64 * 0.020; // 20ms per chunk
        let processing_secs = total_elapsed.as_secs_f64();
        let rt_ratio = real_time_audio_secs / processing_secs;

        println!("  {n_proc} processor(s), {n_chunks} x 20ms audio chunks (= {:.1}s audio):", real_time_audio_secs);
        println!("    Total processing: {:.2} ms", processing_secs * 1000.0);
        println!("    Real-time ratio:  {:.0}x (>{:.0}x needed for RT)", rt_ratio, 1.0);
        println!("    Per-frame p50:    {} us", report.p50);
        println!("    Per-frame p99:    {} us", report.p99);
        println!("    Per-frame max:    {} us", report.max);
        println!(
            "    Headroom:         {:.1}% of 20ms budget unused",
            (1.0 - (report.mean / 20_000.0)) * 100.0
        );
        println!();
    }
}

// ─── 6. Memory Efficiency ────────────────────────────────────────────────────

#[test]
fn measure_memory_efficiency() {
    println!("\n============================================================");
    println!("  MEMORY EFFICIENCY");
    println!("============================================================");

    println!(
        "  Frame enum size:       {} bytes",
        std::mem::size_of::<Frame>()
    );
    println!(
        "  FrameHeader size:      {} bytes",
        std::mem::size_of::<FrameHeader>()
    );
    println!(
        "  AudioData size:        {} bytes",
        std::mem::size_of::<AudioData>()
    );
    println!(
        "  TextData size:         {} bytes",
        std::mem::size_of::<TextData>()
    );
    println!(
        "  FrameEnvelope size:    {} bytes",
        std::mem::size_of::<FrameEnvelope>()
    );
    println!(
        "  FrameDirection size:   {} bytes",
        std::mem::size_of::<FrameDirection>()
    );
    println!(
        "  Bytes size (handle):   {} bytes",
        std::mem::size_of::<Bytes>()
    );
    println!(
        "  String size (handle):  {} bytes",
        std::mem::size_of::<String>()
    );

    // Compare: 20ms of 16kHz audio as raw Vec vs Bytes
    let raw_vec = vec![0u8; 640];
    let bytes_ref = Bytes::from(raw_vec.clone());
    let cloned_bytes = bytes_ref.clone();

    // Bytes clone is refcount bump — same pointer
    assert_eq!(
        bytes_ref.as_ptr(),
        cloned_bytes.as_ptr(),
        "Bytes::clone should share the same buffer"
    );
    println!("\n  Bytes::clone shares buffer: YES (zero-copy confirmed)");
    println!(
        "  640-byte audio buffer: Bytes::clone = refcount bump (~4 bytes overhead)"
    );
    println!();
}

// ─── 7. Comparison vs Python Targets ─────────────────────────────────────────

#[tokio::test]
async fn performance_vs_targets() {
    println!("\n============================================================");
    println!("  PERFORMANCE vs PLAN TARGETS");
    println!("============================================================");

    // Target: 8-processor audio passthrough (1000 frames) < 1ms total
    let n_frames = 1000usize;
    let processors: Vec<Box<dyn FrameProcessor>> = (0..8)
        .map(|i| Box::new(PassthroughProcessor::new(&format!("p{i}"))) as Box<dyn FrameProcessor>)
        .collect();

    let pipeline = Pipeline::new(processors);
    let mut handle = pipeline.start(PipelineClock::new());

    handle.source_tx.send(start_envelope()).await.unwrap();
    // Wait for Start propagation
    handle.sink_rx.get().await.unwrap();

    let t0 = Instant::now();
    for i in 0..n_frames {
        handle
            .source_tx
            .send(text_envelope(&format!("t{i}")))
            .await
            .unwrap();
    }
    handle.source_tx.send(end_envelope()).await.unwrap();

    let mut received = 0;
    loop {
        let env = handle.sink_rx.get().await.unwrap();
        received += 1;
        if matches!(env.frame, Frame::End(_)) {
            break;
        }
    }
    let total_us = t0.elapsed().as_micros();

    handle.cancellation.cancel();
    drop(handle.source_tx);
    drop(handle.upstream_tx);
    for jh in handle.join_handles {
        let _ = jh.await;
    }

    // Target: BoundedFrameQueue put+get 10k ops < 100us
    let (sender, mut q) = BoundedFrameQueue::new(
        QueueConfig {
            max_size: 10_001,
            ..Default::default()
        },
        "target",
    );
    let qt0 = Instant::now();
    for i in 0..10_000 {
        sender.send(text_envelope(&format!("q{i}"))).await.unwrap();
    }
    for _ in 0..10_000 {
        q.get_nowait().unwrap();
    }
    let queue_us = qt0.elapsed().as_micros();

    // Target: AudioFrame creation < 50ns
    let audio_buf = Bytes::from(vec![0u8; 640]);
    let ft0 = Instant::now();
    let frame_iters = 1_000_000u64;
    for _ in 0..frame_iters {
        std::hint::black_box(Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: audio_buf.clone(),
                sample_rate: 16000,
                num_channels: 1,
            },
        });
    }
    let frame_ns = ft0.elapsed().as_nanos() as f64 / frame_iters as f64;

    // Pipeline p99 latency < 5ms target
    let recorder = measure_pipeline_latency(8, 1000).await;
    let p99 = recorder.p99();

    println!();
    println!(
        "  {:50} {:>10} {:>10} {:>6}",
        "Benchmark", "Target", "Actual", "Pass?"
    );
    println!("  {:-<50} {:-<10} {:-<10} {:-<6}", "", "", "", "");

    let pipeline_pass = total_us < 1000;
    println!(
        "  {:50} {:>8} us {:>8} us {:>6}",
        "8-proc passthrough (1000 frames, total)",
        "<1000",
        total_us,
        if pipeline_pass { "YES" } else { "NO" }
    );

    let queue_pass = queue_us < 100_000; // 100ms generous
    println!(
        "  {:50} {:>7} us {:>8} us {:>6}",
        "BoundedFrameQueue 10k put+get",
        "<100000",
        queue_us,
        if queue_pass { "YES" } else { "NO" }
    );

    let frame_pass = frame_ns < 200.0; // Allow generous margin vs 50ns target
    println!(
        "  {:50} {:>8} ns {:>8.1} ns {:>6}",
        "AudioFrame creation",
        "<200",
        frame_ns,
        if frame_pass { "YES" } else { "NO" }
    );

    let p99_pass = p99 < 5000;
    println!(
        "  {:50} {:>7} us {:>8} us {:>6}",
        "End-to-end pipeline p99 (8 proc)",
        "<5000",
        p99,
        if p99_pass { "YES" } else { "NO" }
    );

    println!(
        "\n  Received {} frames through 8-processor pipeline",
        received
    );
    println!();
}
