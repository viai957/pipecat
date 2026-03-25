use bytes::Bytes;
use criterion::{criterion_group, criterion_main, Criterion};

use pipecat_core::clock::PipelineClock;
use pipecat_core::{AudioData, Frame, FrameDirection, FrameHeader, TextData};

use pipecat_pipeline::{
    BoundedFrameQueue, FrameEnvelope, FrameProcessor, PassthroughProcessor, Pipeline, QueueConfig,
};

fn frame_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_creation");

    group.bench_function("start_frame", |b| {
        b.iter(|| Frame::Start(FrameHeader::new()))
    });

    group.bench_function("text_frame", |b| {
        b.iter(|| Frame::Text {
            header: FrameHeader::new(),
            data: TextData {
                text: "hello".into(),
            },
        })
    });

    group.bench_function("audio_frame", |b| {
        let audio = Bytes::from(vec![0u8; 640]);
        b.iter(|| Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: audio.clone(), // Bytes clone is cheap (refcount)
                sample_rate: 16000,
                num_channels: 1,
            },
        })
    });

    group.finish();
}

fn queue_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("queue_operations");

    group.bench_function("put_get_data_frame", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (sender, mut q) = BoundedFrameQueue::new(QueueConfig::default(), "bench");
                for _ in 0..100 {
                    sender
                        .send(FrameEnvelope {
                            frame: Frame::Text {
                                header: FrameHeader::new(),
                                data: TextData {
                                    text: "bench".into(),
                                },
                            },
                            direction: FrameDirection::Downstream,
                        })
                        .await
                        .unwrap();
                }
                for _ in 0..100 {
                    q.get_nowait().unwrap();
                }
            });
        });
    });

    group.bench_function("put_get_system_frame", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (sender, mut q) = BoundedFrameQueue::new(QueueConfig::default(), "bench");
                for _ in 0..100 {
                    sender
                        .send(FrameEnvelope {
                            frame: Frame::Start(FrameHeader::new()),
                            direction: FrameDirection::Downstream,
                        })
                        .await
                        .unwrap();
                }
                for _ in 0..100 {
                    q.get_nowait().unwrap();
                }
            });
        });
    });

    group.finish();
}

fn pipeline_passthrough(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("pipeline_passthrough");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    for n_processors in [1, 4, 8] {
        group.bench_function(format!("{n_processors}_processors"), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let processors: Vec<Box<dyn FrameProcessor>> = (0..n_processors)
                        .map(|i| {
                            Box::new(PassthroughProcessor::new(&format!("p{i}")))
                                as Box<dyn FrameProcessor>
                        })
                        .collect();

                    let pipeline = Pipeline::new(processors);
                    let clock = PipelineClock::new();
                    let mut handle = pipeline.start(clock);

                    // Send Start + 100 frames + End
                    handle
                        .source_tx
                        .send(FrameEnvelope {
                            frame: Frame::Start(FrameHeader::new()),
                            direction: FrameDirection::Downstream,
                        })
                        .await
                        .unwrap();

                    for i in 0..100 {
                        handle
                            .source_tx
                            .send(FrameEnvelope {
                                frame: Frame::Text {
                                    header: FrameHeader::new(),
                                    data: TextData {
                                        text: format!("msg{i}"),
                                    },
                                },
                                direction: FrameDirection::Downstream,
                            })
                            .await
                            .unwrap();
                    }

                    handle
                        .source_tx
                        .send(FrameEnvelope {
                            frame: Frame::End(FrameHeader::new()),
                            direction: FrameDirection::Downstream,
                        })
                        .await
                        .unwrap();

                    // Wait for End to arrive at sink
                    let mut received = 0;
                    loop {
                        let Some(env) = handle.sink_rx.get().await else {
                            break;
                        };
                        received += 1;
                        if matches!(env.frame, Frame::End(_)) {
                            break;
                        }
                    }

                    handle.cancellation.cancel();
                    drop(handle.source_tx);
                    drop(handle.upstream_tx);
                    for jh in handle.join_handles {
                        let _ = jh.await;
                    }

                    received
                });
            });
        });
    }

    group.finish();
}

fn audio_frame_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_frame_clone");

    // 20ms of 16kHz mono audio = 640 bytes
    let audio_data = AudioData {
        audio: Bytes::from(vec![0u8; 640]),
        sample_rate: 16000,
        num_channels: 1,
    };
    let frame = Frame::AudioRawInput {
        header: FrameHeader::new(),
        audio: audio_data,
    };

    group.bench_function("clone_audio_frame", |b| b.iter(|| frame.clone()));

    group.finish();
}

criterion_group!(
    benches,
    frame_creation,
    queue_operations,
    pipeline_passthrough,
    audio_frame_clone
);
criterion_main!(benches);
