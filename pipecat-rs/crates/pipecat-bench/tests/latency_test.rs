use pipecat_bench::{LatencyRecorder, LatencyReport};

#[test]
fn uniform_distribution_percentiles() {
    let mut rec = LatencyRecorder::new(3);

    // Record values 1..=1000 (uniform distribution)
    for i in 1..=1000 {
        rec.record(i);
    }

    assert_eq!(rec.count(), 1000);

    // p50 should be around 500
    let p50 = rec.p50();
    assert!(
        (490..=510).contains(&p50),
        "p50 = {p50}, expected ~500"
    );

    // p95 should be around 950
    let p95 = rec.p95();
    assert!(
        (940..=960).contains(&p95),
        "p95 = {p95}, expected ~950"
    );

    // p99 should be around 990
    let p99 = rec.p99();
    assert!(
        (980..=1000).contains(&p99),
        "p99 = {p99}, expected ~990"
    );

    // p99.9 should be around 999-1000
    let p999 = rec.p999();
    assert!(
        (995..=1000).contains(&p999),
        "p999 = {p999}, expected ~999"
    );

    // mean should be around 500.5
    let mean = rec.mean();
    assert!(
        mean > 490.0 && mean < 510.0,
        "mean = {mean}, expected ~500.5"
    );

    // min and max
    assert_eq!(rec.min(), 1);
    assert!(rec.max() >= 1000);
}

#[test]
fn bimodal_distribution_percentiles() {
    let mut rec = LatencyRecorder::new(3);

    // 90% of values are low (100 us), 10% are high (10_000 us)
    for _ in 0..900 {
        rec.record(100);
    }
    for _ in 0..100 {
        rec.record(10_000);
    }

    assert_eq!(rec.count(), 1000);

    // p50 should be 100 (most values are 100)
    assert_eq!(rec.p50(), 100);

    // p95 should still be around 10_000 (the high tail)
    let p95 = rec.p95();
    assert!(
        p95 >= 9_900 && p95 <= 10_100,
        "p95 = {p95}, expected ~10000"
    );
}

#[test]
fn report_produces_valid_output() {
    let mut rec = LatencyRecorder::new(3);
    for i in 1..=500 {
        rec.record(i * 2);
    }

    let report = rec.report();

    // Verify structural properties
    assert_eq!(report.count, 500);
    assert!(report.min <= 2);
    assert!(report.max >= 1000);
    assert!(report.p50 <= report.p95);
    assert!(report.p95 <= report.p99);
    assert!(report.p99 <= report.p999);
    assert!(report.mean > 0.0);

    // Verify Display output
    let output = format!("{report}");
    assert!(output.contains("Latency Report (500 samples)"));
    assert!(output.contains("p50"));
    assert!(output.contains("p95"));
    assert!(output.contains("p99"));
    assert!(output.contains("p99.9"));
    assert!(output.contains("mean"));
    assert!(output.contains("min"));
    assert!(output.contains("max"));
}

#[test]
fn report_struct_is_clone() {
    let mut rec = LatencyRecorder::new(3);
    rec.record(42);

    let report = rec.report();
    let cloned: LatencyReport = report.clone();
    assert_eq!(cloned.count, report.count);
    assert_eq!(cloned.p50, report.p50);
    assert_eq!(cloned.max, report.max);
}

#[test]
fn reset_then_re_record() {
    let mut rec = LatencyRecorder::new(3);

    // First batch
    for i in 1..=100 {
        rec.record(i);
    }
    assert_eq!(rec.count(), 100);

    rec.reset();
    assert_eq!(rec.count(), 0);

    // Second batch with different values
    for i in 500..=600 {
        rec.record(i);
    }
    assert_eq!(rec.count(), 101);
    assert!(rec.min() >= 500);
}
