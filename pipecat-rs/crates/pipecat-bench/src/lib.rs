//! Benchmarking and latency-recording utilities for Pipecat pipelines.
//!
//! This crate provides [`LatencyRecorder`], a wrapper around an HdrHistogram
//! that records latency values in microseconds and produces percentile reports.

use std::fmt;

use hdrhistogram::Histogram;

/// Maximum recordable value: 60 seconds in microseconds.
const MAX_VALUE_US: u64 = 60_000_000;

/// A summary of latency percentiles and basic statistics.
#[derive(Debug, Clone)]
pub struct LatencyReport {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
    pub mean: f64,
    pub max: u64,
    pub min: u64,
    pub count: u64,
}

impl fmt::Display for LatencyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Latency Report ({} samples)", self.count)?;
        writeln!(f, "  p50  : {} us", self.p50)?;
        writeln!(f, "  p95  : {} us", self.p95)?;
        writeln!(f, "  p99  : {} us", self.p99)?;
        writeln!(f, "  p99.9: {} us", self.p999)?;
        writeln!(f, "  mean : {:.2} us", self.mean)?;
        writeln!(f, "  min  : {} us", self.min)?;
        write!(f, "  max  : {} us", self.max)
    }
}

/// Wraps an `hdrhistogram::Histogram<u64>` for recording latency values
/// in microseconds.
///
/// # Example
///
/// ```
/// use pipecat_bench::LatencyRecorder;
///
/// let mut recorder = LatencyRecorder::new(3);
/// recorder.record(100);
/// recorder.record(200);
/// recorder.record(300);
///
/// assert_eq!(recorder.count(), 3);
/// let report = recorder.report();
/// println!("{report}");
/// ```
pub struct LatencyRecorder {
    histogram: Histogram<u64>,
}

impl LatencyRecorder {
    /// Create a new recorder with the given number of significant value digits
    /// (typically 1-5, where 3 is a good default).
    ///
    /// The histogram can record values from 1 to 60,000,000 microseconds
    /// (60 seconds).
    pub fn new(significant_figures: u8) -> Self {
        let histogram = Histogram::new_with_max(MAX_VALUE_US, significant_figures)
            .expect("failed to create histogram");
        Self { histogram }
    }

    /// Record a latency value in microseconds.
    ///
    /// Values exceeding 60 seconds (60,000,000 us) are silently clamped to
    /// the maximum.
    pub fn record(&mut self, value_us: u64) {
        let clamped = value_us.min(MAX_VALUE_US);
        self.histogram
            .record(clamped)
            .expect("value within histogram range after clamping");
    }

    /// 50th percentile (median) in microseconds.
    pub fn p50(&self) -> u64 {
        self.histogram.value_at_percentile(50.0)
    }

    /// 95th percentile in microseconds.
    pub fn p95(&self) -> u64 {
        self.histogram.value_at_percentile(95.0)
    }

    /// 99th percentile in microseconds.
    pub fn p99(&self) -> u64 {
        self.histogram.value_at_percentile(99.0)
    }

    /// 99.9th percentile in microseconds.
    pub fn p999(&self) -> u64 {
        self.histogram.value_at_percentile(99.9)
    }

    /// Mean latency in microseconds.
    pub fn mean(&self) -> f64 {
        self.histogram.mean()
    }

    /// Maximum recorded value in microseconds.
    pub fn max(&self) -> u64 {
        self.histogram.max()
    }

    /// Minimum recorded value in microseconds.
    pub fn min(&self) -> u64 {
        self.histogram.min()
    }

    /// Total number of recorded values.
    pub fn count(&self) -> u64 {
        self.histogram.len()
    }

    /// Clear all recorded values from the histogram.
    pub fn reset(&mut self) {
        self.histogram.reset();
    }

    /// Produce a [`LatencyReport`] summarising all recorded latencies.
    pub fn report(&self) -> LatencyReport {
        LatencyReport {
            p50: self.p50(),
            p95: self.p95(),
            p99: self.p99(),
            p999: self.p999(),
            mean: self.mean(),
            max: self.max(),
            min: self.min(),
            count: self.count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_recording() {
        let mut rec = LatencyRecorder::new(3);
        rec.record(100);
        rec.record(200);
        rec.record(300);

        assert_eq!(rec.count(), 3);
        assert!(rec.min() <= 100);
        assert!(rec.max() >= 300);
        assert!(rec.mean() > 0.0);
    }

    #[test]
    fn percentiles_with_uniform_data() {
        let mut rec = LatencyRecorder::new(3);
        // Record values 1..=1000
        for i in 1..=1000 {
            rec.record(i);
        }

        assert_eq!(rec.count(), 1000);

        // p50 should be roughly 500
        let p50 = rec.p50();
        assert!(p50 >= 490 && p50 <= 510, "p50 = {p50}, expected ~500");

        // p95 should be roughly 950
        let p95 = rec.p95();
        assert!(p95 >= 940 && p95 <= 960, "p95 = {p95}, expected ~950");

        // p99 should be roughly 990
        let p99 = rec.p99();
        assert!(p99 >= 980 && p99 <= 1000, "p99 = {p99}, expected ~990");
    }

    #[test]
    fn min_max_single_value() {
        let mut rec = LatencyRecorder::new(3);
        rec.record(42);

        assert_eq!(rec.min(), 42);
        assert_eq!(rec.max(), 42);
        assert_eq!(rec.count(), 1);
    }

    #[test]
    fn reset_clears_histogram() {
        let mut rec = LatencyRecorder::new(3);
        rec.record(100);
        rec.record(200);
        assert_eq!(rec.count(), 2);

        rec.reset();
        assert_eq!(rec.count(), 0);
    }

    #[test]
    fn report_contains_all_fields() {
        let mut rec = LatencyRecorder::new(3);
        for i in 1..=100 {
            rec.record(i * 10);
        }

        let report = rec.report();
        assert_eq!(report.count, 100);
        assert!(report.min <= 10);
        assert!(report.max >= 1000);
        assert!(report.p50 > 0);
        assert!(report.p95 > report.p50);
        assert!(report.p99 >= report.p95);
        assert!(report.p999 >= report.p99);
        assert!(report.mean > 0.0);
    }

    #[test]
    fn report_display_is_nonempty() {
        let mut rec = LatencyRecorder::new(3);
        rec.record(500);

        let report = rec.report();
        let output = format!("{report}");
        assert!(output.contains("Latency Report"));
        assert!(output.contains("p50"));
        assert!(output.contains("p95"));
        assert!(output.contains("p99"));
        assert!(output.contains("mean"));
        assert!(output.contains("min"));
        assert!(output.contains("max"));
    }

    #[test]
    fn large_values_clamped() {
        let mut rec = LatencyRecorder::new(3);
        // Value exceeds MAX_VALUE_US, should be clamped to MAX_VALUE_US.
        // HdrHistogram may round up to the next representable bucket boundary,
        // so we allow a small margin (1% above MAX_VALUE_US).
        rec.record(100_000_000);
        assert_eq!(rec.count(), 1);
        let max = rec.max();
        assert!(
            max <= MAX_VALUE_US + MAX_VALUE_US / 100,
            "max = {max}, expected near {MAX_VALUE_US}"
        );
    }

    #[test]
    fn empty_recorder_stats() {
        let rec = LatencyRecorder::new(3);
        assert_eq!(rec.count(), 0);
        assert_eq!(rec.min(), 0);
        assert_eq!(rec.max(), 0);
        // mean of empty histogram is 0.0
        assert_eq!(rec.mean(), 0.0);
    }
}
