use std::time::Instant;

/// A monotonic pipeline clock for generating presentation timestamps (PTS).
///
/// The clock starts at zero when created and returns elapsed microseconds
/// from that point forward. It uses `std::time::Instant` internally, so
/// it is monotonic and not affected by wall-clock adjustments.
#[derive(Debug, Clone)]
pub struct PipelineClock {
    epoch: Instant,
}

impl PipelineClock {
    /// Create a new clock. The epoch is set to "now".
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }

    /// Returns microseconds elapsed since this clock was created.
    pub fn now_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    /// Returns milliseconds elapsed since this clock was created.
    pub fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// Generate a PTS value (microseconds since epoch).
    pub fn pts(&self) -> u64 {
        self.now_us()
    }
}

impl Default for PipelineClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn clock_starts_near_zero() {
        let clock = PipelineClock::new();
        let t = clock.now_us();
        // Should be very small (< 1ms = 1000us)
        assert!(t < 1_000, "clock should start near zero, got {t}us");
    }

    #[test]
    fn clock_is_monotonic() {
        let clock = PipelineClock::new();
        let a = clock.now_us();
        thread::sleep(Duration::from_millis(1));
        let b = clock.now_us();
        assert!(b > a, "clock must be monotonic");
    }

    #[test]
    fn pts_returns_microseconds() {
        let clock = PipelineClock::new();
        thread::sleep(Duration::from_millis(10));
        let pts = clock.pts();
        // Should be at least 10ms = 10_000us (allow some slack)
        assert!(pts >= 5_000, "expected >= 5000us, got {pts}us");
    }

    #[test]
    fn now_ms_works() {
        let clock = PipelineClock::new();
        thread::sleep(Duration::from_millis(10));
        let ms = clock.now_ms();
        assert!(ms >= 5, "expected >= 5ms, got {ms}ms");
    }

    #[test]
    fn default_works() {
        let clock = PipelineClock::default();
        let _ = clock.now_us();
    }
}
