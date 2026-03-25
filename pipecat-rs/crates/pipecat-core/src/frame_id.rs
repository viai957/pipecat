use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique frame IDs.
static NEXT_FRAME_ID: AtomicU64 = AtomicU64::new(1);

/// A unique identifier for a frame instance.
///
/// Frame IDs are monotonically increasing and globally unique within
/// a single process lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(u64);

impl FrameId {
    /// Generate the next unique frame ID.
    pub fn next() -> Self {
        Self(NEXT_FRAME_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Create a frame ID from a raw value (for deserialization / testing).
    pub fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Get the raw numeric value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FrameId({})", self.0)
    }
}

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique() {
        let a = FrameId::next();
        let b = FrameId::next();
        assert_ne!(a, b);
        assert!(b.as_u64() > a.as_u64());
    }

    #[test]
    fn ids_are_monotonic() {
        let ids: Vec<FrameId> = (0..100).map(|_| FrameId::next()).collect();
        for window in ids.windows(2) {
            assert!(window[1].as_u64() > window[0].as_u64());
        }
    }

    #[test]
    fn from_raw_roundtrip() {
        let id = FrameId::from_raw(999);
        assert_eq!(id.as_u64(), 999);
    }

    #[test]
    fn display_and_debug() {
        let id = FrameId::from_raw(42);
        assert_eq!(format!("{}", id), "42");
        assert_eq!(format!("{:?}", id), "FrameId(42)");
    }

    #[test]
    fn thread_safety() {
        use std::thread;
        let handles: Vec<_> = (0..8)
            .map(|_| {
                thread::spawn(|| {
                    (0..1000).map(|_| FrameId::next()).collect::<Vec<_>>()
                })
            })
            .collect();

        let mut all_ids: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .map(|id| id.as_u64())
            .collect();

        let total = all_ids.len();
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(all_ids.len(), total, "all IDs must be unique across threads");
    }
}
