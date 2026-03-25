//! Pre-allocated audio buffer pool for zero-allocation hot paths.
//!
//! The [`AudioBufferPool`] maintains a stack of pre-allocated `BytesMut` buffers
//! that can be acquired and released in O(1) time. This eliminates allocation
//! jitter in real-time audio processing paths.
//!
//! # Design
//!
//! - **O(1) acquire/release**: Stack-based free list (Vec::pop/push).
//! - **Pre-allocated**: All buffers created upfront; no allocation during steady state.
//! - **Capacity guarantee**: Each buffer is pre-allocated to `buffer_size` bytes.
//! - **Overflow safety**: If the pool is exhausted, a new buffer is allocated on demand
//!   (with a tracing warning). This prevents deadlocks at the cost of one allocation.
//!
//! # Example
//!
//! ```
//! use pipecat_audio::buffer_pool::AudioBufferPool;
//!
//! let mut pool = AudioBufferPool::new(4, 640); // 4 buffers of 640 bytes each
//! let mut buf = pool.acquire();
//! buf.extend_from_slice(&[0u8; 640]);
//! pool.release(buf);
//! ```

use bytes::BytesMut;

/// A pool of pre-allocated `BytesMut` buffers for real-time audio processing.
pub struct AudioBufferPool {
    /// Free list of available buffers.
    free: Vec<BytesMut>,
    /// Size of each buffer in bytes.
    buffer_size: usize,
    /// Total number of buffers created (including overflow).
    total_created: usize,
    /// High-water mark of simultaneous buffers in use.
    high_water_mark: usize,
    /// Current number of buffers in use (not in free list).
    in_use: usize,
}

impl AudioBufferPool {
    /// Create a new pool with `count` pre-allocated buffers of `buffer_size` bytes each.
    pub fn new(count: usize, buffer_size: usize) -> Self {
        let free = (0..count)
            .map(|_| BytesMut::with_capacity(buffer_size))
            .collect();
        Self {
            free,
            buffer_size,
            total_created: count,
            high_water_mark: 0,
            in_use: 0,
        }
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns a cleared `BytesMut` with capacity >= `buffer_size`.
    /// If the pool is exhausted, allocates a new buffer (with a tracing warning).
    pub fn acquire(&mut self) -> BytesMut {
        self.in_use += 1;
        if self.in_use > self.high_water_mark {
            self.high_water_mark = self.in_use;
        }

        match self.free.pop() {
            Some(mut buf) => {
                buf.clear();
                buf
            }
            None => {
                // Pool exhausted — allocate on demand to avoid deadlock.
                tracing::warn!(
                    pool_size = self.total_created,
                    in_use = self.in_use,
                    "AudioBufferPool exhausted, allocating overflow buffer"
                );
                self.total_created += 1;
                BytesMut::with_capacity(self.buffer_size)
            }
        }
    }

    /// Release a buffer back to the pool.
    ///
    /// The buffer is returned to the free list for reuse. If the buffer's
    /// capacity has grown beyond `buffer_size` (due to reallocation), it is
    /// dropped to prevent memory bloat.
    pub fn release(&mut self, buf: BytesMut) {
        self.in_use = self.in_use.saturating_sub(1);
        // Only recycle buffers that haven't grown beyond reasonable size.
        // A buffer that was reallocated to 10× the expected size wastes memory.
        if buf.capacity() <= self.buffer_size * 2 {
            self.free.push(buf);
        }
        // else: drop the oversized buffer
    }

    /// Number of buffers currently available in the free list.
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Number of buffers currently in use.
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// High-water mark of simultaneous buffers in use.
    pub fn high_water_mark(&self) -> usize {
        self.high_water_mark
    }

    /// Total number of buffers created (including overflow allocations).
    pub fn total_created(&self) -> usize {
        self.total_created
    }

    /// The size of each buffer in bytes.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_acquire_release() {
        let mut pool = AudioBufferPool::new(4, 640);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.in_use(), 0);

        let buf = pool.acquire();
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.in_use(), 1);
        assert!(buf.capacity() >= 640);

        pool.release(buf);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn overflow_allocation() {
        let mut pool = AudioBufferPool::new(2, 640);

        let _b1 = pool.acquire();
        let _b2 = pool.acquire();
        // Pool exhausted — should still work
        let b3 = pool.acquire();
        assert_eq!(pool.total_created(), 3);
        assert_eq!(pool.in_use(), 3);
        assert!(b3.capacity() >= 640);
    }

    #[test]
    fn high_water_mark_tracking() {
        let mut pool = AudioBufferPool::new(4, 640);
        let b1 = pool.acquire();
        let b2 = pool.acquire();
        assert_eq!(pool.high_water_mark(), 2);

        pool.release(b1);
        assert_eq!(pool.high_water_mark(), 2); // Should not decrease

        let _b3 = pool.acquire();
        let _b4 = pool.acquire();
        let _b5 = pool.acquire();
        assert_eq!(pool.high_water_mark(), 4); // b2 still held + 3 new

        pool.release(b2);
    }

    #[test]
    fn acquired_buffer_is_cleared() {
        let mut pool = AudioBufferPool::new(1, 640);
        let mut buf = pool.acquire();
        buf.extend_from_slice(&[0xFF; 100]);
        pool.release(buf);

        let buf = pool.acquire();
        assert!(buf.is_empty()); // Should be cleared
        assert!(buf.capacity() >= 640);
    }

    #[test]
    fn oversized_buffer_dropped() {
        let mut pool = AudioBufferPool::new(1, 640);
        let mut buf = pool.acquire();
        // Grow the buffer way beyond pool size
        buf.extend_from_slice(&[0u8; 10000]);
        pool.release(buf);
        // Oversized buffer should be dropped, not recycled
        assert_eq!(pool.available(), 0);
    }
}
