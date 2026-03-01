//! Duplicate-description suppression for the VLM worker pipeline.
//!
//! When the system is stuck in a visual loop the gate engine may fire
//! repeatedly on frames that are perceptually identical. `DedupFilter`
//! prevents redundant SpacetimeDB writes by comparing each outgoing VLM
//! description against the last emitted one using a fast FNV-1a hash and
//! an exact-string guard.
//!
//! # Example
//!
//! ```rust
//! use vidarax_core::dedup::DedupFilter;
//!
//! let mut filter = DedupFilter::new();
//!
//! assert!(filter.should_emit("person walking left"));
//! assert!(!filter.should_emit("person walking left")); // duplicate suppressed
//! assert!(filter.should_emit("person running right")); // changed — emit
//! assert_eq!(filter.suppressed_count(), 1);
//! ```

/// FNV-1a 64-bit offset basis and prime (public domain algorithm).
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute the FNV-1a 64-bit hash of a byte sequence.
///
/// This is O(n) in the string length and allocation-free.
#[inline]
fn fnv_hash(s: &str) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Per-session duplicate-description filter.
///
/// Keeps track of the last emitted VLM description (via its FNV-1a hash and
/// the full string) and suppresses re-emission of identical content.
///
/// # Thread safety
///
/// `DedupFilter` is intentionally *not* `Sync`. Each VLM worker thread owns
/// its own instance, avoiding any shared-state synchronisation overhead.
pub struct DedupFilter {
    last_description: String,
    last_hash: u64,
    suppressed_count: u64,
}

impl DedupFilter {
    /// Create a new filter with no prior state (first call always emits).
    pub fn new() -> Self {
        Self {
            last_description: String::new(),
            // A hash of the empty string is not `0`, so we use `0` as a
            // sentinel that can never collide with a real description hash.
            last_hash: 0,
            suppressed_count: 0,
        }
    }

    /// Decide whether the given description should be emitted.
    ///
    /// Returns `true` (emit) when the description has changed since the last
    /// call; returns `false` (suppress) when it is identical to the last
    /// emitted description.
    ///
    /// Two-phase comparison: the FNV-1a hash is checked first (O(1)) and the
    /// full string equality only if the hash matches (guards against the
    /// astronomically unlikely hash collision).
    pub fn should_emit(&mut self, description: &str) -> bool {
        let hash = fnv_hash(description);
        if hash == self.last_hash && description == self.last_description {
            self.suppressed_count += 1;
            return false;
        }
        self.last_description.clear();
        self.last_description.push_str(description);
        self.last_hash = hash;
        true
    }

    /// Number of descriptions suppressed since this filter was created (or
    /// last reset via [`DedupFilter::reset`]).
    pub fn suppressed_count(&self) -> u64 {
        self.suppressed_count
    }

    /// Reset the filter to its initial state.
    ///
    /// After reset the next call to [`should_emit`] always returns `true`
    /// regardless of its argument.
    pub fn reset(&mut self) {
        self.last_description.clear();
        self.last_hash = 0;
        self.suppressed_count = 0;
    }
}

impl Default for DedupFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{DedupFilter, fnv_hash};

    #[test]
    fn first_call_always_emits() {
        let mut f = DedupFilter::new();
        assert!(f.should_emit("hello world"));
    }

    #[test]
    fn identical_description_is_suppressed() {
        let mut f = DedupFilter::new();
        assert!(f.should_emit("same content"));
        assert!(!f.should_emit("same content"));
        assert!(!f.should_emit("same content"));
        assert_eq!(f.suppressed_count(), 2);
    }

    #[test]
    fn changed_description_emits_again() {
        let mut f = DedupFilter::new();
        assert!(f.should_emit("frame A"));
        assert!(!f.should_emit("frame A"));
        assert!(f.should_emit("frame B"));
        assert_eq!(f.suppressed_count(), 1);
    }

    #[test]
    fn reset_clears_state() {
        let mut f = DedupFilter::new();
        assert!(f.should_emit("content"));
        assert!(!f.should_emit("content"));

        f.reset();

        assert!(f.should_emit("content")); // after reset, same content emits again
        assert_eq!(f.suppressed_count(), 0);
    }

    #[test]
    fn empty_string_description_is_handled() {
        let mut f = DedupFilter::new();
        assert!(f.should_emit("")); // sentinel hash 0 differs from fnv("") which is non-zero
        assert!(!f.should_emit(""));
    }

    #[test]
    fn fnv_hash_is_deterministic() {
        let a = fnv_hash("hello");
        let b = fnv_hash("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn fnv_hash_differs_for_distinct_inputs() {
        assert_ne!(fnv_hash("alpha"), fnv_hash("beta"));
        assert_ne!(fnv_hash(""), fnv_hash("a"));
    }

    #[test]
    fn default_is_same_as_new() {
        let mut a = DedupFilter::new();
        let mut b = DedupFilter::default();
        // Both should emit on first call.
        assert!(a.should_emit("x"));
        assert!(b.should_emit("x"));
    }

    #[test]
    fn suppressed_count_tracks_across_multiple_bursts() {
        let mut f = DedupFilter::new();
        f.should_emit("A");
        for _ in 0..5 {
            f.should_emit("A");
        }
        f.should_emit("B");
        for _ in 0..3 {
            f.should_emit("B");
        }
        assert_eq!(f.suppressed_count(), 8);
    }
}
