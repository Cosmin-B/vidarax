/// O(1) per-frame loop detection via perceptual hash repetition.
/// Zero allocation, fixed-size ring buffer of 8 recent hashes.
pub struct LoopDetector {
    recent_hashes: [u64; 8],
    cursor: usize,
    threshold: u32,
    repeat_trigger: usize,
}

impl LoopDetector {
    /// Create a new detector.
    /// - `threshold`: max hamming distance to consider two hashes "same screen" (default: 6)
    /// - `repeat_trigger`: how many matches needed to fire (default: 3)
    pub fn new(threshold: u32, repeat_trigger: usize) -> Self {
        Self {
            recent_hashes: [u64::MAX; 8], // initialized to a value unlikely to match anything
            cursor: 0,
            threshold,
            repeat_trigger,
        }
    }

    /// Check if the given hash indicates a loop.
    /// Returns `true` if `repeat_trigger` or more of the recent 8 hashes
    /// are within `threshold` hamming distance of the given hash.
    #[inline]
    pub fn check(&mut self, hash: u64) -> bool {
        let mut matches: u32 = 0;
        for &h in &self.recent_hashes {
            matches += ((h ^ hash).count_ones() < self.threshold) as u32;
        }

        self.recent_hashes[self.cursor % 8] = hash;
        self.cursor = self.cursor.wrapping_add(1);

        matches as usize >= self.repeat_trigger
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.recent_hashes = [u64::MAX; 8];
        self.cursor = 0;
    }
}
