use vidarax_core::loop_detector::LoopDetector;

#[test]
fn no_loop_with_distinct_hashes() {
    let mut detector = LoopDetector::new(6, 3);
    // Each hash has 8 bits set in a different byte, so every pair has hamming
    // distance 16 — well above the threshold of 6.
    assert!(!detector.check(0x0000_0000_0000_00FF));
    assert!(!detector.check(0x0000_0000_00FF_0000));
    assert!(!detector.check(0x0000_00FF_0000_0000));
    assert!(!detector.check(0xFF00_0000_0000_0000));
}

#[test]
fn detects_loop_after_repeat_trigger() {
    let mut detector = LoopDetector::new(6, 3);
    let hash = 0xDEADBEEF_CAFEBABE;
    assert!(!detector.check(hash));
    assert!(!detector.check(hash));
    assert!(!detector.check(hash));
    // 4th identical hash: 3 of the 8 recent slots now match → triggers
    assert!(detector.check(hash));
}

#[test]
fn similar_hashes_within_threshold_count_as_match() {
    let mut detector = LoopDetector::new(6, 3);
    let base = 0xDEADBEEF_CAFEBABE;
    // Flip 1 bit each time — hamming distance = 1, well under threshold of 6
    assert!(!detector.check(base));
    assert!(!detector.check(base ^ 0x01));
    assert!(!detector.check(base ^ 0x02));
    assert!(detector.check(base ^ 0x04)); // 3 matches within hamming 6
}

#[test]
fn resets_after_different_content() {
    let mut detector = LoopDetector::new(6, 3);
    let loop_hash = 0xAAAA_AAAA_AAAA_AAAA;
    detector.check(loop_hash);
    detector.check(loop_hash);
    detector.check(loop_hash);
    // Fill with completely different hashes to push out the old ones
    for i in 0..8u64 {
        detector.check(i * 0x1111_1111_1111_1111);
    }
    // Now the old loop hash is gone from the buffer
    assert!(!detector.check(loop_hash));
}
