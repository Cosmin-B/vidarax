use vidarax_core::ingest::compute_semantic_frame_indices;

#[test]
fn indices_for_single_chunk() {
    // 25 frames, chunk_size=25 (1 chunk), 2 frames per chunk
    // select_semantic_images picks frame 0 and frame 24 (evenly spaced)
    let indices = compute_semantic_frame_indices(25, 25, 2);
    assert_eq!(indices, vec![0, 24]);
}

#[test]
fn indices_for_multiple_chunks() {
    // 50 frames, chunk_size=25 (2 chunks), 2 frames per chunk
    // chunk 0: frames 0-24 → selects 0, 24
    // chunk 1: frames 25-49 → selects 25, 49
    let indices = compute_semantic_frame_indices(50, 25, 2);
    assert_eq!(indices, vec![0, 24, 25, 49]);
}

#[test]
fn single_frame_per_chunk_picks_middle() {
    // 30 frames, chunk_size=10, 1 frame per chunk
    // chunk 0: frames 0-9 → middle = 5
    // chunk 1: frames 10-19 → middle = 15
    // chunk 2: frames 20-29 → middle = 25
    let indices = compute_semantic_frame_indices(30, 10, 1);
    assert_eq!(indices, vec![5, 15, 25]);
}

#[test]
fn partial_last_chunk() {
    // 27 frames, chunk_size=25, 2 frames per chunk
    // chunk 0: frames 0-24 → selects 0, 24
    // chunk 1: frames 25-26 (only 2 frames) → selects 25, 26
    let indices = compute_semantic_frame_indices(27, 25, 2);
    assert_eq!(indices, vec![0, 24, 25, 26]);
}

#[test]
fn zero_frames_returns_empty() {
    assert!(compute_semantic_frame_indices(0, 25, 2).is_empty());
    assert!(compute_semantic_frame_indices(100, 25, 0).is_empty());
}

#[test]
fn frames_per_chunk_exceeds_chunk_size() {
    // 10 frames, chunk_size=5, 8 frames per chunk (more than chunk has)
    // chunk 0: frames 0-4 → all 5 selected
    // chunk 1: frames 5-9 → all 5 selected
    let indices = compute_semantic_frame_indices(10, 5, 8);
    assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
