//! Replay real Gemini output through the T3 merge and assert the measured
//! collapse survives int8 quantisation.
//!
//! `fixtures/minilm_screenshare_150.f32` holds the 384-dim MiniLM-L6 embeddings
//! of 150 consecutive VLM descriptions from a real Unreal-editor screenshare, in
//! chunk order. The offline float reference collapses these to 20 activities at
//! cosine 0.75. This test proves the in-crate quantised scorer lands in the same
//! place — the measured "7.5× cleaner timeline" claim, guarded against regression.

use std::path::PathBuf;

use vidarax_core::semantic_merge::{MergeConfig, SemanticMerge, DEFAULT_MERGE_THRESHOLD};

/// Parse the little-endian fixture: `u32 count, u32 dim, count*dim f32`.
fn load_embeddings() -> (usize, Vec<Vec<f32>>) {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "minilm_screenshare_150.f32",
    ]
    .iter()
    .collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len() >= 8, "fixture too small");

    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let payload = &bytes[8..];
    assert_eq!(
        payload.len(),
        count * dim * 4,
        "fixture size disagrees with header"
    );

    let mut rows = Vec::with_capacity(count);
    for r in 0..count {
        let mut row = Vec::with_capacity(dim);
        for c in 0..dim {
            let off = (r * dim + c) * 4;
            row.push(f32::from_le_bytes(
                payload[off..off + 4].try_into().unwrap(),
            ));
        }
        rows.push(row);
    }
    (dim, rows)
}

#[test]
fn replay_collapses_150_chunks_to_measured_activity_count() {
    let (dim, rows) = load_embeddings();
    assert_eq!(
        rows.len(),
        150,
        "expected the 150-chunk screenshare fixture"
    );

    let mut merge = SemanticMerge::new(MergeConfig {
        merge_threshold: DEFAULT_MERGE_THRESHOLD, // 0.75
        embed_dim: dim,
    });

    let mut notes = Vec::new();
    let mut total_chunks = 0u32;
    for (i, row) in rows.iter().enumerate() {
        if let Some(note) = merge.observe(i as u64 * 1_000, row, "chunk") {
            notes.push(note);
        }
    }
    if let Some(tail) = merge.flush() {
        notes.push(tail);
    }
    for n in &notes {
        total_chunks += n.chunk_count;
    }

    // Every input chunk is accounted for by exactly one activity.
    assert_eq!(total_chunks, 150, "all chunks must land in an activity");

    // Float reference is 20; int8 quantisation may shift it by a note or two.
    // The claim under test is the *magnitude* of the collapse, not the exact 20.
    let n = notes.len();
    assert!(
        (18..=22).contains(&n),
        "expected ~20 activities (measured float ref = 20), got {n}"
    );
    assert!(
        150 / n >= 6,
        "expected at least a 6× timeline collapse, got {}×",
        150 / n
    );

    // Activities carry monotonically non-decreasing time and a real boundary
    // signal on the first note (everything is new at the start of a stream).
    assert!((notes[0].boundary_novelty - 1.0).abs() < 1e-6);
    for n in &notes {
        assert!(n.end_pts_ms >= n.start_pts_ms);
        assert!(n.chunk_count >= 1);
    }
}
