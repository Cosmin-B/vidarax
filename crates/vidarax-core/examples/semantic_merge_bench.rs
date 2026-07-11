//! Bench + honest cost model for the T3 merge and the static-frame gate.
//!
//! Run: `cargo run --release --example semantic_merge_bench`
//!
//! Reports three things:
//!   1. per-`observe` latency merging 150 real Gemini descriptions (goal A: fast)
//!   2. per-`decide` latency of the upstream static gate (per *frame*, so hotter)
//!   3. the two levers on the Gemini bill (goal B), stated honestly per regime.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use vidarax_core::semantic_merge::{
    GateDecision, MergeConfig, PhashFrameGate, PhashGateConfig, SemanticMerge, StaticFrameGate,
    DEFAULT_MERGE_THRESHOLD,
};

fn load_fixture() -> (usize, Vec<Vec<f32>>) {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "minilm_screenshare_150.f32",
    ]
    .iter()
    .collect();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let payload = &bytes[8..];
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

fn run_once(rows: &[Vec<f32>], dim: usize) -> usize {
    let mut m = SemanticMerge::new(MergeConfig {
        merge_threshold: DEFAULT_MERGE_THRESHOLD,
        embed_dim: dim,
    });
    let mut notes = 0usize;
    for (i, row) in rows.iter().enumerate() {
        if m.observe(i as u64 * 1000, row, "chunk").is_some() {
            notes += 1;
        }
    }
    if m.flush().is_some() {
        notes += 1;
    }
    notes
}

fn main() {
    let (dim, rows) = load_fixture();
    let n_chunks = rows.len();
    println!("fixture: {n_chunks} real descriptions, {dim}-dim MiniLM embeddings\n");

    // ---- (1) merge throughput ----
    let notes = run_once(&rows, dim);
    let iters = 4000;
    // warm
    for _ in 0..50 {
        black_box(run_once(&rows, dim));
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(run_once(&rows, dim));
    }
    let elapsed = t0.elapsed();
    let per_observe = elapsed.as_nanos() as f64 / (iters as f64 * n_chunks as f64);
    println!("T3 MERGE (goal A — runs once per VLM description, ~1/sec)");
    println!(
        "  collapse:      {n_chunks} chunks -> {notes} activities ({:.1}x cleaner timeline)",
        n_chunks as f64 / notes as f64
    );
    println!(
        "  per observe():  {per_observe:.0} ns  ({:.2} µs)",
        per_observe / 1000.0
    );
    println!(
        "  full 150-chunk pass: {:.1} µs total\n",
        elapsed.as_nanos() as f64 / iters as f64 / 1000.0
    );

    // ---- (2) static-gate throughput (per frame — the hotter path) ----
    // Synthetic frame embeddings reusing the description dim: 32 frozen frames
    // (a paused/idle screen) then one that moved, repeated.
    let dimf = dim;
    let frozen = rows[0].clone();
    let moved = rows[rows.len() / 2].clone();
    let mut gate = StaticFrameGate::with_dim(dimf);
    let mut frames: Vec<&Vec<f32>> = Vec::new();
    for _ in 0..200 {
        for _ in 0..32 {
            frames.push(&frozen);
        }
        frames.push(&moved);
    }
    // warm
    for f in frames.iter().take(64) {
        black_box(gate.decide(f));
    }
    let mut gate = StaticFrameGate::with_dim(dimf);
    let mut reuse = 0usize;
    let t1 = Instant::now();
    for f in &frames {
        if gate.decide(f) == GateDecision::Reuse {
            reuse += 1;
        }
    }
    let e1 = t1.elapsed();
    let per_decide = e1.as_nanos() as f64 / frames.len() as f64;
    println!("STATIC GATE — embedding variant (needs a per-frame embedding)");
    println!(
        "  per decide():  {per_decide:.0} ns  ({:.2} µs)",
        per_decide / 1000.0
    );
    println!(
        "  on a SYNTHETIC 32-frozen : 1-moved pattern: {}/{} frames reused ({:.0}% calls skipped)",
        reuse,
        frames.len(),
        100.0 * reuse as f64 / frames.len() as f64
    );

    // ---- (2b) pHash gate — the recommended per-frame path (no embedding) ----
    // Same idle pattern, but keyed on the 64-bit hash the frame pipeline already
    // computes. One XOR + popcount per frame; no embedding, no server, no ONNX.
    let frozen_h: u64 = 0xABCD_1234_5678_9F0E;
    let moved_h: u64 = 0x0123_4567_89AB_CDEF;
    let mut phashes: Vec<u64> = Vec::new();
    for _ in 0..200 {
        phashes.extend(std::iter::repeat_n(frozen_h, 32));
        phashes.push(moved_h);
    }
    let mut pgate = PhashFrameGate::new(PhashGateConfig::new());
    for &h in phashes.iter().take(64) {
        black_box(pgate.decide(h));
    }
    let mut pgate = PhashFrameGate::new(PhashGateConfig::new());
    let mut preuse = 0usize;
    let t2 = Instant::now();
    for &h in &phashes {
        if pgate.decide(h) == GateDecision::Reuse {
            preuse += 1;
        }
    }
    let e2 = t2.elapsed();
    let per_pdecide = e2.as_nanos() as f64 / phashes.len() as f64;
    println!("\nPHASH GATE — per-frame path (no embedding, no server)");
    println!("  per decide():  {per_pdecide:.1} ns  (one XOR + popcount)");
    println!(
        "  on a SYNTHETIC 32-frozen : 1-moved pattern: {}/{} frames reused ({:.0}% calls skipped)",
        preuse,
        phashes.len(),
        100.0 * preuse as f64 / phashes.len() as f64
    );

    // ---- (3) the two levers, stated honestly ----
    println!("\nWHAT THIS COULD DO TO THE GEMINI BILL");
    println!(
        "  idle / paused screen  -> pHash gate skips the call on the synthetic pattern above."
    );
    println!("  active chrome content -> gate correctly almost never fires (every frame might");
    println!("                           carry a moment); the merge still collapses the OUTPUT");
    println!(
        "                           {:.1}x on this fixture, cutting repeated notes.",
        n_chunks as f64 / notes.max(1) as f64
    );
    println!("  the frozen-frame reuse is a calibrated heuristic: its usual error is overspend,");
    println!("  but a lossy hash can reuse across a real change, so the risk is low, not zero.");
}
