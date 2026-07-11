//! Replay real per-chunk VLM descriptions through the ACTUAL [`NoveltyGate`] to
//! see how one novelty signal would sort the same chunks. This is a POST-HOC
//! replay: it reads the VLM's own output, which only exists after the call has
//! been paid for, so it is a post-hoc reference point, not a measurement of
//! live end-to-end savings, and not a proven bound on them in either direction.
//!
//! Pipeline:
//!   1. `benchmarks/measure_novelty.py` runs the VLM at a dense cadence over a
//!      real screen recording and dumps every per-chunk description to
//!      `/tmp/novelty_descriptions.json`. Failed or empty chunks are omitted, so
//!      this replays only the usable chunks and does not price failed calls.
//!   2. This example feeds each description's OCR-style MinHash signature into a
//!      fresh gate, in order, committing every kept chunk, and tallies drop /
//!      escalate / admit across a τ sweep.
//!
//! Two effects pull the live drop rate in opposite directions, so this number
//! is a reference point, not a bound. Text-only MinHash puts a reworded but
//! unchanged screen in the escalate band, while the live gate's semantic
//! embedding collapses those paraphrases into drops, which would push live
//! drops up. But this replay also feeds the gate the VLM's own description
//! text, a richer signal than the live PRE-VLM gate ever sees (it has only OCR
//! text, a frame embedding, and a phash), which cuts the other way. Net
//! direction is not established. Weighting text at 1.0 (embedding and phash
//! zeroed) makes the fused score exactly `1 − max Jaccard` over the window.
//!
//! Run: `cargo run -p vidarax-core --example novelty_replay [path.json]`

use std::fs;

use serde_json::Value;
use vidarax_core::novelty::{MinHashSig, NoveltyConfig, NoveltyDecision, NoveltyGate};

/// One (τ_lo, τ_hi) operating point in the sweep.
struct Op {
    label: &'static str,
    tau_lo: f32,
    tau_hi: f32,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/novelty_descriptions.json".to_string());
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e} (run measure_novelty.py first)"));
    let doc: Value = serde_json::from_str(&raw).expect("invalid JSON");

    let descs: Vec<String> = doc["descriptions"]
        .as_array()
        .expect("descriptions[] missing")
        .iter()
        .filter_map(|r| r["description"].as_str().map(str::to_owned))
        .collect();

    let n = descs.len();
    assert!(n > 0, "no descriptions to replay");

    // Precompute the text signatures once; every operating point replays them.
    let sigs: Vec<MinHashSig> = descs.iter().map(|d| MinHashSig::from_ocr_text(d)).collect();

    // A constant unit embedding keeps the zero-weighted embed term finite (a
    // zero vector would quantize to a NaN cosine, and 0.0 * NaN = NaN would
    // poison the fused score). phash is a constant 0 for the same reason.
    const EMBED: [f32; 1] = [1.0];
    const PHASH: u64 = 0;

    println!(
        "novelty replay — {n} real VLM chunk descriptions from {}",
        doc["source"].as_str().unwrap_or("?")
    );
    println!(
        "  source cadence: fps={} chunk_size={} (dense 'watch everything' extreme)",
        doc["fixed_fps"], doc["chunk_size"]
    );
    println!(
        "  gate: text-only (w_text=1.0, w_embed=0, w_phash=0), window=8 — post-hoc reference\n"
    );

    // A representative spread of thresholds around the shipped defaults
    // (τ_lo=0.12, τ_hi=0.45). Wider drop bands trade recall for spend.
    let ops = [
        Op {
            label: "shipped default",
            tau_lo: 0.12,
            tau_hi: 0.45,
        },
        Op {
            label: "mild",
            tau_lo: 0.15,
            tau_hi: 0.40,
        },
        Op {
            label: "moderate",
            tau_lo: 0.20,
            tau_hi: 0.45,
        },
        Op {
            label: "aggressive",
            tau_lo: 0.25,
            tau_hi: 0.50,
        },
        Op {
            label: "very aggressive",
            tau_lo: 0.30,
            tau_hi: 0.55,
        },
    ];

    println!(
        "{:<16} {:>6} {:>6} {:>4} {:>4} {:>4}   {:>9} {:>10} {:>9}",
        "operating point",
        "τ_lo",
        "τ_hi",
        "drop",
        "esc",
        "adm",
        "drop%",
        "cost×(cons)",
        "cost×(opt)"
    );
    println!("{}", "-".repeat(92));

    for op in &ops {
        let cfg = NoveltyConfig {
            w_text: 1.0,
            w_embed: 0.0,
            w_phash: 0.0,
            tau_hi: op.tau_hi,
            tau_lo: op.tau_lo,
            window: 8,
        };
        let mut gate = NoveltyGate::new(cfg, EMBED.len());

        let (mut drop, mut esc, mut adm) = (0u32, 0u32, 0u32);
        for sig in &sigs {
            match gate.evaluate(sig, &EMBED, PHASH) {
                NoveltyDecision::Drop => drop += 1,
                NoveltyDecision::Escalate { .. } => {
                    esc += 1;
                    // Confirmed-kept: commit so later chunks compare against it.
                    // Conservative — it keeps the window fuller, lowering drops.
                    gate.commit(sig, &EMBED, PHASH);
                }
                NoveltyDecision::Admit { .. } => {
                    adm += 1;
                    gate.commit(sig, &EMBED, PHASH);
                }
            }
        }

        let drop_pct = 100.0 * drop as f32 / n as f32;
        // Conservative: every escalate still costs a full VLM call.
        let cost_cons = n as f32 / (adm + esc).max(1) as f32;
        // Optimistic: escalates resolve on a cheap confirm; only admits pay full.
        let cost_opt = n as f32 / adm.max(1) as f32;

        println!(
            "{:<16} {:>6.2} {:>6.2} {:>4} {:>4} {:>4}   {:>8.1}% {:>9.2}× {:>8.2}×",
            op.label, op.tau_lo, op.tau_hi, drop, esc, adm, drop_pct, cost_cons, cost_opt
        );
    }

    println!(
        "\ncost×(cons) = full-VLM calls saved if every escalate costs a full call \
         (admits+escalates pay).\ncost×(opt)  = if escalates resolve on a cheap confirm \
         (only admits pay full VLM).\nFirst chunk always Admits (empty window). These are a \
         post-hoc reference point from a\nricher signal than the live gate has, not a bound on live drops in either direction."
    );

    // Why escalate dominates: bucket the per-chunk fused novelty at the shipped
    // default so the reader can SEE the mass sitting in the ambiguous band.
    let cfg = NoveltyConfig {
        w_text: 1.0,
        w_embed: 0.0,
        w_phash: 0.0,
        tau_hi: 0.45,
        tau_lo: 0.12,
        window: 8,
    };
    let mut gate = NoveltyGate::new(cfg, EMBED.len());
    let mut hist = [0u32; 10]; // deciles of fused novelty in [0,1)
    for sig in &sigs {
        let (decision, bd) = gate.evaluate_detailed(sig, &EMBED, PHASH);
        let bucket = ((bd.fused * 10.0) as usize).min(9);
        hist[bucket] += 1;
        if !matches!(decision, NoveltyDecision::Drop) {
            gate.commit(sig, &EMBED, PHASH);
        }
    }
    println!("\ntext-novelty distribution (fused = 1 − max Jaccard, shipped default bands):");
    println!("  drop ≤ 0.12          escalate 0.12–0.45          admit ≥ 0.45");
    for (b, &c) in hist.iter().enumerate() {
        let lo = b as f32 / 10.0;
        let band = if lo < 0.12 {
            "drop"
        } else if lo < 0.45 {
            "escalate"
        } else {
            "admit"
        };
        let bar = "█".repeat(c as usize);
        println!("  {lo:.1}–{:.1} {:<9} {:>3} {bar}", lo + 0.1, band, c);
    }
    println!(
        "\nParaphrase-sensitivity: near-identical screens get reworded each call, so token\n\
         Jaccard lands them in the escalate band, not the drop band. A stable semantic\n\
         EMBEDDING (the live gate's w_embed=0.40 signal) would collapse those paraphrases\n\
         into drops, pushing live drops up; but this replay reads the VLM's own post-call\n\
         text, a richer signal than the live pre-VLM gate sees, which pushes the other way.\n\
         So treat these numbers as a reference point, not a proven floor or ceiling."
    );
}
