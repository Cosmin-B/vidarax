//! Calibrate the live embedding-only novelty policy against labelled raw JPEGs.
//!
//! The manifest is TSV, one candidate per line:
//!
//! ```text
//! pts_ms<TAB>novel|redundant<TAB>path/to/frame.jpg
//! ```
//!
//! The sweep uses the production transport and live reuse policy. Output goes
//! to stdout so evidence can be stored outside the repository.
//!
//! Run:
//!
//! ```text
//! cargo run -p vidarax-core --release --example novelty_live_calibration -- \
//!   127.0.0.1:8765 /tmp/novelty-labels.tsv 800 2000 0.50 0.98 1.10
//! ```
//!
//! Positional values after the manifest are baseline VLM p50 milliseconds,
//! max reuse milliseconds, cumulative drift budget, minimum semantic recall,
//! and minimum net speedup.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vidarax_core::embedding_sidecar::{EmbeddingSidecarClient, EMBEDDING_DIM};
use vidarax_core::novelty::{LiveNoveltyConfig, LiveNoveltyGate, LiveNoveltyOutcome};

const MIN_EVIDENCE_SAMPLES: usize = 30;

struct Candidate {
    pts_ms: u64,
    novel: bool,
    embedding: [f32; EMBEDDING_DIM],
}

#[derive(Clone, Copy)]
struct OperatingPoint {
    threshold: f32,
    admitted: usize,
    reused: usize,
    novel_total: usize,
    novel_admitted: usize,
    redundant_total: usize,
    redundant_reused: usize,
    forced_refresh: usize,
    recall: f64,
    reuse_rate: f64,
    redundant_reuse_rate: f64,
    net_speedup: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: novelty_live_calibration <sidecar_addr> <manifest.tsv> \
             [vlm_p50_ms=800] [max_reuse_ms=2000] [max_drift=0.50] \
             [min_recall=0.98] [min_speedup=1.10]"
        );
        std::process::exit(64);
    }
    let sidecar_addr = &args[1];
    let manifest_path = PathBuf::from(&args[2]);
    let vlm_p50_ms = parse_arg(&args, 3, 800.0_f64, "vlm_p50_ms");
    let max_reuse_ms = parse_arg(&args, 4, 2_000_u64, "max_reuse_ms");
    let max_cumulative_drift = parse_arg(&args, 5, 0.50_f32, "max_drift");
    let min_recall = parse_arg(&args, 6, 0.98_f64, "min_recall");
    let min_speedup = parse_arg(&args, 7, 1.10_f64, "min_speedup");
    if !vlm_p50_ms.is_finite()
        || vlm_p50_ms <= 0.0
        || !max_cumulative_drift.is_finite()
        || max_cumulative_drift <= 0.0
        || !min_recall.is_finite()
        || !(0.0..=1.0).contains(&min_recall)
        || !min_speedup.is_finite()
        || min_speedup <= 0.0
    {
        eprintln!("latency, drift, recall, and speedup limits must be positive and finite");
        std::process::exit(64);
    }

    let rows = read_manifest(&manifest_path);
    if rows.len() < MIN_EVIDENCE_SAMPLES {
        eprintln!(
            "calibration requires at least {MIN_EVIDENCE_SAMPLES} labelled frames; got {}",
            rows.len()
        );
        std::process::exit(65);
    }
    let mut client = EmbeddingSidecarClient::new(sidecar_addr, 30_000)
        .unwrap_or_else(|err| panic!("invalid sidecar address {sidecar_addr}: {err}"));
    let mut candidates = Vec::with_capacity(rows.len());
    let mut embedding_latencies_ms = Vec::with_capacity(rows.len());
    for (pts_ms, novel, image_path) in rows {
        let jpeg = std::fs::read(&image_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", image_path.display()));
        let started = Instant::now();
        let embedding = client
            .embed(&jpeg)
            .unwrap_or_else(|err| panic!("embed {}: {err}", image_path.display()));
        embedding_latencies_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        candidates.push(Candidate {
            pts_ms,
            novel,
            embedding,
        });
    }

    embedding_latencies_ms.sort_by(f64::total_cmp);
    let embedding_mean_ms =
        embedding_latencies_ms.iter().sum::<f64>() / embedding_latencies_ms.len() as f64;
    let embedding_p50_ms = percentile(&embedding_latencies_ms, 0.50);
    let embedding_p95_ms = percentile(&embedding_latencies_ms, 0.95);
    let mut best: Option<OperatingPoint> = None;

    println!("live semantic-novelty calibration");
    println!("  manifest: {}", manifest_path.display());
    println!("  samples: {}", candidates.len());
    println!("  sidecar: {sidecar_addr}");
    println!(
        "  embedding latency: p50={embedding_p50_ms:.2}ms p95={embedding_p95_ms:.2}ms \
         mean={embedding_mean_ms:.2}ms"
    );
    println!(
        "  baseline VLM p50: {vlm_p50_ms:.2}ms; max reuse: {max_reuse_ms}ms; \
         max drift: {max_cumulative_drift:.2}"
    );
    println!();
    println!(
        "{:>7} {:>8} {:>8} {:>10} {:>10} {:>8}",
        "tau_lo", "recall", "reuse", "redundant", "forced", "speedup"
    );

    for step in 1..=50 {
        let threshold = step as f32 / 100.0;
        let point = evaluate(
            &candidates,
            threshold,
            max_reuse_ms,
            max_cumulative_drift,
            vlm_p50_ms,
            embedding_mean_ms,
        );
        println!(
            "{:>7.2} {:>7.2}% {:>7.2}% {:>9.2}% {:>10} {:>7.2}x",
            point.threshold,
            point.recall * 100.0,
            point.reuse_rate * 100.0,
            point.redundant_reuse_rate * 100.0,
            point.forced_refresh,
            point.net_speedup,
        );
        if point.recall >= min_recall
            && best.is_none_or(|current| {
                point.net_speedup > current.net_speedup
                    || (point.net_speedup == current.net_speedup
                        && point.threshold < current.threshold)
            })
        {
            best = Some(point);
        }
    }

    let Some(best) = best else {
        eprintln!("CALIBRATION FAIL: no threshold met semantic recall >= {min_recall:.2}");
        std::process::exit(2);
    };
    println!();
    println!(
        "selected tau_lo={:.2}: recall={:.2}% reuse={:.2}% net_speedup={:.2}x \
         ({}/{} novel admitted; {}/{} redundant reused; {} forced refreshes)",
        best.threshold,
        best.recall * 100.0,
        best.reuse_rate * 100.0,
        best.net_speedup,
        best.novel_admitted,
        best.novel_total,
        best.redundant_reused,
        best.redundant_total,
        best.forced_refresh,
    );
    println!(
        "  calls: admitted={} reused={} total={}",
        best.admitted,
        best.reused,
        candidates.len()
    );
    if best.net_speedup < min_speedup {
        eprintln!(
            "CALIBRATION FAIL: recall-safe policy speedup {:.2}x < required {min_speedup:.2}x",
            best.net_speedup
        );
        std::process::exit(3);
    }
    println!("CALIBRATION PASS");
}

fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, default: T, name: &str) -> T {
    args.get(index).map_or(default, |raw| {
        raw.parse()
            .unwrap_or_else(|_| panic!("invalid {name}: {raw}"))
    })
}

type ManifestRow = (u64, bool, PathBuf);

fn read_manifest(path: &Path) -> Vec<ManifestRow> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read manifest {}: {err}", path.display()));
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut rows = Vec::new();
    let mut last_pts = None;
    for (line_number, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("pts_ms\t") {
            continue;
        }
        let mut fields = line.splitn(3, '\t');
        let pts_ms: u64 = fields
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("manifest line {} has invalid pts_ms", line_number + 1));
        if last_pts.is_some_and(|last| pts_ms < last) {
            panic!("manifest line {} moves backwards in time", line_number + 1);
        }
        last_pts = Some(pts_ms);
        let novel = match fields.next().map(str::trim) {
            Some("novel" | "change" | "1" | "true") => true,
            Some("redundant" | "same" | "0" | "false") => false,
            _ => panic!("manifest line {} has invalid label", line_number + 1),
        };
        let image = fields
            .next()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| panic!("manifest line {} has no image path", line_number + 1));
        let image_path = PathBuf::from(image);
        rows.push((
            pts_ms,
            novel,
            if image_path.is_absolute() {
                image_path
            } else {
                parent.join(image_path)
            },
        ));
    }
    rows
}

fn evaluate(
    candidates: &[Candidate],
    threshold: f32,
    max_reuse_ms: u64,
    max_cumulative_drift: f32,
    vlm_p50_ms: f64,
    embedding_mean_ms: f64,
) -> OperatingPoint {
    let config = LiveNoveltyConfig {
        reuse_threshold: threshold,
        max_reuse_ms,
        max_cumulative_drift,
        ..LiveNoveltyConfig::default()
    };
    let mut gate = LiveNoveltyGate::try_new(&config).expect("valid calibration config");
    let mut admitted = 0;
    let mut reused = 0;
    let mut novel_total = 0;
    let mut novel_admitted = 0;
    let mut redundant_total = 0;
    let mut redundant_reused = 0;
    let mut forced_refresh = 0;

    for candidate in candidates {
        let outcome = gate.evaluate(&candidate.embedding, candidate.pts_ms);
        let reuse = matches!(outcome, LiveNoveltyOutcome::Reuse);
        if candidate.novel {
            novel_total += 1;
            if !reuse {
                novel_admitted += 1;
            }
        } else {
            redundant_total += 1;
            if reuse {
                redundant_reused += 1;
            }
        }
        if reuse {
            reused += 1;
        } else {
            admitted += 1;
            if matches!(outcome, LiveNoveltyOutcome::ForcedRefresh) {
                forced_refresh += 1;
            }
            gate.commit(&candidate.embedding, candidate.pts_ms);
        }
    }

    let total = candidates.len();
    let recall = ratio(novel_admitted, novel_total);
    let reuse_rate = ratio(reused, total);
    let redundant_reuse_rate = ratio(redundant_reused, redundant_total);
    let baseline_ms = total as f64 * vlm_p50_ms;
    let gated_ms = total as f64 * embedding_mean_ms + admitted as f64 * vlm_p50_ms;
    OperatingPoint {
        threshold,
        admitted,
        reused,
        novel_total,
        novel_admitted,
        redundant_total,
        redundant_reused,
        forced_refresh,
        recall,
        reuse_rate,
        redundant_reuse_rate,
        net_speedup: baseline_ms / gated_ms.max(f64::EPSILON),
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}
