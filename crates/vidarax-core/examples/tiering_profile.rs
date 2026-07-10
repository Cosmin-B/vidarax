//! Tiering profile: turns two real, already-produced artifacts into one
//! read-only report on how a live tiered VLM run actually behaved.
//!
//! Inputs (both are things the running system writes today; nothing new is
//! captured or generated here):
//!
//!   1. the WAL append log -- newline-delimited `TimelineEvent` records
//!      written by `vidarax_core::timeline` (crates/vidarax-core/src/timeline.rs).
//!      Read with `read_all_events`; this tool never writes to the WAL.
//!   2. an optional saved Prometheus text dump, produced by the operator with
//!      `curl <server>/metrics > metrics.txt`. Read as a plain file so this
//!      stays offline and dependency-free; nothing here opens a socket.
//!
//! Confirmed by reading the source before writing this tool:
//!
//!   - The escalation event kinds come from two call sites, both driven by
//!     the same `used_second_pass` flag returned from `run_tiered`, and both
//!     are real, confirmed WAL kinds (four total, not two). The per-frame
//!     path (crates/vidarax-core/src/webrtc/workers.rs, around line 1495):
//!     `event_type = if used_second_pass { "vlm_tiered" } else { "vlm" }`.
//!     The WHIP clip-mode path (crates/vidarax-core/src/webrtc/clip.rs,
//!     around line 434), which batches several buffered frames into one
//!     tiered call instead of inferring per frame, uses its own kind
//!     strings so it doesn't collide with the per-frame counts above:
//!     `event_type = if used_second_pass { "clip_vlm_tiered" } else {
//!     "clip_vlm" }`. Both call `WalEventSink::emit_event_sync`, which
//!     calls `append_run_event(run_id, event_type, payload)`
//!     (crates/vidarax-api/src/wal_sink.rs), and `append_run_event` passes
//!     `kind` straight into the stored `TimelineEvent`
//!     (crates/vidarax-api/src/state.rs). So all four strings -- `vlm`,
//!     `vlm_tiered`, `clip_vlm`, `clip_vlm_tiered` -- are confirmed WAL
//!     kinds, and the funnel below counts all four: `vlm`/`clip_vlm` as
//!     local-only, `vlm_tiered`/`clip_vlm_tiered` as escalated. A WAL with
//!     only clip-mode events (no `vlm`/`vlm_tiered`) still produces a
//!     non-empty funnel.
//!   - "tier1_knn_hit" is NOT a WAL event kind. The only place that string
//!     appears in the codebase is inside a
//!     `tracing::info!(..., "tier1_knn_hit: skipping vlm inference")` call in
//!     workers.rs, gated behind `#[cfg(feature = "training")]`. That call
//!     never touches `event_tx` or the WAL writer, so a KNN cache hit
//!     produces a log line and nothing else durable. This tool does not
//!     invent a WAL count for it: cache hits are reported below as
//!     "not in WAL".
//!   - The Prometheus metric names come straight from `render_prometheus` in
//!     crates/vidarax-api/src/inference_metrics.rs (not a markdown list, just
//!     the four exact metric names, one per line, for grep-ability):
//!     `vidarax_infer_requests_total{provider="...",status="ok"|"error"}`
//!     `vidarax_infer_tokens_total{provider="...",kind="prompt"|"completion"|"thinking"}`
//!     `vidarax_infer_latency_ms_sum{provider="..."}`
//!     `vidarax_infer_latency_ms_count{provider="..."}`
//!     Providers are "vllm", "sglang", "gemini", "mlx". These counters are
//!     process-global per-provider inference totals, not scoped to any one
//!     run or code path, so the "local" (vllm+sglang+mlx) vs "gemini" split
//!     below is per-provider inference totals from /metrics, and includes
//!     any direct inference on that provider, not only WHIP tiering
//!     escalations. Do not read it as an isolated measurement of tiering.
//!
//! Usage:
//!   cargo run -p vidarax-core --example tiering_profile -- <wal_path> [metrics_path]
//!   cargo run -p vidarax-core --example tiering_profile -- <metrics_path>   (metrics-only)
//!
//! The two arguments don't need a fixed order: each path is sniffed by its
//! first non-blank line (a `vidarax_infer...` line means metrics; a
//! tab-separated line whose first field parses as a number means WAL) so a
//! metrics-only or WAL-only call both work from a single argument, and a
//! two-argument call works regardless of which path comes first.
//!
//! Read-only end to end: no WAL append, no HTTP request. Every number below
//! comes from artifacts a real run produced.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use vidarax_core::timeline::{read_all_events, TimelineEvent};

// The four WAL event kinds the tiering workers actually emit (confirmed
// above): the per-frame path uses vlm/vlm_tiered, the WHIP clip-mode path
// uses clip_vlm/clip_vlm_tiered for the same local-vs-escalated distinction.
const KIND_LOCAL_ONLY: &str = "vlm";
const KIND_ESCALATED: &str = "vlm_tiered";
const KIND_LOCAL_ONLY_CLIP: &str = "clip_vlm";
const KIND_ESCALATED_CLIP: &str = "clip_vlm_tiered";

// Providers recognized in the /metrics dump (inference_metrics.rs). These
// are process-global per-provider inference totals, not scoped to WHIP
// tiering, so grouping them this way is a labeling convenience only (see
// the module doc comment and print_metrics_split). mlx is on-device VLM
// inference (mlx-vlm on Apple Silicon), so it counts on the LOCAL side of
// the split alongside vllm/sglang, not with the hosted second-pass provider.
const LOCAL_PROVIDERS: [&str; 3] = ["vllm", "sglang", "mlx"];
const SECOND_PASS_PROVIDERS: [&str; 1] = ["gemini"];

// ─── WAL funnel ─────────────────────────────────────────────────────────────

/// Local-only vs escalated keyframe counts, either overall or for one run.
#[derive(Debug, Default, Clone, Copy)]
struct RunFunnel {
    local_only: u64,
    escalated: u64,
}

impl RunFunnel {
    fn inferred(&self) -> u64 {
        self.local_only + self.escalated
    }
}

/// Walks the decoded WAL events once and buckets local-only / escalated
/// counts, both overall and per `run_id`. Counts all four confirmed kinds:
/// `vlm`/`clip_vlm` as local-only, `vlm_tiered`/`clip_vlm_tiered` as
/// escalated (the clip-mode pair comes from the WHIP clip path, see the
/// module doc comment). Every other event kind (`scene_cut`,
/// `state_transition`, `run_deleted`, ...) is left out on purpose: this is a
/// tiering funnel, not a general event count.
fn compute_funnel(events: &[TimelineEvent]) -> (RunFunnel, BTreeMap<String, RunFunnel>) {
    let mut overall = RunFunnel::default();
    let mut per_run: BTreeMap<String, RunFunnel> = BTreeMap::new();

    for event in events {
        match event.kind.as_str() {
            KIND_LOCAL_ONLY | KIND_LOCAL_ONLY_CLIP => {
                overall.local_only += 1;
                per_run.entry(event.run_id.clone()).or_default().local_only += 1;
            }
            KIND_ESCALATED | KIND_ESCALATED_CLIP => {
                overall.escalated += 1;
                per_run.entry(event.run_id.clone()).or_default().escalated += 1;
            }
            _ => {}
        }
    }

    (overall, per_run)
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (whole as f64)
    }
}

/// Prints one funnel row (overall or one run). When a row has zero
/// escalations but at least one local-only keyframe, this prints the exact
/// "tiering not active" note the caller asked for so a flat run is obvious
/// at a glance instead of reading as a rounding artifact.
fn print_funnel_row(label: &str, funnel: &RunFunnel) {
    let inferred = funnel.inferred();
    if inferred == 0 {
        println!("    {label:<28} no vlm / vlm_tiered / clip_vlm / clip_vlm_tiered events");
        return;
    }
    println!(
        "    {:<28} {:>10} {:>10} {:>11.1}%",
        label,
        funnel.local_only,
        funnel.escalated,
        pct(funnel.escalated, inferred)
    );
    if funnel.escalated == 0 {
        println!("      0 escalations: tiering not active on this run");
    }
}

fn print_wal_funnel(path: &Path, events: &[TimelineEvent]) {
    println!("=== WAL funnel: {} ===", path.display());
    println!("  events read (all kinds): {}", events.len());
    println!("  cache hits (tier1_knn_hit): not in WAL -- log-only, see module doc comment");

    let (overall, per_run) = compute_funnel(events);
    if overall.inferred() == 0 {
        println!("  no vlm / vlm_tiered / clip_vlm / clip_vlm_tiered events found (empty WAL, missing file, or only other event kinds present)");
        println!();
        return;
    }

    println!();
    println!(
        "    {:<28} {:>10} {:>10} {:>12}",
        "", "local", "escalated", "escal. rate"
    );
    print_funnel_row("OVERALL", &overall);
    println!();

    if per_run.len() > 1 {
        println!("  per run:");
        for (run_id, funnel) in &per_run {
            print_funnel_row(run_id, funnel);
        }
        println!();
    }
}

// ─── Metrics split ──────────────────────────────────────────────────────────

/// Everything this tool reads out of one provider's block of
/// `vidarax_infer_*` lines. Fields left at zero when the corresponding line
/// is absent from the dump (e.g. a provider that never ran).
#[derive(Debug, Default, Clone, Copy)]
struct ProviderTotals {
    ok: u64,
    err: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    thinking_tokens: u64,
    latency_sum_ms: u64,
    latency_count: u64,
}

impl ProviderTotals {
    // Saturating rather than plain `+`: a malformed or adversarial /metrics
    // dump can carry a value that clamps to u64::MAX on the way in (see
    // parse_metric_line), and two such lines added together would otherwise
    // overflow and panic in a debug build. Saturating just caps the report
    // at u64::MAX instead of crashing a read-only tool over bad input.
    fn requests(&self) -> u64 {
        self.ok.saturating_add(self.err)
    }

    fn tokens(&self) -> u64 {
        self.prompt_tokens
            .saturating_add(self.completion_tokens)
            .saturating_add(self.thinking_tokens)
    }

    fn mean_latency_ms(&self) -> Option<f64> {
        if self.latency_count == 0 {
            None
        } else {
            Some(self.latency_sum_ms as f64 / self.latency_count as f64)
        }
    }

    fn add(&mut self, other: &ProviderTotals) {
        self.ok = self.ok.saturating_add(other.ok);
        self.err = self.err.saturating_add(other.err);
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.thinking_tokens = self.thinking_tokens.saturating_add(other.thinking_tokens);
        self.latency_sum_ms = self.latency_sum_ms.saturating_add(other.latency_sum_ms);
        self.latency_count = self.latency_count.saturating_add(other.latency_count);
    }
}

/// `(metric_name, label pairs, value)` for one parsed Prometheus line.
type MetricLine<'a> = (&'a str, Vec<(&'a str, &'a str)>, f64);

/// Parses one Prometheus text-exposition line into a [`MetricLine`]. Mirrors
/// the flat, one-sample-per-line shape `render_prometheus` produces
/// (crates/vidarax-api/src/inference_metrics.rs): no HELP/TYPE comments to
/// skip over except defensively, no multi-line samples.
fn parse_metric_line(line: &str) -> Option<MetricLine<'_>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let brace_open = line.find('{')?;
    let brace_close = line.find('}')?;
    if brace_close < brace_open {
        return None;
    }

    let name = &line[..brace_open];
    let label_str = &line[brace_open + 1..brace_close];
    let value_str = line[brace_close + 1..].trim();
    let value: f64 = value_str.parse().ok()?;
    // A counter is never NaN, infinite, or negative. Reject those here so a
    // malformed or adversarial dump (a stray "nan"/"inf" token, a negative
    // sample) is skipped instead of flowing into the u64 totals below.
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let mut labels = Vec::new();
    for pair in label_str.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let eq = pair.find('=')?;
        let key = &pair[..eq];
        let raw_val = pair[eq + 1..].trim();
        labels.push((key, raw_val.trim_matches('"')));
    }

    Some((name, labels, value))
}

/// Converts an already-validated (finite, non-negative) Prometheus sample
/// into a u64 count. `parse_metric_line` has already rejected NaN/inf/
/// negative values by this point; this clamp only guards the remaining
/// case, an oversized-but-finite value (e.g. `1e30`), so the cast below
/// lands on u64::MAX instead of relying on the implicit saturation the `as`
/// cast happens to perform.
fn saturating_metric_value(value: f64) -> u64 {
    value.min(u64::MAX as f64) as u64
}

/// Pure parse: text in, per-provider totals out. Kept separate from any I/O
/// so it can run against a fixture in a unit test without touching disk.
fn parse_metrics(text: &str) -> BTreeMap<String, ProviderTotals> {
    let mut out: BTreeMap<String, ProviderTotals> = BTreeMap::new();

    for line in text.lines() {
        let Some((name, labels, value)) = parse_metric_line(line) else {
            continue;
        };
        let Some(provider) = labels
            .iter()
            .find(|(k, _)| *k == "provider")
            .map(|(_, v)| *v)
        else {
            continue;
        };
        let entry = out.entry(provider.to_string()).or_default();

        match name {
            "vidarax_infer_requests_total" => {
                match labels.iter().find(|(k, _)| *k == "status").map(|(_, v)| *v) {
                    Some("ok") => entry.ok = saturating_metric_value(value),
                    Some("error") => entry.err = saturating_metric_value(value),
                    _ => {}
                }
            }
            "vidarax_infer_tokens_total" => {
                match labels.iter().find(|(k, _)| *k == "kind").map(|(_, v)| *v) {
                    Some("prompt") => entry.prompt_tokens = saturating_metric_value(value),
                    Some("completion") => entry.completion_tokens = saturating_metric_value(value),
                    Some("thinking") => entry.thinking_tokens = saturating_metric_value(value),
                    _ => {}
                }
            }
            "vidarax_infer_latency_ms_sum" => entry.latency_sum_ms = saturating_metric_value(value),
            "vidarax_infer_latency_ms_count" => {
                entry.latency_count = saturating_metric_value(value)
            }
            // fallback_total, latency_ms_bucket, slo_target_ratio, and
            // error_budget_remaining_ratio are real lines in the dump but are
            // not part of the local vs second-pass split asked for here.
            _ => {}
        }
    }

    out
}

fn aggregate_providers(
    totals: &BTreeMap<String, ProviderTotals>,
    names: &[&str],
) -> ProviderTotals {
    let mut agg = ProviderTotals::default();
    for name in names.iter().copied() {
        if let Some(t) = totals.get(name) {
            agg.add(t);
        }
    }
    agg
}

fn print_provider_row(provider: &str, t: &ProviderTotals) {
    match t.mean_latency_ms() {
        Some(mean) => println!(
            "    {:<10} {:>8} {:>8} {:>12} {:>12} {:>12} {:>16} {:>10.1}",
            provider,
            t.ok,
            t.err,
            t.prompt_tokens,
            t.completion_tokens,
            t.thinking_tokens,
            t.latency_sum_ms,
            mean
        ),
        None => println!(
            "    {:<10} {:>8} {:>8} {:>12} {:>12} {:>12} {:>16} {:>10}",
            provider,
            t.ok,
            t.err,
            t.prompt_tokens,
            t.completion_tokens,
            t.thinking_tokens,
            t.latency_sum_ms,
            "n/a"
        ),
    }
}

fn print_split_line(label: &str, part: u64, whole: u64) {
    println!("    {label:<12} {part:>10}   ({:>5.1}%)", pct(part, whole));
}

fn print_metrics_split(path: &Path, totals: &BTreeMap<String, ProviderTotals>) {
    println!("=== metrics split: {} ===", path.display());
    if totals.is_empty() {
        println!("  no vidarax_infer_* lines found (not a vidarax metrics dump, or file is empty)");
        println!();
        return;
    }

    println!("  per provider, as scraped:");
    println!(
        "    {:<10} {:>8} {:>8} {:>12} {:>12} {:>12} {:>16} {:>10}",
        "provider",
        "ok",
        "err",
        "prompt_tok",
        "compl_tok",
        "think_tok",
        "latency_sum_ms",
        "mean_ms"
    );
    for (provider, t) in totals {
        print_provider_row(provider, t);
    }
    println!();

    let local = aggregate_providers(totals, &LOCAL_PROVIDERS);
    let gemini = aggregate_providers(totals, &SECOND_PASS_PROVIDERS);
    let total_requests = local.requests().saturating_add(gemini.requests());
    let total_tokens = local.tokens().saturating_add(gemini.tokens());

    // These are per-provider inference totals from /metrics (includes any
    // direct inference on that provider, not only WHIP tiering escalations);
    // see the module doc comment. Grouping vllm+sglang+mlx as "local" and
    // gemini separately is a labeling convenience, not a claim that this
    // isolates tiering activity.
    println!("  per-provider inference totals from /metrics (includes any direct inference):");
    println!("  LOCAL (vllm + sglang + mlx) vs GEMINI:");
    println!("    requests, {total_requests} total:");
    print_split_line("local", local.requests(), total_requests);
    print_split_line("gemini", gemini.requests(), total_requests);
    println!("    tokens, {total_tokens} total:");
    print_split_line("local", local.tokens(), total_tokens);
    print_split_line("gemini", gemini.tokens(), total_tokens);
    println!("    mean latency (from latency_ms_sum / latency_ms_count):");
    match local.mean_latency_ms() {
        Some(mean) => println!("      local        {mean:>8.1} ms"),
        None => println!("      local        no samples"),
    }
    match gemini.mean_latency_ms() {
        Some(mean) => println!("      gemini       {mean:>8.1} ms"),
        None => println!("      gemini       no samples"),
    }
    println!();
}

// ─── Argument sniffing ──────────────────────────────────────────────────────

enum ArgKind {
    Wal,
    Metrics,
    Unknown,
}

fn first_meaningful_line(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    for line in reader.lines() {
        let line = line.ok()?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn looks_like_metrics_line(line: &str) -> bool {
    line.starts_with("vidarax_infer") || line.starts_with('#')
}

/// The WAL on-disk format (`TimelineEvent::encode_line` in timeline.rs) is
/// `seq \t run_id \t stream_id \t pts_ms \t kind \t payload`, i.e. five tabs
/// with a numeric `seq` in the first field. That shape is distinctive enough
/// to sniff without depending on any particular run_id or kind value.
fn looks_like_wal_line(line: &str) -> bool {
    if line.matches('\t').count() < 5 {
        return false;
    }
    line.split('\t')
        .next()
        .is_some_and(|seq| seq.parse::<u64>().is_ok())
}

fn classify_arg(path: &Path) -> ArgKind {
    match first_meaningful_line(path) {
        Some(line) if looks_like_metrics_line(&line) => ArgKind::Metrics,
        Some(line) if looks_like_wal_line(&line) => ArgKind::Wal,
        _ => ArgKind::Unknown,
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  tiering_profile <wal_path> [metrics_path]");
    eprintln!("  tiering_profile <metrics_path>              (metrics-only; auto-detected)");
    eprintln!();
    eprintln!("  <wal_path>     the WAL append log (newline-delimited TimelineEvent lines)");
    eprintln!("  <metrics_path> a saved `curl <server>/metrics > metrics.txt` dump");
    eprintln!();
    eprintln!("Read-only: never appends to the WAL, never makes a network call.");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        std::process::exit(1);
    }

    let mut wal_path: Option<PathBuf> = None;
    let mut metrics_path: Option<PathBuf> = None;
    let mut unclassified: Vec<PathBuf> = Vec::new();

    for raw in args.iter().take(2) {
        let path = PathBuf::from(raw);
        match classify_arg(&path) {
            ArgKind::Metrics if metrics_path.is_none() => metrics_path = Some(path),
            ArgKind::Wal if wal_path.is_none() => wal_path = Some(path),
            _ => unclassified.push(path),
        }
    }
    // A path that didn't sniff as either (e.g. a freshly created, still-empty
    // WAL file has no lines to sniff) defaults to the WAL slot: that's the
    // primary input this tool exists to read.
    if wal_path.is_none() {
        if let Some(path) = unclassified.into_iter().next() {
            wal_path = Some(path);
        }
    }

    if wal_path.is_none() && metrics_path.is_none() {
        eprintln!("could not find a usable WAL or metrics path in the given arguments");
        print_usage();
        std::process::exit(1);
    }

    if let Some(path) = &wal_path {
        match read_all_events(path) {
            Ok(events) => print_wal_funnel(path, &events),
            Err(err) => eprintln!("failed to read WAL {}: {err}", path.display()),
        }
    }

    if let Some(path) = &metrics_path {
        match std::fs::read_to_string(path) {
            Ok(text) => print_metrics_split(path, &parse_metrics(&text)),
            Err(err) => eprintln!("failed to read metrics file {}: {err}", path.display()),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn event(seq: u64, run_id: &str, kind: &str) -> TimelineEvent {
        TimelineEvent {
            seq,
            run_id: run_id.to_string(),
            stream_id: "stream-1".to_string(),
            pts_ms: seq * 100,
            kind: kind.to_string(),
            payload: "{}".to_string(),
        }
    }

    #[test]
    fn funnel_counts_only_vlm_and_vlm_tiered() {
        let events = vec![
            event(1, "run-a", "vlm"),
            event(2, "run-a", "vlm_tiered"),
            event(3, "run-a", "scene_cut"),
            event(4, "run-b", "vlm"),
            event(5, "run-b", "vlm"),
        ];
        let (overall, per_run) = compute_funnel(&events);
        assert_eq!(overall.local_only, 3);
        assert_eq!(overall.escalated, 1);

        let a = per_run.get("run-a").unwrap();
        assert_eq!(a.local_only, 1);
        assert_eq!(a.escalated, 1);

        let b = per_run.get("run-b").unwrap();
        assert_eq!(b.local_only, 2);
        assert_eq!(b.escalated, 0);
    }

    #[test]
    fn funnel_ignores_runs_with_no_tiering_events() {
        let events = vec![event(1, "run-c", "state_transition")];
        let (overall, per_run) = compute_funnel(&events);
        assert_eq!(overall.inferred(), 0);
        assert!(per_run.is_empty());
    }

    #[test]
    fn parse_metric_line_extracts_name_labels_value() {
        let (name, labels, value) =
            parse_metric_line(r#"vidarax_infer_tokens_total{provider="vllm",kind="prompt"} 120"#)
                .unwrap();
        assert_eq!(name, "vidarax_infer_tokens_total");
        assert_eq!(labels, vec![("provider", "vllm"), ("kind", "prompt")]);
        assert!((value - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_metric_line_skips_comments_and_blank_lines() {
        assert!(parse_metric_line("# HELP something").is_none());
        assert!(parse_metric_line("").is_none());
        assert!(parse_metric_line("   ").is_none());
    }

    #[test]
    fn parse_metrics_aggregates_local_and_second_pass() {
        let text = r#"
vidarax_infer_requests_total{provider="vllm",status="ok"} 10
vidarax_infer_requests_total{provider="vllm",status="error"} 1
vidarax_infer_tokens_total{provider="vllm",kind="prompt"} 500
vidarax_infer_tokens_total{provider="vllm",kind="completion"} 100
vidarax_infer_latency_ms_sum{provider="vllm"} 300
vidarax_infer_latency_ms_count{provider="vllm"} 10
vidarax_infer_requests_total{provider="gemini",status="ok"} 4
vidarax_infer_tokens_total{provider="gemini",kind="prompt"} 800
vidarax_infer_tokens_total{provider="gemini",kind="thinking"} 200
vidarax_infer_latency_ms_sum{provider="gemini"} 2000
vidarax_infer_latency_ms_count{provider="gemini"} 4
"#;
        let totals = parse_metrics(text);
        let vllm = totals.get("vllm").unwrap();
        assert_eq!(vllm.ok, 10);
        assert_eq!(vllm.err, 1);
        assert_eq!(vllm.tokens(), 600);
        assert_eq!(vllm.mean_latency_ms(), Some(30.0));

        let local = aggregate_providers(&totals, &LOCAL_PROVIDERS);
        assert_eq!(local.requests(), 11);
        let second_pass = aggregate_providers(&totals, &SECOND_PASS_PROVIDERS);
        assert_eq!(second_pass.tokens(), 1000);
        assert_eq!(second_pass.mean_latency_ms(), Some(500.0));
    }

    #[test]
    fn wal_and_metrics_lines_sniff_correctly() {
        assert!(looks_like_metrics_line(
            r#"vidarax_infer_requests_total{provider="vllm",status="ok"} 1"#
        ));
        assert!(!looks_like_metrics_line("1\trun-a\tstream-1\t100\tvlm\t{}"));
        assert!(looks_like_wal_line("1\trun-a\tstream-1\t100\tvlm\t{}"));
        assert!(!looks_like_wal_line(
            r#"vidarax_infer_requests_total{provider="vllm",status="ok"} 1"#
        ));
    }

    #[test]
    fn funnel_counts_clip_only_run() {
        // Regression: a WAL from a clip-mode-only run (WHIP clip.rs path,
        // no frame-path vlm/vlm_tiered events at all) used to report an
        // empty funnel because compute_funnel only matched the frame-path
        // kind strings.
        let events = vec![
            event(1, "run-clip", "clip_vlm"),
            event(2, "run-clip", "clip_vlm_tiered"),
            event(3, "run-clip", "clip_vlm_tiered"),
        ];
        let (overall, per_run) = compute_funnel(&events);
        assert_eq!(overall.local_only, 1);
        assert_eq!(overall.escalated, 2);
        assert_eq!(overall.inferred(), 3);

        let run = per_run.get("run-clip").unwrap();
        assert_eq!(run.local_only, 1);
        assert_eq!(run.escalated, 2);
    }

    #[test]
    fn funnel_counts_clip_and_frame_kinds_together() {
        let events = vec![
            event(1, "run-a", "clip_vlm"),
            event(2, "run-a", "vlm"),
            event(3, "run-a", "clip_vlm_tiered"),
            event(4, "run-b", "vlm_tiered"),
        ];
        let (overall, per_run) = compute_funnel(&events);
        assert_eq!(overall.local_only, 2);
        assert_eq!(overall.escalated, 2);

        let a = per_run.get("run-a").unwrap();
        assert_eq!(a.local_only, 2);
        assert_eq!(a.escalated, 1);

        let b = per_run.get("run-b").unwrap();
        assert_eq!(b.local_only, 0);
        assert_eq!(b.escalated, 1);
    }

    #[test]
    fn parse_metric_line_rejects_non_finite_and_negative_values() {
        assert!(parse_metric_line(
            r#"vidarax_infer_requests_total{provider="vllm",status="ok"} nan"#
        )
        .is_none());
        assert!(parse_metric_line(
            r#"vidarax_infer_requests_total{provider="vllm",status="ok"} inf"#
        )
        .is_none());
        assert!(parse_metric_line(
            r#"vidarax_infer_requests_total{provider="vllm",status="ok"} -1"#
        )
        .is_none());
    }

    #[test]
    fn parse_metrics_skips_negative_and_non_finite_lines() {
        let text = r#"
vidarax_infer_requests_total{provider="vllm",status="ok"} -5
vidarax_infer_requests_total{provider="vllm",status="error"} nan
"#;
        let totals = parse_metrics(text);
        // Both lines were rejected outright, so no entry is created for
        // "vllm" at all (as opposed to an entry with garbage zeroed fields).
        assert!(totals.get("vllm").is_none());
    }

    #[test]
    fn parse_metrics_saturates_oversized_value_without_panic() {
        // Regression: an oversized value (e.g. a corrupted or adversarial
        // /metrics dump reporting 1e30 requests) used to saturate to
        // u64::MAX on the way in, and ProviderTotals::requests() computing
        // `ok + err` on two such saturated fields would then overflow and
        // panic in a debug build. This must return a saturated, sane
        // result instead of panicking.
        let text = r#"
vidarax_infer_requests_total{provider="vllm",status="ok"} 1e30
vidarax_infer_requests_total{provider="vllm",status="error"} 1e30
"#;
        let totals = parse_metrics(text);
        let vllm = totals.get("vllm").unwrap();
        assert_eq!(vllm.ok, u64::MAX);
        assert_eq!(vllm.err, u64::MAX);
        assert_eq!(vllm.requests(), u64::MAX);

        // aggregate_providers runs the same values through ProviderTotals::add;
        // that must saturate too instead of overflowing.
        let local = aggregate_providers(&totals, &LOCAL_PROVIDERS);
        assert_eq!(local.requests(), u64::MAX);
    }
}
