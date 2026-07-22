//! Tiering funnel probe.
//!
//! Drives the real `run_tiered` router over a synthetic keyframe stream and
//! prints the three-tier funnel: how many keyframes are served from the local
//! cache (tier 1), answered by the local first-pass model alone (tier 2), or
//! escalated to the accurate second-pass model (tier 3). It also reports the
//! token split so you can see what fraction of spend lands on the expensive
//! second-pass provider at a given confidence threshold.
//!
//! Nothing here touches the network. The provider is a scripted stub, so the
//! numbers describe the *shape* of the routing, not measured provider cost. Per
//! call token and latency figures are configurable constants below; for real
//! spend read the per-provider counters at `/metrics` (vidarax_infer_*), and for
//! the real per-run funnel read the WAL event kinds (`vlm`/`vlm_tiered`, or
//! `clip_vlm`/`clip_vlm_tiered` in clip mode); tier-1 cache hits are log-only
//! (`tracing`), not WAL events. This probe is for reasoning about the routing policy: how
//! the confidence threshold trades local coverage against second-pass cost.
//!
//! Usage:
//!   cargo run -p vidarax-core --example tiering_funnel -- [keyframes] [threshold] [cache_hit_pct]
//!
//! Defaults: 1000 keyframes, threshold 0.70, 30% tier-1 cache hit rate.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use vidarax_core::provider::{
    InferenceProvider, InferenceRequest, InferenceResult, ProviderError, ProviderKind, TokenUsage,
};
use vidarax_core::tiered_vlm::{run_tiered, TieredVlmConfig};

// Illustrative per-call cost. These are stand-ins for the shape of the two
// tiers (the second pass is heavier and reports hidden thinking tokens), not
// measured values. Swap in your own numbers, or read the real split at
// /metrics, to model actual spend.
const LOCAL_PROMPT_TOKENS: u32 = 320;
const LOCAL_COMPLETION_TOKENS: u32 = 48;
const LOCAL_LATENCY_MS: u64 = 40;

const SECOND_PROMPT_TOKENS: u32 = 780;
const SECOND_COMPLETION_TOKENS: u32 = 96;
const SECOND_THINKING_TOKENS: u32 = 210;
const SECOND_LATENCY_MS: u64 = 620;

/// A scripted inference provider that answers the local first-pass model with a
/// confidence the harness sets per keyframe, and answers the second-pass model
/// with a confident teacher label. It is the single point every real inference
/// call passes through, so it tallies the local vs second-pass token and latency
/// spend directly.
struct ScriptedProvider {
    local_model: Arc<str>,
    second_model: Arc<str>,
    /// Confidence the next local first-pass call should report, as f32 bits.
    next_conf_bits: AtomicU32,
    /// When set, the next local first-pass call emits output with no confidence
    /// field, so the router falls back to the parse default and escalates. This
    /// models the real path where a local model returns malformed output.
    next_malformed: AtomicBool,

    local_calls: AtomicU64,
    local_prompt_tokens: AtomicU64,
    local_completion_tokens: AtomicU64,
    local_latency_ms: AtomicU64,

    second_calls: AtomicU64,
    second_prompt_tokens: AtomicU64,
    second_completion_tokens: AtomicU64,
    second_thinking_tokens: AtomicU64,
    second_latency_ms: AtomicU64,
}

impl ScriptedProvider {
    fn new(local_model: Arc<str>, second_model: Arc<str>) -> Self {
        Self {
            local_model,
            second_model,
            next_conf_bits: AtomicU32::new(0.0f32.to_bits()),
            next_malformed: AtomicBool::new(false),
            local_calls: AtomicU64::new(0),
            local_prompt_tokens: AtomicU64::new(0),
            local_completion_tokens: AtomicU64::new(0),
            local_latency_ms: AtomicU64::new(0),
            second_calls: AtomicU64::new(0),
            second_prompt_tokens: AtomicU64::new(0),
            second_completion_tokens: AtomicU64::new(0),
            second_thinking_tokens: AtomicU64::new(0),
            second_latency_ms: AtomicU64::new(0),
        }
    }

    fn set_next_confidence(&self, conf: f32) {
        self.next_conf_bits.store(conf.to_bits(), Ordering::Relaxed);
    }

    fn set_next_malformed(&self, malformed: bool) {
        self.next_malformed.store(malformed, Ordering::Relaxed);
    }
}

impl InferenceProvider for ScriptedProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Vllm
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
        // The router rewrites request.model to the tier it is calling, so the
        // model id tells us which side of the funnel this call is.
        if request.model == self.second_model {
            self.second_calls.fetch_add(1, Ordering::Relaxed);
            self.second_prompt_tokens
                .fetch_add(SECOND_PROMPT_TOKENS as u64, Ordering::Relaxed);
            self.second_completion_tokens
                .fetch_add(SECOND_COMPLETION_TOKENS as u64, Ordering::Relaxed);
            self.second_thinking_tokens
                .fetch_add(SECOND_THINKING_TOKENS as u64, Ordering::Relaxed);
            self.second_latency_ms
                .fetch_add(SECOND_LATENCY_MS, Ordering::Relaxed);

            let total = SECOND_PROMPT_TOKENS + SECOND_COMPLETION_TOKENS + SECOND_THINKING_TOKENS;
            return Ok(InferenceResult {
                provider: ProviderKind::Gemini,
                model: Arc::clone(&self.second_model),
                output_text: r#"{"event_type":"state","confidence":0.95,"description":"teacher"}"#
                    .to_string(),
                fallback_used: false,
                finish_reason: Some("stop".to_string()),
                inference_latency_ms: SECOND_LATENCY_MS,
                usage: TokenUsage {
                    prompt_tokens: SECOND_PROMPT_TOKENS,
                    completion_tokens: SECOND_COMPLETION_TOKENS,
                    thinking_tokens: SECOND_THINKING_TOKENS,
                    total_tokens: total,
                },
            });
        }

        // Local first pass: report the confidence the harness scripted.
        let conf = f32::from_bits(self.next_conf_bits.load(Ordering::Relaxed));
        self.local_calls.fetch_add(1, Ordering::Relaxed);
        self.local_prompt_tokens
            .fetch_add(LOCAL_PROMPT_TOKENS as u64, Ordering::Relaxed);
        self.local_completion_tokens
            .fetch_add(LOCAL_COMPLETION_TOKENS as u64, Ordering::Relaxed);
        self.local_latency_ms
            .fetch_add(LOCAL_LATENCY_MS, Ordering::Relaxed);

        let malformed = self.next_malformed.load(Ordering::Relaxed);
        let output_text = if malformed {
            // No confidence field: the router's parser falls back to its
            // default, which sits below the usual threshold and escalates.
            r#"{"event_type":"state","description":"local"}"#.to_string()
        } else {
            format!(
                "{{\"event_type\":\"state\",\"confidence\":{conf:.4},\"description\":\"local\"}}"
            )
        };
        let total = LOCAL_PROMPT_TOKENS + LOCAL_COMPLETION_TOKENS;
        Ok(InferenceResult {
            provider: ProviderKind::Vllm,
            model: Arc::clone(&self.local_model),
            output_text,
            fallback_used: false,
            finish_reason: Some("stop".to_string()),
            inference_latency_ms: LOCAL_LATENCY_MS,
            usage: TokenUsage {
                prompt_tokens: LOCAL_PROMPT_TOKENS,
                completion_tokens: LOCAL_COMPLETION_TOKENS,
                thinking_tokens: 0,
                total_tokens: total,
            },
        })
    }
}

/// Small deterministic LCG so the run is reproducible without pulling in a rng.
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f32 {
        // Numerical Recipes constants; take the high bits for a value in [0, 1).
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn base_request(prompt: &str) -> InferenceRequest {
    InferenceRequest {
        model: Arc::from("placeholder"),
        prompt: Arc::from(prompt),
        input_images: Vec::new(),
        input_videos: Vec::new(),
        max_tokens: 256,
        temperature: 0.0,
        timeout_ms: 10_000,
        allow_fallback: false,
        guided_json: None,
        scheduling: Default::default(),
    }
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (whole as f64)
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let keyframes: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let threshold: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.70);
    let cache_hit_pct: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    // Share of local outputs that arrive with no parseable confidence field.
    // These escalate through the parse fallback rather than a real low score.
    let malformed_pct: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let local_model: Arc<str> = Arc::from("local-first-pass");
    let second_model: Arc<str> = Arc::from("accurate-second-pass");

    let config = TieredVlmConfig {
        first_pass_model: Arc::clone(&local_model),
        second_pass_model: Arc::clone(&second_model),
        second_pass_threshold: threshold,
        second_pass_max_tokens: 512,
    };
    assert!(
        config.is_tiered(),
        "funnel needs distinct models or the second pass never runs"
    );

    let provider = ScriptedProvider::new(Arc::clone(&local_model), Arc::clone(&second_model));

    let mut conf_rng = Lcg(0x9E3779B97F4A7C15);
    let mut cache_rng = Lcg(0xD1B54A32D192ED03);
    let mut malformed_rng = Lcg(0x2545F4914F6CDD1D);

    let mut tier1_cache = 0u64;
    let mut tier2_local_only = 0u64;
    let mut tier3_escalated = 0u64;
    let mut escalated_low_conf = 0u64;
    let mut escalated_malformed = 0u64;

    for i in 0..keyframes {
        // Tier 1: local cache answer, modeled at the harness level because the
        // real kNN cache lives in the vlm worker ahead of run_tiered.
        let cache_roll = (cache_rng.next_unit() * 100.0) as u64;
        if cache_roll < cache_hit_pct {
            tier1_cache += 1;
            continue;
        }

        let conf = conf_rng.next_unit();
        let malformed = ((malformed_rng.next_unit() * 100.0) as u64) < malformed_pct;
        provider.set_next_confidence(conf);
        provider.set_next_malformed(malformed);

        let run = run_tiered(
            &provider,
            &config,
            base_request(&format!("kf-{i}")),
            1024,
            10_000,
            None,
        )
        .expect("scripted provider never errors");

        if run.used_second_pass {
            tier3_escalated += 1;
            if malformed {
                escalated_malformed += 1;
            } else {
                escalated_low_conf += 1;
            }
        } else {
            tier2_local_only += 1;
        }
    }

    let local_tokens = provider.local_prompt_tokens.load(Ordering::Relaxed)
        + provider.local_completion_tokens.load(Ordering::Relaxed);
    let second_tokens = provider.second_prompt_tokens.load(Ordering::Relaxed)
        + provider.second_completion_tokens.load(Ordering::Relaxed)
        + provider.second_thinking_tokens.load(Ordering::Relaxed);
    let total_tokens = local_tokens + second_tokens;

    let local_latency = provider.local_latency_ms.load(Ordering::Relaxed);
    let second_latency = provider.second_latency_ms.load(Ordering::Relaxed);

    let inferred = tier2_local_only + tier3_escalated;

    println!("tiering funnel  (synthetic routing, no network)");
    println!("  keyframes         {keyframes}");
    println!("  threshold         {threshold:.2}  (escalate when local confidence < threshold)");
    println!("  cache hit target  {cache_hit_pct}%");
    println!("  malformed target  {malformed_pct}%  (local output with no parseable confidence)");
    println!();
    println!(
        "  tier 1  cache      {tier1_cache:>7}   {:>5.1}%  of keyframes (0 inference)",
        pct(tier1_cache, keyframes)
    );
    println!(
        "  tier 2  local only {tier2_local_only:>7}   {:>5.1}%  of keyframes",
        pct(tier2_local_only, keyframes)
    );
    println!(
        "  tier 3  escalated  {tier3_escalated:>7}   {:>5.1}%  of keyframes",
        pct(tier3_escalated, keyframes)
    );
    println!(
        "  escalation rate    {:>5.1}%  of inferred keyframes reached the second pass",
        pct(tier3_escalated, inferred)
    );
    println!(
        "    from low conf    {escalated_low_conf:>7}   {:>5.1}%  of escalations (real low score)",
        pct(escalated_low_conf, tier3_escalated)
    );
    println!(
        "    from malformed   {escalated_malformed:>7}   {:>5.1}%  of escalations (parse fallback)",
        pct(escalated_malformed, tier3_escalated)
    );
    println!();
    println!(
        "  local calls        {:>7}",
        provider.local_calls.load(Ordering::Relaxed)
    );
    println!(
        "  second calls       {:>7}",
        provider.second_calls.load(Ordering::Relaxed)
    );
    println!();
    println!(
        "  tokens local       {local_tokens:>10}   {:>5.1}%",
        pct(local_tokens, total_tokens)
    );
    println!(
        "  tokens second pass {second_tokens:>10}   {:>5.1}%",
        pct(second_tokens, total_tokens)
    );
    println!("  tokens total       {total_tokens:>10}");
    println!();
    println!("  latency local sum  {local_latency:>10} ms");
    println!("  latency second sum {second_latency:>10} ms");
    println!();
    println!(
        "  note: the second pass carries {:.0}% of keyframes but {:.0}% of tokens.",
        pct(tier3_escalated, inferred),
        pct(second_tokens, total_tokens)
    );
    println!("  note: real routing also escalates when the local output has no parseable");
    println!("        confidence field (parse fallback is 0.5, below the usual 0.7).");
}
