use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use vidarax_core::admission::InferenceAdmission;
use vidarax_core::metrics::PipelineMetrics;
use vidarax_core::provider::{InferenceObserver, ProviderKind, TokenUsage};

const LATENCY_BUCKETS_MS: [u64; 14] = [
    10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000, 30000, 60000,
];

pub struct InferenceMetrics {
    vllm: ProviderMetrics,
    sglang: ProviderMetrics,
    gemini: ProviderMetrics,
    mlx: ProviderMetrics,
}

impl InferenceMetrics {
    pub fn new() -> Self {
        Self {
            vllm: ProviderMetrics::new(),
            sglang: ProviderMetrics::new(),
            gemini: ProviderMetrics::new(),
            mlx: ProviderMetrics::new(),
        }
    }

    pub fn record_success(
        &self,
        provider: ProviderKind,
        latency_ms: u64,
        fallback_used: bool,
        usage: TokenUsage,
    ) {
        let metrics = self.provider(provider);
        metrics.success_total.fetch_add(1, Ordering::Relaxed);
        if fallback_used {
            metrics.fallback_total.fetch_add(1, Ordering::Relaxed);
        }
        metrics
            .prompt_tokens_total
            .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
        metrics
            .completion_tokens_total
            .fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
        metrics
            .thinking_tokens_total
            .fetch_add(usage.thinking_tokens as u64, Ordering::Relaxed);
        metrics.record_latency(latency_ms);
    }

    pub fn record_error(&self, provider: ProviderKind, latency_ms: u64) {
        let metrics = self.provider(provider);
        metrics.error_total.fetch_add(1, Ordering::Relaxed);
        metrics.record_latency(latency_ms);
    }

    /// Returns `true` when either provider shows p95 latency > 5 000 ms.
    ///
    /// Used by `GET /v1/models` to report `"saturated"` availability status
    /// when inference providers are reachable but overloaded.
    pub fn is_high_latency(&self) -> bool {
        self.vllm.is_high_latency()
            || self.sglang.is_high_latency()
            || self.gemini.is_high_latency()
            || self.mlx.is_high_latency()
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        self.render_provider("vllm", &self.vllm, &mut out);
        self.render_provider("sglang", &self.sglang, &mut out);
        self.render_provider("gemini", &self.gemini, &mut out);
        self.render_provider("mlx", &self.mlx, &mut out);
        out
    }

    pub fn render_admission_prometheus(admission: &InferenceAdmission) -> String {
        use std::fmt::Write as _;

        let snapshot = admission.snapshot();
        let mut out = String::new();
        let _ = writeln!(out, "vidarax_infer_admission_active {}", snapshot.active);
        let _ = writeln!(out, "vidarax_infer_admission_waiting {}", snapshot.waiting);
        let _ = writeln!(
            out,
            "vidarax_infer_admission_acquired_total {}",
            snapshot.acquired_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_timeouts_total{{limit=\"global\"}} {}",
            snapshot.timeout_global_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_timeouts_total{{limit=\"principal\"}} {}",
            snapshot.timeout_principal_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_waiter_rejections_total {}",
            snapshot.waiter_rejected_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_wait_duration_us_sum {}",
            snapshot.wait_duration_us
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_wait_duration_us_count {}",
            snapshot.wait_duration_count
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_active_tokens {}",
            snapshot.active_tokens
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_active_bytes {}",
            snapshot.active_bytes
        );
        let limits = admission.limits();
        let _ = writeln!(
            out,
            "vidarax_infer_admission_token_limit {}",
            limits.max_in_flight_tokens
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_byte_limit {}",
            limits.max_in_flight_bytes
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_deadline_missed_total {}",
            snapshot.deadline_missed_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_budget_rejections_total {}",
            snapshot.budget_rejected_total
        );
        for (class, count) in [
            ("urgent_live", snapshot.urgent_acquired_total),
            ("live", snapshot.live_acquired_total),
            ("offline", snapshot.offline_acquired_total),
        ] {
            let _ = writeln!(
                out,
                "vidarax_infer_admission_acquired_by_class_total{{class=\"{class}\"}} {count}"
            );
        }
        let _ = writeln!(
            out,
            "vidarax_infer_admission_urgent_live_acquired_total {}",
            snapshot.urgent_acquired_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_live_acquired_total {}",
            snapshot.live_acquired_total
        );
        let _ = writeln!(
            out,
            "vidarax_infer_admission_offline_acquired_total {}",
            snapshot.offline_acquired_total
        );
        out
    }

    fn render_provider(&self, name: &str, p: &ProviderMetrics, out: &mut String) {
        use std::fmt::Write as _;
        let ok = p.success_total.load(Ordering::Relaxed);
        let err = p.error_total.load(Ordering::Relaxed);
        let fallback = p.fallback_total.load(Ordering::Relaxed);
        let prompt_tokens = p.prompt_tokens_total.load(Ordering::Relaxed);
        let completion_tokens = p.completion_tokens_total.load(Ordering::Relaxed);
        let thinking_tokens = p.thinking_tokens_total.load(Ordering::Relaxed);
        let sum_ms = p.latency_sum_ms.load(Ordering::Relaxed);
        let count = p.latency_count.load(Ordering::Relaxed);

        let _ = writeln!(
            out,
            "vidarax_infer_requests_total{{provider=\"{name}\",status=\"ok\"}} {ok}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_requests_total{{provider=\"{name}\",status=\"error\"}} {err}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_fallback_total{{provider=\"{name}\"}} {fallback}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_tokens_total{{provider=\"{name}\",kind=\"prompt\"}} {prompt_tokens}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_tokens_total{{provider=\"{name}\",kind=\"completion\"}} {completion_tokens}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_tokens_total{{provider=\"{name}\",kind=\"thinking\"}} {thinking_tokens}"
        );

        let mut cumulative = 0u64;
        for (idx, le) in LATENCY_BUCKETS_MS.iter().enumerate() {
            cumulative += p.latency_buckets[idx].load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "vidarax_infer_latency_ms_bucket{{provider=\"{name}\",le=\"{le}\"}} {cumulative}"
            );
        }
        let _ = writeln!(
            out,
            "vidarax_infer_latency_ms_bucket{{provider=\"{name}\",le=\"+Inf\"}} {count}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_latency_ms_sum{{provider=\"{name}\"}} {sum_ms}"
        );
        let _ = writeln!(
            out,
            "vidarax_infer_latency_ms_count{{provider=\"{name}\"}} {count}"
        );

        // SLO tracking baselines for dashboards/alerts.
        let _ = writeln!(
            out,
            "vidarax_infer_slo_target_ratio{{provider=\"{name}\"}} 0.99"
        );
        let total = ok + err;
        let error_budget_remaining = if total == 0 {
            1.0
        } else {
            let error_rate = err as f64 / total as f64;
            (0.01_f64 - error_rate).max(0.0) / 0.01_f64
        };
        let _ = writeln!(out, "vidarax_infer_error_budget_remaining_ratio{{provider=\"{name}\"}} {error_budget_remaining:.6}");
    }

    fn provider(&self, provider: ProviderKind) -> &ProviderMetrics {
        match provider {
            ProviderKind::Vllm => &self.vllm,
            ProviderKind::Sglang => &self.sglang,
            ProviderKind::Gemini => &self.gemini,
            ProviderKind::Mlx => &self.mlx,
        }
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Lets `vidarax-core` (the WHIP VLM workers, clip workers, and tiered
/// inference router) record into `/metrics` without depending on
/// `vidarax-api`. Delegates straight to the inherent methods above, which
/// already have matching signatures.
impl InferenceObserver for InferenceMetrics {
    fn record_success(
        &self,
        provider: ProviderKind,
        latency_ms: u64,
        fallback_used: bool,
        usage: TokenUsage,
    ) {
        InferenceMetrics::record_success(self, provider, latency_ms, fallback_used, usage)
    }

    fn record_error(&self, provider: ProviderKind, latency_ms: u64) {
        InferenceMetrics::record_error(self, provider, latency_ms)
    }
}

/// Records provider detail and the provider-agnostic pipeline totals from the
/// same inference outcome. The recorded-file reasoning path uses this observer
/// so its VLM stage is visible in the same dashboard as live WHIP traffic.
pub struct PipelineInferenceObserver {
    inference: Arc<InferenceMetrics>,
    pipeline: Arc<PipelineMetrics>,
}

impl PipelineInferenceObserver {
    pub fn new(inference: Arc<InferenceMetrics>, pipeline: Arc<PipelineMetrics>) -> Self {
        Self {
            inference,
            pipeline,
        }
    }

    fn record_pipeline_outcome(&self, latency_ms: u64) {
        self.pipeline.inc_vlm_inferences();
        self.pipeline.vlm_latency_ms.record(latency_ms);
    }
}

impl InferenceObserver for PipelineInferenceObserver {
    fn record_success(
        &self,
        provider: ProviderKind,
        latency_ms: u64,
        fallback_used: bool,
        usage: TokenUsage,
    ) {
        self.inference
            .record_success(provider, latency_ms, fallback_used, usage);
        self.record_pipeline_outcome(latency_ms);
    }

    fn record_error(&self, provider: ProviderKind, latency_ms: u64) {
        self.inference.record_error(provider, latency_ms);
        self.record_pipeline_outcome(latency_ms);
    }
}

struct ProviderMetrics {
    success_total: AtomicU64,
    error_total: AtomicU64,
    fallback_total: AtomicU64,
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,
    thinking_tokens_total: AtomicU64,
    latency_sum_ms: AtomicU64,
    latency_count: AtomicU64,
    latency_buckets: [AtomicU64; LATENCY_BUCKETS_MS.len()],
}

impl ProviderMetrics {
    fn new() -> Self {
        Self {
            success_total: AtomicU64::new(0),
            error_total: AtomicU64::new(0),
            fallback_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            completion_tokens_total: AtomicU64::new(0),
            thinking_tokens_total: AtomicU64::new(0),
            latency_sum_ms: AtomicU64::new(0),
            latency_count: AtomicU64::new(0),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// p95 > 5 000 ms: fewer than 95% of requests fit within the 5 000 ms bucket.
    fn is_high_latency(&self) -> bool {
        let total = self.latency_count.load(Ordering::Relaxed);
        if total < 10 {
            return false; // too few samples
        }
        // LATENCY_BUCKETS_MS includes 10 ms through 60 s.
        // Index 8 (inclusive) gives cumulative count ≤ 5 000 ms.
        let within_5s: u64 = self
            .latency_buckets
            .iter()
            .take(9)
            .map(|b| b.load(Ordering::Relaxed))
            .sum();
        // saturated when fewer than 95% of requests complete within 5 s
        within_5s.saturating_mul(100) < total.saturating_mul(95)
    }

    fn record_latency(&self, latency_ms: u64) {
        self.latency_sum_ms.fetch_add(latency_ms, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
        for (idx, bucket) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if latency_ms <= *bucket {
                self.latency_buckets[idx].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::InferenceMetrics;
    use vidarax_core::admission::{AdmissionLimits, InferenceAdmission};
    use vidarax_core::provider::{ProviderKind, TokenUsage};

    #[test]
    fn renders_provider_metrics() {
        let metrics = InferenceMetrics::new();
        metrics.record_success(
            ProviderKind::Vllm,
            30,
            false,
            TokenUsage {
                prompt_tokens: 120,
                completion_tokens: 45,
                thinking_tokens: 0,
                total_tokens: 165,
            },
        );
        metrics.record_error(ProviderKind::Vllm, 300);
        let text = metrics.render_prometheus();
        assert!(text.contains("vidarax_infer_requests_total{provider=\"vllm\",status=\"ok\"} 1"));
        assert!(text.contains("vidarax_infer_requests_total{provider=\"vllm\",status=\"error\"} 1"));
        assert!(text.contains("vidarax_infer_latency_ms_count{provider=\"vllm\"} 2"));
        assert!(text.contains("vidarax_infer_tokens_total{provider=\"vllm\",kind=\"prompt\"} 120"));
        assert!(
            text.contains("vidarax_infer_tokens_total{provider=\"vllm\",kind=\"completion\"} 45")
        );
    }

    #[test]
    fn renders_admission_metrics_without_principal_labels() {
        let admission = InferenceAdmission::new(AdmissionLimits {
            global_in_flight: 2,
            per_principal_in_flight: 1,
            global_waiters: 2,
            wait_timeout: Duration::from_millis(5),
            max_in_flight_tokens: 1_000_000,
            max_in_flight_bytes: 1024 * 1024 * 1024,
        })
        .unwrap();
        let _permit = admission.acquire("secret-tenant-name").unwrap();
        let _ = admission.acquire("secret-tenant-name");

        let text = InferenceMetrics::render_admission_prometheus(&admission);

        assert!(text.contains("vidarax_infer_admission_active 1"));
        assert!(text.contains("vidarax_infer_admission_timeouts_total{limit=\"principal\"} 1"));
        assert!(text.contains("vidarax_infer_admission_wait_duration_us_count 2"));
        assert!(!text.contains("secret-tenant-name"));
    }

    #[test]
    fn renders_mlx_metrics_under_its_own_label_distinct_from_vllm() {
        // mlx is a distinct ProviderKind (on-device mlx-vlm), so its counters
        // must land under the "mlx" series, not fold into "vllm".
        let metrics = InferenceMetrics::new();
        metrics.record_success(
            ProviderKind::Mlx,
            15,
            false,
            TokenUsage {
                prompt_tokens: 80,
                completion_tokens: 20,
                thinking_tokens: 0,
                total_tokens: 100,
            },
        );
        let text = metrics.render_prometheus();
        assert!(text.contains("vidarax_infer_requests_total{provider=\"mlx\",status=\"ok\"} 1"));
        assert!(text.contains("vidarax_infer_tokens_total{provider=\"mlx\",kind=\"prompt\"} 80"));
        assert!(text.contains("vidarax_infer_requests_total{provider=\"vllm\",status=\"ok\"} 0"));
    }

    #[test]
    fn accumulates_tokens_across_calls() {
        let metrics = InferenceMetrics::new();
        for _ in 0..3 {
            metrics.record_success(
                ProviderKind::Gemini,
                10,
                false,
                TokenUsage {
                    prompt_tokens: 100,
                    completion_tokens: 10,
                    thinking_tokens: 50,
                    total_tokens: 160,
                },
            );
        }
        let text = metrics.render_prometheus();
        assert!(
            text.contains("vidarax_infer_tokens_total{provider=\"gemini\",kind=\"prompt\"} 300")
        );
        assert!(
            text.contains("vidarax_infer_tokens_total{provider=\"gemini\",kind=\"thinking\"} 150")
        );
    }
}
