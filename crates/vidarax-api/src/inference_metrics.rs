use std::sync::atomic::{AtomicU64, Ordering};

use vidarax_core::provider::{InferenceObserver, ProviderKind, TokenUsage};

const LATENCY_BUCKETS_MS: [u64; 10] = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

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
        // LATENCY_BUCKETS_MS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
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
    use super::InferenceMetrics;
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
