use std::sync::Arc;

use crate::provider::{
    teacher_label_schema, InferenceObserver, InferenceProvider, InferenceRequest, InferenceResult,
    ProviderError,
};

/// Configuration for tiered VLM inference routing.
///
/// First-pass model handles the common case. The second-pass model only runs
/// when first-pass confidence is below `second_pass_threshold`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use vidarax_core::tiered_vlm::TieredVlmConfig;
///
/// let config = TieredVlmConfig {
///     first_pass_model: Arc::from("Qwen/Qwen3-VL-2B-Instruct"),
///     second_pass_model: Arc::from("Qwen/Qwen3-VL-8B-Instruct"),
///     second_pass_threshold: 0.7,
///     second_pass_max_tokens: 256,
/// };
/// assert!(config.is_tiered());
/// assert!(config.needs_second_pass(0.5));
/// assert!(!config.needs_second_pass(0.8));
/// ```
#[derive(Debug, Clone)]
pub struct TieredVlmConfig {
    /// Model ID for first-pass inference.
    pub first_pass_model: Arc<str>,
    /// Model ID for second-pass inference.
    pub second_pass_model: Arc<str>,
    /// Confidence threshold: if first-pass confidence < this, run second-pass.
    /// Range: 0.0 to 1.0. Default: 0.7.
    pub second_pass_threshold: f32,
    /// Max tokens for second-pass output.
    pub second_pass_max_tokens: u32,
}

impl TieredVlmConfig {
    /// Create a config that uses the same model for both passes (no tiering).
    pub fn single_model(model: &str) -> Self {
        Self {
            first_pass_model: Arc::from(model),
            second_pass_model: Arc::from(model),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 256,
        }
    }

    /// Returns true if the models differ (tiered routing is active).
    pub fn is_tiered(&self) -> bool {
        self.first_pass_model != self.second_pass_model
    }

    /// Returns true if the given confidence score warrants a second-pass.
    ///
    /// Only triggers when tiered routing is active AND confidence is strictly
    /// below threshold.
    pub fn needs_second_pass(&self, first_pass_confidence: f32) -> bool {
        self.is_tiered() && first_pass_confidence < self.second_pass_threshold
    }
}

impl Default for TieredVlmConfig {
    fn default() -> Self {
        Self {
            first_pass_model: Arc::from("Qwen/Qwen3-VL-8B-Instruct"),
            second_pass_model: Arc::from("Qwen/Qwen3-VL-8B-Instruct"),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 256,
        }
    }
}

#[derive(Debug)]
pub struct TieredVlmRun {
    pub result: InferenceResult,
    pub used_second_pass: bool,
    pub request: InferenceRequest,
}

#[derive(Debug)]
pub struct TieredVlmError {
    pub error: ProviderError,
    pub request: InferenceRequest,
}

// Both Result variants carry InferenceRequest; boxing the error adds a cold-path allocation without shrinking the Result.
#[allow(clippy::result_large_err)]
pub fn run_tiered<I>(
    provider: &I,
    config: &TieredVlmConfig,
    mut request: InferenceRequest,
    guided_json_first_max_tokens: u32,
    second_pass_timeout_ms: u64,
    observer: Option<&dyn InferenceObserver>,
) -> Result<TieredVlmRun, TieredVlmError>
where
    I: InferenceProvider + ?Sized,
{
    request.model = Arc::clone(&config.first_pass_model);
    if request.guided_json.is_some() {
        request.max_tokens = guided_json_first_max_tokens;
    }
    let first_result = match provider.infer(&request) {
        Ok(result) => result,
        Err(error) => return Err(TieredVlmError { error, request }),
    };
    if let Some(o) = observer {
        o.record_success(
            first_result.provider,
            first_result.inference_latency_ms,
            first_result.fallback_used,
            first_result.usage,
        );
    }

    let first_conf = parse_confidence_from_output(&first_result.output_text);
    if config.needs_second_pass(first_conf) {
        request.model = Arc::clone(&config.second_pass_model);
        request.max_tokens = config.second_pass_max_tokens;
        request.timeout_ms = second_pass_timeout_ms;
        request.guided_json = Some(Arc::from(teacher_label_schema()));
        match provider.infer(&request) {
            Ok(mut result) => {
                // Record the second pass under its own provider with its own
                // latency and token usage, before the accumulate() below folds
                // the first pass's spend into it. Recording after accumulate
                // would bill the local first-pass tokens to whichever provider
                // served the second pass.
                if let Some(o) = observer {
                    o.record_success(
                        result.provider,
                        result.inference_latency_ms,
                        result.fallback_used,
                        result.usage,
                    );
                }
                // e2e token/latency accounting: the analysis spanned both
                // passes, so fold the first pass's spend into what we surface.
                result.usage.accumulate(first_result.usage);
                result.inference_latency_ms = result
                    .inference_latency_ms
                    .saturating_add(first_result.inference_latency_ms);
                return Ok(TieredVlmRun {
                    result,
                    used_second_pass: true,
                    request,
                });
            }
            Err(_) => {
                // The first pass was already recorded above; a failed second
                // pass has no successful outcome of its own to attribute, and
                // the caller only sees the first-pass result here, so nothing
                // further is recorded.
                return Ok(TieredVlmRun {
                    result: first_result,
                    used_second_pass: false,
                    request,
                });
            }
        }
    }

    Ok(TieredVlmRun {
        result: first_result,
        used_second_pass: false,
        request,
    })
}

pub fn parse_confidence_from_output(text: &str) -> f32 {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(conf) = val.get("confidence").and_then(|v| v.as_f64()) {
            let conf = conf as f32;
            // A garbage local output that somehow produces NaN or an infinite
            // value must not silently defeat the escalation check: NaN compares
            // false against any threshold, so `needs_second_pass` would treat it
            // as confident enough to stay local. Fall back to the same 0.5 used
            // for a missing field so it lands on the "unknown, escalate at the
            // default 0.7 threshold" path instead.
            if !conf.is_finite() {
                return 0.5;
            }
            return conf.clamp(0.0, 1.0);
        }
    }
    0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{
        teacher_label_schema, InferenceObserver, InferenceProvider, InferenceRequest,
        InferenceResult, ProviderError, ProviderKind, TokenUsage,
    };
    use std::sync::Mutex;

    struct RecordingProvider {
        requests: Mutex<Vec<InferenceRequest>>,
    }

    impl RecordingProvider {
        fn requests(&self) -> Vec<InferenceRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl InferenceProvider for RecordingProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Vllm
        }

        fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request.clone());
            let output_text = if requests.len() == 1 {
                r#"{"confidence":0.2,"description":"first"}"#
            } else {
                r#"{"confidence":0.9,"description":"second"}"#
            };
            Ok(InferenceResult {
                provider: ProviderKind::Vllm,
                model: request.model.clone(),
                output_text: output_text.to_string(),
                fallback_used: false,
                finish_reason: None,
                inference_latency_ms: 0,
                usage: TokenUsage::default(),
            })
        }
    }

    #[test]
    fn tiered_run_forces_teacher_schema_on_second_pass() {
        let provider = RecordingProvider {
            requests: Mutex::new(Vec::new()),
        };
        let config = TieredVlmConfig {
            first_pass_model: Arc::from("small"),
            second_pass_model: Arc::from("teacher"),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 512,
        };
        let schema: Arc<str> = Arc::from(r#"{"type":"object","properties":{"custom":{}}}"#);
        let request = InferenceRequest {
            model: Arc::from("placeholder"),
            prompt: Arc::from("prompt"),
            input_images: Vec::new(),
            input_videos: Vec::new(),
            max_tokens: 128,
            temperature: 0.0,
            timeout_ms: 1000,
            allow_fallback: true,
            guided_json: Some(Arc::clone(&schema)),
        };

        let output = run_tiered(&provider, &config, request, 1024, 1000, None).unwrap();

        assert!(output.used_second_pass);
        assert_eq!(&*output.result.model, "teacher");
        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(&*requests[0].model, "small");
        assert_eq!(&*requests[1].model, "teacher");
        assert_eq!(requests[1].max_tokens, 512);
        assert_eq!(requests[0].guided_json.as_deref(), Some(&*schema));
        assert_eq!(
            requests[1].guided_json.as_deref(),
            Some(teacher_label_schema())
        );
    }

    // ── InferenceObserver attribution ───────────────────────────────────────

    /// Answers the first-pass model with a low-confidence local result and
    /// the second-pass model with a confident result from a different
    /// provider, so tests can tell which pass an observer call came from by
    /// its provider and usage rather than by call order alone.
    struct TieredMockProvider {
        second_pass_model: Arc<str>,
    }

    impl InferenceProvider for TieredMockProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Vllm
        }

        fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
            if request.model == self.second_pass_model {
                Ok(InferenceResult {
                    provider: ProviderKind::Gemini,
                    model: request.model.clone(),
                    output_text: r#"{"confidence":0.95,"description":"second"}"#.to_string(),
                    fallback_used: false,
                    finish_reason: Some("stop".to_string()),
                    inference_latency_ms: 200,
                    usage: TokenUsage {
                        prompt_tokens: 50,
                        completion_tokens: 20,
                        thinking_tokens: 5,
                        total_tokens: 75,
                    },
                })
            } else {
                Ok(InferenceResult {
                    provider: ProviderKind::Sglang,
                    model: request.model.clone(),
                    output_text: r#"{"confidence":0.1,"description":"first"}"#.to_string(),
                    fallback_used: false,
                    finish_reason: Some("stop".to_string()),
                    inference_latency_ms: 40,
                    usage: TokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        thinking_tokens: 0,
                        total_tokens: 15,
                    },
                })
            }
        }
    }

    #[derive(Default)]
    struct RecordingObserver {
        successes: Mutex<Vec<(ProviderKind, u64, bool, TokenUsage)>>,
        errors: Mutex<Vec<(ProviderKind, u64)>>,
    }

    impl InferenceObserver for RecordingObserver {
        fn record_success(
            &self,
            provider: ProviderKind,
            latency_ms: u64,
            fallback_used: bool,
            usage: TokenUsage,
        ) {
            self.successes
                .lock()
                .unwrap()
                .push((provider, latency_ms, fallback_used, usage));
        }

        fn record_error(&self, provider: ProviderKind, latency_ms: u64) {
            self.errors.lock().unwrap().push((provider, latency_ms));
        }
    }

    fn observer_test_request() -> InferenceRequest {
        InferenceRequest {
            model: Arc::from("placeholder"),
            prompt: Arc::from("prompt"),
            input_images: Vec::new(),
            input_videos: Vec::new(),
            max_tokens: 128,
            temperature: 0.0,
            timeout_ms: 1000,
            allow_fallback: true,
            guided_json: None,
        }
    }

    #[test]
    fn non_escalated_run_records_one_success_under_first_pass_provider() {
        let provider = TieredMockProvider {
            second_pass_model: Arc::from("teacher"),
        };
        // Threshold below the first pass's 0.1 confidence, so 0.1 < threshold
        // is false and the run never escalates, even though the models
        // differ enough to keep tiering active.
        let config = TieredVlmConfig {
            first_pass_model: Arc::from("small"),
            second_pass_model: Arc::from("teacher"),
            second_pass_threshold: 0.05,
            second_pass_max_tokens: 512,
        };
        let observer = RecordingObserver::default();

        let output = run_tiered(
            &provider,
            &config,
            observer_test_request(),
            1024,
            1000,
            Some(&observer),
        )
        .unwrap();

        assert!(!output.used_second_pass);
        let successes = observer.successes.lock().unwrap();
        assert_eq!(successes.len(), 1);
        assert_eq!(
            successes[0],
            (
                ProviderKind::Sglang,
                40,
                false,
                TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    thinking_tokens: 0,
                    total_tokens: 15,
                },
            )
        );
        assert!(observer.errors.lock().unwrap().is_empty());
    }

    #[test]
    fn escalated_run_records_each_pass_under_its_own_provider_with_own_usage() {
        let provider = TieredMockProvider {
            second_pass_model: Arc::from("teacher"),
        };
        let config = TieredVlmConfig {
            first_pass_model: Arc::from("small"),
            second_pass_model: Arc::from("teacher"),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 512,
        };
        let observer = RecordingObserver::default();

        let output = run_tiered(
            &provider,
            &config,
            observer_test_request(),
            1024,
            1000,
            Some(&observer),
        )
        .unwrap();

        assert!(output.used_second_pass);

        let first_pass_usage = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            thinking_tokens: 0,
            total_tokens: 15,
        };
        let second_pass_usage = TokenUsage {
            prompt_tokens: 50,
            completion_tokens: 20,
            thinking_tokens: 5,
            total_tokens: 75,
        };

        let successes = observer.successes.lock().unwrap();
        assert_eq!(successes.len(), 2);
        assert_eq!(
            successes[0],
            (ProviderKind::Sglang, 40, false, first_pass_usage)
        );
        // The second pass is recorded with its own un-accumulated usage and
        // latency, not the totals the caller ultimately sees in
        // output.result (which fold the first pass in via accumulate()).
        assert_eq!(
            successes[1],
            (ProviderKind::Gemini, 200, false, second_pass_usage)
        );

        // The returned result still carries the accumulated totals: this
        // proves the observer recording didn't change run_tiered's return
        // value, only added a side channel.
        assert_eq!(output.result.usage.total_tokens, 90);
        assert_eq!(output.result.inference_latency_ms, 240);
        assert!(observer.errors.lock().unwrap().is_empty());
    }

    #[test]
    fn run_tiered_without_observer_behaves_identically() {
        let provider = TieredMockProvider {
            second_pass_model: Arc::from("teacher"),
        };
        let config = TieredVlmConfig {
            first_pass_model: Arc::from("small"),
            second_pass_model: Arc::from("teacher"),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 512,
        };

        // No observer wired: run_tiered must not require one to function,
        // and there is nothing for it to record into.
        let output = run_tiered(
            &provider,
            &config,
            observer_test_request(),
            1024,
            1000,
            None,
        )
        .unwrap();

        assert!(output.used_second_pass);
        assert_eq!(output.result.usage.total_tokens, 90);
    }

    #[test]
    fn confidence_above_one_is_clamped_to_one() {
        assert_eq!(parse_confidence_from_output(r#"{"confidence": 1.5}"#), 1.0);
    }

    #[test]
    fn confidence_below_zero_is_clamped_to_zero() {
        assert_eq!(parse_confidence_from_output(r#"{"confidence": -0.3}"#), 0.0);
    }

    #[test]
    fn confidence_that_overflows_to_infinity_falls_back_to_default() {
        // "1e40" is a perfectly valid, finite JSON number (serde_json parses it
        // into an f64 fine), but it is far outside f32's range. Narrowing it to
        // f32 produces +infinity, which is exactly the kind of garbage a
        // malformed local output can produce. This is the real-world path to a
        // non-finite confidence: serde_json's own Value parser rejects numbers
        // that overflow *f64* (they simply fail to parse, hitting the existing
        // "not JSON" fallback), so f64-to-f32 narrowing is the only way a
        // successfully-parsed confidence value can end up non-finite.
        assert_eq!(parse_confidence_from_output(r#"{"confidence": 1e40}"#), 0.5);
        assert_eq!(
            parse_confidence_from_output(r#"{"confidence": -1e40}"#),
            0.5
        );
    }

    #[test]
    fn confidence_number_too_large_for_json_falls_back_to_default() {
        // This number overflows even f64, so serde_json refuses to parse the
        // document at all. It lands on the same fallback as any other
        // non-JSON text, exercising the pre-existing "invalid JSON" path
        // rather than the new non-finite guard, but the observable result is
        // still the safe default.
        assert_eq!(
            parse_confidence_from_output(r#"{"confidence": 1e400}"#),
            0.5
        );
    }

    #[test]
    fn valid_confidence_is_returned_unchanged() {
        assert_eq!(
            parse_confidence_from_output(r#"{"confidence": 0.42}"#),
            0.42
        );
    }

    #[test]
    fn missing_confidence_field_falls_back_to_default() {
        assert_eq!(parse_confidence_from_output(r#"{"other": 1}"#), 0.5);
    }

    #[test]
    fn non_json_output_falls_back_to_default() {
        assert_eq!(parse_confidence_from_output("not json at all"), 0.5);
    }

    #[test]
    fn clamped_max_confidence_does_not_trigger_escalation() {
        let config = TieredVlmConfig {
            first_pass_model: Arc::from("small"),
            second_pass_model: Arc::from("teacher"),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 256,
        };
        let confidence = parse_confidence_from_output(r#"{"confidence": 1.5}"#);
        assert_eq!(confidence, 1.0);
        assert!(!config.needs_second_pass(confidence));
    }

    #[test]
    fn clamped_min_confidence_triggers_escalation() {
        let config = TieredVlmConfig {
            first_pass_model: Arc::from("small"),
            second_pass_model: Arc::from("teacher"),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 256,
        };
        let confidence = parse_confidence_from_output(r#"{"confidence": -0.3}"#);
        assert_eq!(confidence, 0.0);
        assert!(config.needs_second_pass(confidence));
    }
}

// ── DistillationConfig ────────────────────────────────────────────────────────

/// Configuration for auto-distillation: teacher labeling + training pair collection.
///
/// When `enabled` is false, no frames are sampled and no DB writes happen.
///
/// # Environment variables
///
/// | Variable                             | Default   | Description |
/// |--------------------------------------|-----------|-------------|
/// | `VIDARAX_DISTILL_ENABLED`            | `false`   | Enable/disable collection |
/// | `VIDARAX_DISTILL_EMBEDDING_URL`      | —         | SigLIP2 embedding server URL |
/// | `VIDARAX_DISTILL_TEACHER_MODEL`      | (second-pass model) | Teacher VLM model ID |
/// | `VIDARAX_DISTILL_MAX_PAIRS`          | `10000`   | Max pairs per tenant before eviction |
/// | `VIDARAX_DISTILL_COLLECTION_RATE`    | `0.1`     | Fraction of keyframes to sample |
/// | `VIDARAX_DISTILL_DISTANCE_THRESHOLD` | `0.2`     | KNN distance accept threshold |
/// | `VIDARAX_DISTILL_KNN_K`              | `7`       | K for KNN classification |
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Whether the auto-distillation pipeline is active.
    pub enabled: bool,
    /// Base URL of the SigLIP2 embedding server (e.g. `http://127.0.0.1:8765`).
    pub embedding_server_url: Option<String>,
    /// Model ID used for teacher labeling (guided-JSON structured output).
    pub teacher_model: Arc<str>,
    /// Maximum training pairs stored per tenant before the oldest are evicted.
    pub max_pairs_per_tenant: usize,
    /// Fraction of keyframes to sample for collection (0.0 – 1.0).
    pub collection_rate: f32,
    /// Cosine-distance threshold for KNN classification acceptance.
    pub distance_threshold: f32,
    /// Number of neighbours for KNN voting.
    pub knn_k: usize,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            embedding_server_url: None,
            teacher_model: Arc::from("Qwen/Qwen3-VL-8B-Instruct"),
            max_pairs_per_tenant: 10_000,
            collection_rate: 0.1,
            distance_threshold: 0.2,
            knn_k: 7,
        }
    }
}
