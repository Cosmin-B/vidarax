use std::sync::Arc;

use crate::provider::{
    teacher_label_schema, InferenceProvider, InferenceRequest, InferenceResult, ProviderError,
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

    let first_conf = parse_confidence_from_output(&first_result.output_text);
    if config.needs_second_pass(first_conf) {
        request.model = Arc::clone(&config.second_pass_model);
        request.max_tokens = config.second_pass_max_tokens;
        request.timeout_ms = second_pass_timeout_ms;
        request.guided_json = Some(Arc::from(teacher_label_schema()));
        match provider.infer(&request) {
            Ok(result) => {
                return Ok(TieredVlmRun {
                    result,
                    used_second_pass: true,
                    request,
                });
            }
            Err(_) => {
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
            return conf as f32;
        }
    }
    0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{
        teacher_label_schema, InferenceProvider, InferenceRequest, InferenceResult, ProviderError,
        ProviderKind,
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

        let output = run_tiered(&provider, &config, request, 1024, 1000).unwrap();

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
