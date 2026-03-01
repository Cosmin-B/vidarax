/// Configuration for tiered VLM inference routing.
///
/// First-pass: fast, cheap model (e.g. Qwen3-VL-2B, <200ms).
/// Second-pass: accurate model (e.g. Qwen3-VL-8B, ~400ms) — only called
/// when first-pass confidence is below `second_pass_threshold`.
///
/// # Examples
///
/// ```
/// use vidarax_core::tiered_vlm::TieredVlmConfig;
///
/// let config = TieredVlmConfig {
///     first_pass_model: "Qwen/Qwen3-VL-2B-Instruct".to_string(),
///     second_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
///     second_pass_threshold: 0.7,
///     second_pass_max_tokens: 256,
/// };
/// assert!(config.is_tiered());
/// assert!(config.needs_second_pass(0.5));
/// assert!(!config.needs_second_pass(0.8));
/// ```
#[derive(Debug, Clone)]
pub struct TieredVlmConfig {
    /// Model ID for the fast first-pass inference.
    pub first_pass_model: String,
    /// Model ID for the accurate second-pass inference.
    pub second_pass_model: String,
    /// Confidence threshold: if first-pass confidence < this, run second-pass.
    /// Range: 0.0 to 1.0. Default: 0.7.
    pub second_pass_threshold: f32,
    /// Max tokens for second-pass (can be higher than first-pass for detailed output).
    pub second_pass_max_tokens: u32,
}

impl TieredVlmConfig {
    /// Create a config that uses the same model for both passes (no tiering).
    pub fn single_model(model: &str) -> Self {
        Self {
            first_pass_model: model.to_string(),
            second_pass_model: model.to_string(),
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
            first_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            second_pass_model: "Qwen/Qwen3-VL-8B-Instruct".to_string(),
            second_pass_threshold: 0.7,
            second_pass_max_tokens: 256,
        }
    }
}
