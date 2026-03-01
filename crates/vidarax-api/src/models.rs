use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct CreateRunRequest {
    pub mode: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferRequest {
    pub run_id: Option<String>,
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_ms: Option<u64>,
    pub allow_fallback: Option<bool>,
    pub primary_provider: Option<String>,
    #[serde(default)]
    pub output_schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct InferBatchRequest {
    pub requests: Vec<InferRequest>,
    pub max_parallel: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AnalyzeFrameInput {
    pub frame_index: u64,
    pub pts_ms: u64,
    pub perceptual_hash: u64,
    pub luma_mean: f32,
    pub flicker_score: f32,
    pub ghosting_score: f32,
    pub noise_variance_score: f32,
}

#[derive(Debug, Deserialize)]
pub struct AnalyzeFramesRequest {
    pub mode: Option<String>,
    pub model: String,
    pub stream_id: Option<String>,
    pub sampling_policy: Option<String>,
    pub fixed_fps: Option<f32>,
    #[serde(default)]
    pub frames: Vec<AnalyzeFrameInput>,
    pub window_size: Option<usize>,
    pub segment_ms: Option<u64>,
    pub trace_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RealtimeReasonRequest {
    pub source_uri: String,
    pub mode: Option<String>,
    pub model: String,
    pub stream_id: Option<String>,
    pub sampling_policy: Option<String>,
    pub fixed_fps: Option<f32>,
    pub max_frames: Option<u64>,
    pub chunk_size: Option<usize>,
    pub window_size: Option<usize>,
    pub segment_ms: Option<u64>,
    pub marker_correction_window_frames: Option<u64>,
    pub semantic_inference: Option<bool>,
    pub semantic_frames_per_chunk: Option<usize>,
    pub semantic_timeout_ms: Option<u64>,
    pub primary_provider: Option<String>,
    pub semantic_prompt: Option<String>,
    pub first_pass_model: Option<String>,
    pub second_pass_model: Option<String>,
    pub second_pass_threshold: Option<f32>,
    pub trace_id: Option<String>,
    #[serde(default)]
    pub output_schema: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct InferResponse {
    pub request_id: String,
    pub run_id: Option<String>,
    pub provider: String,
    pub model: String,
    pub fallback_used: bool,
    pub output_text: String,
    pub finish_reason: Option<String>,
    pub inference_latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct InferBatchItemError {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct InferBatchItemResult {
    pub index: usize,
    pub ok: bool,
    pub result: Option<InferResponse>,
    pub error: Option<InferBatchItemError>,
}

#[derive(Debug, Serialize)]
pub struct InferBatchResponse {
    pub request_id: String,
    pub processed: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub results: Vec<InferBatchItemResult>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeWindow {
    pub start_ms: u64,
    pub end_ms: u64,
    pub segment_id: String,
    pub source: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeObject {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeEvent {
    pub r#type: String,
    pub score: f32,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeAnnotations {
    pub summary: String,
    pub objects: Vec<AnalyzeObject>,
    pub events: Vec<AnalyzeEvent>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeFallback {
    pub used: bool,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeTrace {
    pub request_id: String,
    pub trace_id: String,
    pub span_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeMarker {
    pub marker_id: String,
    pub run_id: String,
    pub stream_id: String,
    pub event_type: String,
    pub status: String,
    pub start_frame: u64,
    pub end_frame: u64,
    pub start_pts_ms: u64,
    pub end_pts_ms: u64,
    pub confidence: f32,
    pub supersedes_marker_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeFrameMetadata {
    pub run_id: String,
    pub stream_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub mode: String,
    pub model: String,
    pub sampling_policy: String,
    pub sample_fps: f32,
    pub window: AnalyzeWindow,
    pub annotations: AnalyzeAnnotations,
    pub confidence: f32,
    pub fallback: AnalyzeFallback,
    pub trace: AnalyzeTrace,
    pub ordering_key: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeFramesResponse {
    pub request_id: String,
    pub run_id: String,
    pub generated: usize,
    pub metadata: Vec<AnalyzeFrameMetadata>,
    pub markers: Vec<AnalyzeMarker>,
}

#[derive(Debug, Serialize)]
pub struct RealtimeReasonResponse {
    pub request_id: String,
    pub run_id: String,
    pub generated: usize,
    pub markers_emitted: usize,
    pub decoded_frames: usize,
    pub sample_fps: f32,
    pub lag_p95_ms: u64,
    pub lag_p99_ms: u64,
    pub metadata: Vec<AnalyzeFrameMetadata>,
    pub markers: Vec<AnalyzeMarker>,
}

#[derive(Debug, Serialize)]
pub struct CreateRunResponse {
    pub run_id: String,
    pub request_id: String,
    pub status: &'static str,
    pub mode: String,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FieldError {
    pub field: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPolicy {
    SourceFpsAdaptive,
    Fixed,
}

impl SamplingPolicy {
    pub fn parse(raw: Option<&str>) -> Result<Self, &'static str> {
        match raw
            .unwrap_or("source_fps_adaptive")
            .to_ascii_lowercase()
            .as_str()
        {
            "source_fps_adaptive" | "adaptive" => Ok(Self::SourceFpsAdaptive),
            "fixed" => Ok(Self::Fixed),
            _ => Err("sampling_policy must be one of: source_fps_adaptive, fixed"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceFpsAdaptive => "source_fps_adaptive",
            Self::Fixed => "fixed",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct FeedbackRequest {
    pub rating: u32,
    pub category: String,
    pub feedback: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelCatalogItem {
    pub id: String,
    pub tier: String,
    pub availability: String,
    pub providers_available: Vec<String>,
    pub fallback_candidates: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelCatalogResponse {
    pub request_id: String,
    pub models: Vec<ModelCatalogItem>,
}
