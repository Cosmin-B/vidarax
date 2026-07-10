use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─── Clip mode ────────────────────────────────────────────────────────────────

/// Parameters for clip-mode multi-frame VLM inference.
///
/// When `clip_mode` is set on a streaming request, frames are accumulated
/// into temporal windows and submitted as multi-image VLM calls instead of
/// being processed one keyframe at a time.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClipConfig {
    /// Frames per second to sample from the stream (1–30, default 6).
    #[serde(default = "default_target_fps")]
    pub target_fps: u32,
    /// Duration of each clip window in seconds (0.1–60, default 0.5).
    #[serde(default = "default_clip_length_seconds")]
    pub clip_length_seconds: f32,
    /// Minimum delay between clip emissions in seconds (0–60, default 0.5).
    #[serde(default = "default_delay_seconds")]
    pub delay_seconds: f32,
}

fn default_target_fps() -> u32 {
    6
}
fn default_clip_length_seconds() -> f32 {
    0.5
}
fn default_delay_seconds() -> f32 {
    0.5
}

impl ClipConfig {
    /// Convert to the core ClipConfig type used by the worker pipeline.
    pub fn into_core(self) -> vidarax_core::webrtc::clip::ClipConfig {
        vidarax_core::webrtc::clip::ClipConfig {
            target_fps: self.target_fps,
            clip_length_seconds: self.clip_length_seconds,
            delay_seconds: self.delay_seconds,
        }
    }
}

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
    /// Optional cap on the longest edge (px) of each frame sent to the VLM.
    /// The "fewer pixels" lever: smaller frames occupy fewer Gemini image tiles,
    /// cutting per-image prompt tokens. `None` keeps source resolution.
    pub semantic_frame_max_edge: Option<u32>,
    pub semantic_timeout_ms: Option<u64>,
    // Deserialized for request-shape compatibility; not consumed on the realtime path.
    #[allow(dead_code)]
    pub primary_provider: Option<String>,
    pub semantic_prompt: Option<String>,
    pub first_pass_model: Option<String>,
    pub second_pass_model: Option<String>,
    pub second_pass_threshold: Option<f32>,
    pub trace_id: Option<String>,
    #[serde(default)]
    pub output_schema: Option<Value>,
    /// Optional clip-mode config. When set, frames are accumulated into
    /// temporal windows for multi-image VLM inference.
    // Deserialized for request-shape compatibility; not consumed on the realtime path.
    #[allow(dead_code)]
    pub clip_mode: Option<ClipConfig>,
    /// Optional index name for this analysis pass.
    ///
    /// Tagging a run with an `index_name` allows multiple analysis passes over
    /// the same video with different prompts to be stored and queried
    /// independently.  For example:
    ///
    /// - Pass 1: `semantic_prompt = "detect safety violations"`, `index_name = "safety"`
    /// - Pass 2: `semantic_prompt = "count people"`, `index_name = "crowd"`
    ///
    /// Then `GET /v1/runs/{id}/events?index=safety` returns only events from
    /// the first pass.  When absent all events are returned regardless of index.
    pub index_name: Option<String>,
    /// When true, chunks are processed sequentially and each VLM call receives
    /// the previous chunk's description as temporal context. Slower but more
    /// accurate for interaction detection. Default: false (parallel).
    #[serde(default)]
    pub temporal_chain: Option<bool>,
    /// When true, each VLM call includes the previous chunk's frame as an
    /// additional image so the model can visually diff the two states.
    /// Implies temporal_chain=true. Default: false.
    #[serde(default)]
    pub visual_diff: Option<bool>,
    /// When true, only chunks containing a gate-detected scene cut are sent
    /// to VLM. Skips static chunks entirely. Default: false.
    #[serde(default)]
    // Deserialized for request-shape compatibility; not consumed on the realtime path.
    #[allow(dead_code)]
    pub gate_filter: Option<bool>,
    /// When true, extract short MP4 clips instead of JPEG frames for VLM
    /// input.  Each chunk becomes one video segment sent via `input_videos`.
    /// Default: false (JPEG frame mode).
    #[serde(default)]
    pub video_clip_mode: Option<bool>,
    /// Duration of each video clip in seconds when `video_clip_mode` is true.
    /// Must be > 0.  Default: 0.5.
    pub video_clip_duration_s: Option<f32>,
    /// Maximum number of concurrent VLM inference requests in parallel mode.
    /// Higher values increase throughput but may cause queueing on the GPU.
    /// Default: 4.
    pub vlm_concurrency: Option<usize>,
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
    pub tokens: vidarax_core::provider::TokenUsage,
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

/// Aggregate token + latency spend for a pipeline run, summed across every
/// analyzed chunk (and every tiered pass within a chunk). Lets callers see the
/// full cost of an analysis — "how many tokens did this cost, how long did the
/// model work" — without post-hoc log scraping.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct TokenMetrics {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub thinking_tokens: u32,
    pub total_tokens: u32,
    /// Summed wall-clock inference latency across chunks (server-side model time).
    pub inference_latency_ms: u64,
    /// Number of chunks that actually ran inference (denominator for per-chunk means).
    pub chunks_analyzed: usize,
}

impl TokenMetrics {
    /// Fold one analyzed chunk's token spend and latency into the run total,
    /// saturating on overflow and bumping the analyzed-chunk count.
    pub fn accumulate_chunk(&mut self, usage: vidarax_core::provider::TokenUsage, latency_ms: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(usage.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(usage.completion_tokens);
        self.thinking_tokens = self.thinking_tokens.saturating_add(usage.thinking_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
        self.inference_latency_ms = self.inference_latency_ms.saturating_add(latency_ms);
        self.chunks_analyzed += 1;
    }
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
    pub tokens: TokenMetrics,
    pub metadata: Vec<AnalyzeFrameMetadata>,
    pub markers: Vec<AnalyzeMarker>,
}

#[derive(Debug, Serialize)]
pub struct CreateRunResponse {
    pub run_id: String,
    pub request_id: String,
    pub status: &'static str,
    pub mode: &'static str,
    pub model: Option<&'static str>,
}

// ─── Semantic search ─────────────────────────────────────────────────────────

/// Request body for `POST /v1/search`.
///
/// Performs a substring search over stored VLM descriptions in the WAL.
/// Matches are returned ordered by WAL sequence (ascending).
///
/// # Example
///
/// ```json
/// { "query": "person walking", "run_id": "run-abc123", "limit": 10 }
/// ```
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Query string.  All WAL events whose VLM description contains this
    /// substring (case-insensitive) are returned.
    pub query: String,
    /// Restrict results to a single run.  When absent all runs are searched.
    pub run_id: Option<String>,
    /// Maximum number of results to return (1–500, default 50).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// A single search result hit.
#[derive(Debug, Serialize)]
pub struct SearchHit {
    /// WAL sequence number of the matching event.
    pub seq: u64,
    /// Run the event belongs to.
    pub run_id: String,
    /// Presentation timestamp of the event in milliseconds.
    pub pts_ms: u64,
    /// WAL event kind (`"semantic_chunk_inferred"`, `"vlm"`, etc.).
    pub kind: String,
    /// The matched VLM description excerpt.
    pub description: String,
    /// Index name tag, if the event was produced by a named analysis pass.
    pub index_name: Option<String>,
}

/// Response body for `POST /v1/search`.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub request_id: String,
    /// Total number of events scanned.
    pub scanned: usize,
    /// Number of matching events found (before `limit` is applied).
    pub total_hits: usize,
    pub hits: Vec<SearchHit>,
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

// ─── Stream attach ────────────────────────────────────────────────────────────

/// Optional per-stream configuration sent when attaching a session.
///
/// Used to set the initial VLM prompt, token rate cap, and clip mode.
#[derive(Debug, Clone, Deserialize)]
pub struct AttachStreamRequest {
    /// Initial VLM analysis prompt for this session.
    pub prompt: Option<String>,
    /// Maximum VLM output tokens per second (backpressure).
    /// Overrides the server default (`VIDARAX_WEBRTC_MAX_OUTPUT_TOKENS_PER_SECOND`).
    #[serde(alias = "token_cap", alias = "token-cap")]
    pub max_output_tokens_per_second: Option<u32>,
    /// Optional clip-mode config. When set, frames are accumulated into
    /// temporal windows for multi-image VLM inference instead of per-keyframe.
    pub clip_mode: Option<ClipConfig>,
    /// Optional index name for this streaming session.
    ///
    /// When set, all VLM events emitted during this session are tagged with the
    /// given index name so they can be filtered independently from other
    /// analysis passes on the same run via `GET /v1/runs/{id}/events?index=…`.
    pub index_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── AttachStreamRequest deserialization ─────────────────────────────────

    #[test]
    fn attach_stream_request_parses_max_output_tokens_per_second() {
        let raw = r#"{"max_output_tokens_per_second": 64}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.max_output_tokens_per_second, Some(64u32));
    }

    #[test]
    fn attach_stream_request_token_rate_absent_is_none() {
        let raw = r#"{}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(
            parsed.max_output_tokens_per_second, None,
            "absent field should deserialise as None"
        );
    }

    #[test]
    fn attach_stream_request_parses_prompt_field() {
        let raw = r#"{"prompt": "describe the scene"}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.prompt.as_deref(), Some("describe the scene"));
    }

    #[test]
    fn attach_stream_request_prompt_absent_is_none() {
        let raw = r#"{}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert!(parsed.prompt.is_none(), "absent 'prompt' should be None");
    }

    #[test]
    fn attach_stream_request_parses_all_fields() {
        let raw = r#"{
            "prompt": "what is happening?",
            "max_output_tokens_per_second": 32,
            "clip_mode": {
                "target_fps": 8,
                "clip_length_seconds": 1.5,
                "delay_seconds": 0.25
            }
        }"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.prompt.as_deref(), Some("what is happening?"));
        assert_eq!(parsed.max_output_tokens_per_second, Some(32u32));
        let clip = parsed.clip_mode.unwrap();
        assert_eq!(clip.target_fps, 8);
        assert!((clip.clip_length_seconds - 1.5).abs() < 1e-5);
        assert!((clip.delay_seconds - 0.25).abs() < 1e-5);
    }

    #[test]
    fn attach_stream_request_clip_mode_absent_is_none() {
        let raw = r#"{"prompt": "hello"}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert!(
            parsed.clip_mode.is_none(),
            "absent 'clip_mode' should be None"
        );
    }

    // ─── SamplingPolicy parsing ───────────────────────────────────────────────

    #[test]
    fn sampling_policy_parses_adaptive_aliases() {
        assert_eq!(
            SamplingPolicy::parse(Some("source_fps_adaptive")),
            Ok(SamplingPolicy::SourceFpsAdaptive)
        );
        assert_eq!(
            SamplingPolicy::parse(Some("adaptive")),
            Ok(SamplingPolicy::SourceFpsAdaptive)
        );
    }

    #[test]
    fn sampling_policy_parses_fixed() {
        assert_eq!(
            SamplingPolicy::parse(Some("fixed")),
            Ok(SamplingPolicy::Fixed)
        );
    }

    #[test]
    fn sampling_policy_defaults_to_adaptive_when_none() {
        assert_eq!(
            SamplingPolicy::parse(None),
            Ok(SamplingPolicy::SourceFpsAdaptive)
        );
    }

    #[test]
    fn sampling_policy_rejects_unknown_value() {
        let result = SamplingPolicy::parse(Some("random"));
        assert!(result.is_err());
    }

    // ─── index_name deserialization ───────────────────────────────────────────

    #[test]
    fn realtime_reason_request_parses_index_name() {
        let raw = r#"{
            "source_uri": "file:///tmp/test.mp4",
            "model": "Qwen/Qwen3-VL-2B",
            "index_name": "safety"
        }"#;
        let parsed: RealtimeReasonRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.index_name.as_deref(), Some("safety"));
    }

    #[test]
    fn realtime_reason_request_index_name_absent_is_none() {
        let raw = r#"{
            "source_uri": "file:///tmp/test.mp4",
            "model": "Qwen/Qwen3-VL-2B"
        }"#;
        let parsed: RealtimeReasonRequest = serde_json::from_str(raw).unwrap();
        assert!(parsed.index_name.is_none());
    }

    #[test]
    fn attach_stream_request_parses_index_name() {
        let raw = r#"{"index_name": "crowd"}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.index_name.as_deref(), Some("crowd"));
    }

    #[test]
    fn attach_stream_request_index_name_absent_is_none() {
        let raw = r#"{}"#;
        let parsed: AttachStreamRequest = serde_json::from_str(raw).unwrap();
        assert!(parsed.index_name.is_none());
    }
}
