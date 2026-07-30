use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::{Id as TaskId, JoinError, JoinSet};
use vidarax_core::audio_sidecar::{
    AudioAnalysis, AudioAnalysisRequest, AudioFailureReason, AudioProfile, AudioSidecarClient,
    SidecarCapacity, SpeechEngine, SynthesizedAudio,
};
use vidarax_core::coordinates::{FrameCoordinates, IMAGE_COORDINATE_SCHEMA};
use vidarax_core::crop::CropRegion;
use vidarax_core::gate::{FrameSignal, GateEventType};
use vidarax_core::ingest::pipeline::DecodePipeline;
use vidarax_core::ingest::{
    extract_audio_video_clip, extract_audio_wav, DecodedJpegFrame, MediaInfo, PreparedSource,
};
use vidarax_core::metrics::PipelineMetrics;
use vidarax_core::pipeline::{FrameMetadata, TwoPassPipeline};
use vidarax_core::provider::{
    InferenceImage, InferenceObserver, InferenceProvider, InferenceRequest, InferenceVideo,
    MediaResolution, ProviderError, TokenUsage,
};
use vidarax_core::tiered_vlm::{run_tiered_with_second_pass_schema, TieredVlmConfig};
use vidarax_core::timeline::TimelineEvent;

use crate::models::{
    AnalyzeAnnotations, AnalyzeEvent, AnalyzeFallback, AnalyzeFrameMetadata, AnalyzeMarker,
    AnalyzeObject, AnalyzeTrace, AnalyzeWindow, SamplingPolicy,
};
use crate::semantic::{MarkerInput, SemanticMarker};
use crate::state::AppState;

#[cfg(test)]
static SEMANTIC_TASK_PANIC_CHUNK_FOR_TESTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

const SEMANTIC_OVERLAY_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "event_type": { "type": "string" },
    "object_label": { "type": "string" },
    "summary": { "type": "string" },
    "description": { "type": "string" },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
  },
  "required": ["event_type", "object_label", "summary", "description", "confidence"]
}"#;
const MULTIMODAL_OVERLAY_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "moments": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "start_offset_ms": { "type": "integer" },
          "end_offset_ms": { "type": "integer" },
          "modalities": {
            "type": "array",
            "items": { "type": "string" }
          },
          "kind": { "type": "string" },
          "description": { "type": "string" },
          "intent": { "type": "string" },
          "audio_visual_relation": { "type": "string" },
          "confidence": { "type": "number" }
        },
        "required": ["start_offset_ms", "end_offset_ms", "modalities", "kind", "description", "confidence"]
      }
    }
  },
  "required": ["moments"]
}"#;
const DEFAULT_SEMANTIC_MAX_TOKENS: u32 = 320;
const MULTIMODAL_MAX_TOKENS: u32 = 1_024;
const CUSTOM_SCHEMA_MAX_TOKENS: u32 = 1_024;
const MAX_MULTIMODAL_MOMENTS: usize = 32;

pub struct DecodedSignals {
    pub signals: Vec<FrameSignal>,
    pub sampling_policy: SamplingPolicy,
    pub sample_fps: f32,
    pub coordinates: Option<FrameCoordinates>,
}

#[derive(Debug, Clone)]
pub struct SemanticOverlay {
    pub event_type: String,
    pub object_label: String,
    pub summary: String,
    pub description: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SemanticMoment {
    pub start_offset_ms: u64,
    pub end_offset_ms: u64,
    pub start_pts_ms: u64,
    pub end_pts_ms: u64,
    pub modalities: Vec<String>,
    pub kind: String,
    pub description: String,
    pub intent: Option<String>,
    pub audio_visual_relation: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticMediaMode {
    Frames,
    Video,
    AudioVideo,
}

impl SemanticMediaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Frames => "frames",
            Self::Video => "video",
            Self::AudioVideo => "audio_video",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SemanticMediaConfig {
    pub mode: SemanticMediaMode,
    pub window_ms: u64,
    pub resolution: MediaResolution,
    pub persist_evidence: bool,
    pub timestamp_windows: bool,
}

#[derive(Clone)]
pub struct ClipSpec {
    pub source: Arc<PreparedSource>,
    pub decode_pipeline: Arc<dyn DecodePipeline>,
    pub source_start_ms: u64,
    pub duration_ms: u64,
    pub crop: Option<CropRegion>,
    pub mode: SemanticMediaMode,
    pub resolution: MediaResolution,
    pub persist_evidence: bool,
    pub local_audio: Option<LocalAudioConfig>,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Default)]
pub struct AudioTraceContext {
    pub run_id: Arc<str>,
    pub request_id: Arc<str>,
    pub stream_id: Arc<str>,
}

#[derive(Clone)]
pub struct LocalAudioConfig {
    pub sidecar_addr: Arc<str>,
    pub profile: AudioProfile,
    pub speech_engine: SpeechEngine,
    pub min_confidence: f32,
    pub max_events: u16,
    pub voice_feedback: bool,
    pub trace: AudioTraceContext,
    pub metrics: Arc<PipelineMetrics>,
}

#[derive(Debug, Clone)]
pub struct SemanticMediaEvidence {
    pub bytes: Arc<[u8]>,
    pub media_type: &'static str,
    pub mode: SemanticMediaMode,
    pub source_start_ms: u64,
    pub source_end_ms: u64,
    pub audio_streams: u16,
    pub audio_channels: u16,
    pub audio_mixed: bool,
    pub extraction_ms: u64,
    pub resolution: MediaResolution,
    pub persist_evidence: bool,
}

#[derive(Debug, Clone)]
pub struct FeedbackAudioEvidence {
    pub bytes: Arc<[u8]>,
    pub media_type: &'static str,
    pub model: String,
    pub sample_rate_hz: u32,
    pub processing_ms: u64,
    pub capacity: SidecarCapacity,
}

#[derive(Debug, Clone, Default)]
pub struct LocalAudioTelemetry {
    pub wav_extraction_ms: u64,
    pub wav_bytes: u64,
    pub requested_duration_ms: u64,
    pub round_trip_ms: u64,
    pub failure_reason: Option<AudioFailureReason>,
    pub tts_attempted: bool,
    pub tts_round_trip_ms: u64,
    pub tts_failure_reason: Option<AudioFailureReason>,
}

struct ExtractedLocalAudio {
    wav: Vec<u8>,
    extraction_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ChunkSemanticResult {
    pub overlay: Option<SemanticOverlay>,
    pub raw_output: Option<Value>,
    pub provider: Option<String>,
    pub provider_fallback_used: bool,
    pub used_fallback: bool,
    pub error: Option<String>,
    pub attempted: bool,
    pub finish_reason: Option<String>,
    pub response_chars: Option<usize>,
    /// Token spend for this chunk's analysis (summed across tiered passes).
    pub usage: TokenUsage,
    /// Wall-clock inference latency for this chunk (summed across passes).
    pub inference_latency_ms: u64,
    pub moments: Vec<SemanticMoment>,
    pub media: Option<SemanticMediaEvidence>,
    pub local_audio: Option<AudioAnalysis>,
    pub local_audio_error: Option<String>,
    pub feedback_audio: Option<FeedbackAudioEvidence>,
    pub local_audio_telemetry: LocalAudioTelemetry,
}

impl ChunkSemanticResult {
    pub fn event_payload(
        &self,
        chunk_idx: usize,
        request_id: &str,
        stream_id: &str,
    ) -> Option<Value> {
        self.attempted.then(|| {
            json!({
                "request_id": request_id,
                "stream_id": stream_id,
                "chunk_index": chunk_idx,
                "provider": self.provider,
                "provider_fallback_used": self.provider_fallback_used,
                "semantic_fallback_used": self.used_fallback,
                "semantic_error": self.error,
                "finish_reason": self.finish_reason,
                "response_chars": self.response_chars,
                "event_type": self.overlay.as_ref().map(|o| o.event_type.clone()),
                "object_label": self.overlay.as_ref().map(|o| o.object_label.clone()),
                "summary": self.overlay.as_ref().map(|o| o.summary.clone()),
                "description": self.overlay.as_ref().map(|o| o.description.clone()),
                "confidence": self.overlay.as_ref().map(|o| o.confidence),
                "raw_output": self.raw_output,
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "thinking_tokens": self.usage.thinking_tokens,
                "total_tokens": self.usage.total_tokens,
                "inference_latency_ms": self.inference_latency_ms,
                "moments": &self.moments,
                "local_audio": &self.local_audio,
                "local_audio_error": &self.local_audio_error,
            })
        })
    }
}

pub struct ChunkPrep {
    pub analyzed: Vec<FrameMetadata>,
    pub frame_offset: usize,
    pub chunk_jpegs: Arc<[DecodedJpegFrame]>,
    pub clip_spec: Option<ClipSpec>,
    pub pts_start_ms: u64,
    pub pts_end_ms: u64,
    pub chunk_len: usize,
    pub started: Instant,
}

pub fn load_decoded_signals_from_events(
    events: &[TimelineEvent],
) -> Result<DecodedSignals, String> {
    let Some(decoded_event) = events
        .iter()
        .rev()
        .find(|event| event.kind == "frames_decoded")
    else {
        return Err("frames must be provided when no decoded ingest frames exist".to_string());
    };

    let payload = serde_json::from_str::<Value>(&decoded_event.payload)
        .map_err(|_| "decoded ingest payload is invalid json".to_string())?;
    let sampling_policy = SamplingPolicy::parse(
        payload
            .get("sampling_policy")
            .and_then(|value| value.as_str()),
    )
    .map_err(ToString::to_string)?;
    let Some(signals) = payload.get("signals").and_then(|value| value.as_array()) else {
        return Err("decoded ingest payload is missing signals array".to_string());
    };
    if signals.is_empty() {
        return Err("decoded ingest payload contains no frame signals".to_string());
    }
    if signals.len() > 500_000 {
        return Err("decoded ingest frame signals exceed limit of 500000".to_string());
    }

    let mut out = Vec::with_capacity(signals.len());
    for signal in signals {
        let frame_index = signal
            .get("frame_index")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "decoded signal is missing frame_index".to_string())?;
        let pts_ms = signal
            .get("pts_ms")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "decoded signal is missing pts_ms".to_string())?;
        let perceptual_hash = signal
            .get("perceptual_hash")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "decoded signal is missing perceptual_hash".to_string())?;
        let luma_mean = signal
            .get("luma_mean")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing luma_mean".to_string())?
            as f32;
        let flicker_score = signal
            .get("flicker_score")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing flicker_score".to_string())?
            as f32;
        let ghosting_score = signal
            .get("ghosting_score")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing ghosting_score".to_string())?
            as f32;
        let noise_variance_score = signal
            .get("noise_variance_score")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing noise_variance_score".to_string())?
            as f32;
        if !(0.0..=1.0).contains(&luma_mean)
            || !(0.0..=1.0).contains(&flicker_score)
            || !(0.0..=1.0).contains(&ghosting_score)
            || !(0.0..=1.0).contains(&noise_variance_score)
        {
            return Err("decoded signal values must be normalized to [0.0, 1.0]".to_string());
        }

        out.push(FrameSignal {
            frame_index,
            pts_ms,
            perceptual_hash,
            luma_mean,
            flicker_score,
            ghosting_score,
            noise_variance_score,
        });
    }

    let sample_fps = payload
        .get("sample_fps")
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .or_else(|| estimate_sample_fps(&out))
        .unwrap_or(1.0);
    let coordinates = match payload
        .get("coordinate_schema")
        .and_then(|value| value.as_str())
    {
        Some(IMAGE_COORDINATE_SCHEMA) => {
            let value = payload
                .get("coordinates")
                .cloned()
                .ok_or_else(|| "decoded ingest payload is missing coordinates".to_string())?;
            Some(
                serde_json::from_value(value)
                    .map_err(|_| "decoded ingest coordinates are invalid".to_string())?,
            )
        }
        _ => None,
    };

    Ok(DecodedSignals {
        signals: out,
        sampling_policy,
        sample_fps,
        coordinates,
    })
}

// Realtime chunk preparation receives distinct decode controls and borrowed pipeline state.
#[allow(clippy::too_many_arguments)]
pub async fn prepare_realtime_chunks(
    signals: &[FrameSignal],
    chunk_size: usize,
    decoded_jpegs: Option<&std::collections::HashMap<u64, DecodedJpegFrame>>,
    pipeline: &mut TwoPassPipeline,
    decode_pipeline: &Arc<dyn DecodePipeline>,
    prepared_source: &Arc<PreparedSource>,
    media: SemanticMediaConfig,
    semantic_decode_enabled: bool,
    crop: Option<CropRegion>,
    local_audio: Option<LocalAudioConfig>,
) -> Vec<ChunkPrep> {
    let mut chunk_preps: Vec<ChunkPrep> = Vec::new();
    let mut ranges = Vec::new();
    let source_duration_ms = (media.mode != SemanticMediaMode::Frames)
        .then(|| {
            prepared_source
                .media_info()
                .ok()
                .and_then(|info| info.duration_ms)
        })
        .flatten();
    if media.timestamp_windows && media.mode != SemanticMediaMode::Frames {
        let mut start = 0usize;
        while start < signals.len() {
            let window_end = signals[start].pts_ms.saturating_add(media.window_ms);
            let mut end = start + 1;
            while end < signals.len() && signals[end].pts_ms < window_end {
                end += 1;
            }
            ranges.push(start..end);
            start = end;
        }
    } else {
        ranges.extend(
            (0..signals.len())
                .step_by(chunk_size)
                .map(|start| start..(start + chunk_size).min(signals.len())),
        );
    }

    for (chunk_index, range) in ranges.into_iter().enumerate() {
        let chunk = &signals[range.clone()];
        let pts_start_ms = chunk.first().map(|frame| frame.pts_ms).unwrap_or(0);
        if source_duration_ms.is_some_and(|duration| pts_start_ms >= duration) {
            continue;
        }
        let started = Instant::now();
        let analyzed = pipeline.analyze_batch(chunk).to_vec();
        let frame_offset = range.start;
        let chunk_jpegs: Arc<[DecodedJpegFrame]> = decoded_jpegs
            .map(|lookup| {
                let mut jpegs: Vec<DecodedJpegFrame> = (frame_offset..frame_offset + chunk.len())
                    .filter_map(|idx| lookup.get(&(idx as u64)).cloned())
                    .collect();
                jpegs.sort_by_key(|f| f.frame_index);
                Arc::from(jpegs)
            })
            .unwrap_or_else(|| Arc::from([]));

        let duration_ms = source_duration_ms
            .map(|source_duration| {
                media
                    .window_ms
                    .min(source_duration.saturating_sub(pts_start_ms))
            })
            .unwrap_or(media.window_ms)
            .max(1);
        let clip_spec = if media.mode != SemanticMediaMode::Frames && semantic_decode_enabled {
            Some(ClipSpec {
                source: Arc::clone(prepared_source),
                decode_pipeline: Arc::clone(decode_pipeline),
                source_start_ms: pts_start_ms,
                duration_ms,
                crop,
                mode: media.mode,
                resolution: media.resolution,
                persist_evidence: media.persist_evidence,
                local_audio: local_audio.clone(),
                chunk_index,
            })
        } else {
            None
        };

        chunk_preps.push(ChunkPrep {
            started,
            analyzed,
            frame_offset,
            chunk_jpegs,
            clip_spec,
            pts_start_ms,
            pts_end_ms: chunk.last().map(|f| f.pts_ms).unwrap_or(0),
            chunk_len: chunk.len(),
        });
    }
    chunk_preps
}

#[allow(clippy::too_many_arguments)]
pub async fn run_semantic_dispatch(
    chunk_preps: &[ChunkPrep],
    providers: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    semantic_available: bool,
    semantic_prompt: &str,
    semantic_timeout_ms: u64,
    semantic_frames_per_chunk: usize,
    tiered_config: TieredVlmConfig,
    guided_json_str: Option<Arc<str>>,
    visual_diff: bool,
    temporal_chain: bool,
    vlm_concurrency: usize,
    observer: Option<Arc<dyn InferenceObserver>>,
    inference_dispatch: Option<Arc<tokio::sync::Semaphore>>,
    completion_tx: Option<tokio::sync::mpsc::Sender<(usize, ChunkSemanticResult)>>,
) -> (Vec<Option<ChunkSemanticResult>>, Vec<Instant>) {
    let num_chunks = chunk_preps.len();
    let mut semantic_results: Vec<Option<ChunkSemanticResult>> =
        (0..num_chunks).map(|_| None).collect();
    let mut task_end_times: Vec<Instant> = vec![Instant::now(); num_chunks];

    if !semantic_available {
        return (semantic_results, task_end_times);
    }

    if temporal_chain {
        let mut last_description = String::new();
        let mut last_pts_ms: u64 = 0;
        let mut last_jpeg: Option<Arc<[u8]>> = None;

        for (chunk_idx, prep) in chunk_preps.iter().enumerate() {
            let prompt_with_context = if last_description.is_empty() {
                semantic_prompt.to_string()
            } else {
                format!(
                    "{semantic_prompt}\n[previous_state ({last_pts_ms}ms): {}]",
                    truncate_context(&last_description, 200)
                )
            };

            let prev_jpeg_ref = if visual_diff {
                last_jpeg.as_deref()
            } else {
                None
            };
            let result = infer_chunk_semantics(
                providers.clone(),
                true,
                &prompt_with_context,
                semantic_timeout_ms,
                semantic_frames_per_chunk,
                &prep.chunk_jpegs,
                prep.frame_offset as u64,
                prep.pts_start_ms,
                prep.pts_end_ms,
                tiered_config.clone(),
                guided_json_str.as_ref().map(Arc::clone),
                prev_jpeg_ref,
                prep.clip_spec.clone(),
                observer.clone(),
                inference_dispatch.clone(),
            )
            .await;

            if visual_diff {
                if let Some(frame) = select_semantic_images(&prep.chunk_jpegs, 1).first() {
                    last_jpeg = Some(Arc::clone(&frame.jpeg_bytes));
                }
            }

            if let Some(ref raw) = result.raw_output {
                let s = raw.to_string();
                if s.len() > 4 {
                    last_description.clear();
                    last_description.push_str(truncate_context(&s, 200));
                    last_pts_ms = prep.pts_end_ms;
                }
            } else if let Some(ref overlay) = result.overlay {
                last_description.clear();
                last_description.push_str(truncate_context(&overlay.description, 200));
                last_pts_ms = prep.pts_end_ms;
            }

            if let Some(tx) = &completion_tx {
                let _ = tx.send((chunk_idx, result.clone())).await;
            }
            semantic_results[chunk_idx] = Some(result);
            task_end_times[chunk_idx] = Instant::now();
        }
    } else {
        let mut join_set: JoinSet<(usize, ChunkSemanticResult, Instant)> = JoinSet::new();
        let max_in_flight = vlm_concurrency.max(1);
        let mut task_chunks: HashMap<TaskId, usize> =
            HashMap::with_capacity(max_in_flight.min(num_chunks));
        let mut pending = chunk_preps.iter().enumerate();

        for _ in 0..max_in_flight.min(num_chunks) {
            let (chunk_idx, task_id) = spawn_semantic_task(
                &mut join_set,
                pending.next().expect("bounded by num_chunks"),
                providers.clone(),
                semantic_prompt,
                semantic_timeout_ms,
                semantic_frames_per_chunk,
                tiered_config.clone(),
                guided_json_str.as_ref().map(Arc::clone),
                observer.clone(),
                inference_dispatch.clone(),
            );
            task_chunks.insert(task_id, chunk_idx);
        }

        while let Some(joined) = join_set.join_next_with_id().await {
            match joined {
                Ok((task_id, (idx, result, finished))) => {
                    task_chunks.remove(&task_id);
                    if let Some(tx) = &completion_tx {
                        let _ = tx.send((idx, result.clone())).await;
                    }
                    semantic_results[idx] = Some(result);
                    task_end_times[idx] = finished;
                }
                Err(err) => {
                    let finished = Instant::now();
                    let task_id = err.id();
                    if let Some(idx) = task_chunks.remove(&task_id) {
                        let result = semantic_join_failure_result(err);
                        if let Some(tx) = &completion_tx {
                            let _ = tx.send((idx, result.clone())).await;
                        }
                        semantic_results[idx] = Some(result);
                        task_end_times[idx] = finished;
                    } else {
                        tracing::warn!(
                            task_id = ?task_id,
                            error = %err,
                            "semantic inference task failed without chunk mapping"
                        );
                    }
                }
            }

            if let Some(next) = pending.next() {
                let (chunk_idx, task_id) = spawn_semantic_task(
                    &mut join_set,
                    next,
                    providers.clone(),
                    semantic_prompt,
                    semantic_timeout_ms,
                    semantic_frames_per_chunk,
                    tiered_config.clone(),
                    guided_json_str.as_ref().map(Arc::clone),
                    observer.clone(),
                    inference_dispatch.clone(),
                );
                task_chunks.insert(task_id, chunk_idx);
            }
        }
    }

    (semantic_results, task_end_times)
}

#[allow(clippy::too_many_arguments)]
fn spawn_semantic_task(
    join_set: &mut JoinSet<(usize, ChunkSemanticResult, Instant)>,
    (chunk_idx, prep): (usize, &ChunkPrep),
    providers: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    semantic_prompt: &str,
    semantic_timeout_ms: u64,
    semantic_frames_per_chunk: usize,
    tiered_config: TieredVlmConfig,
    guided_json_str: Option<Arc<str>>,
    observer: Option<Arc<dyn InferenceObserver>>,
    inference_dispatch: Option<Arc<tokio::sync::Semaphore>>,
) -> (usize, TaskId) {
    let providers_c = providers;
    let prompt_c = semantic_prompt.to_string();
    let chunk_jpegs_c = Arc::clone(&prep.chunk_jpegs);
    let clip_spec_c = prep.clip_spec.clone();
    let frame_offset = prep.frame_offset as u64;
    let pts_start_ms = prep.pts_start_ms;
    let pts_end_ms = prep.pts_end_ms;
    let tiered_config_c = tiered_config;
    let guided_json_c = guided_json_str;
    let observer_c = observer;
    let inference_dispatch_c = inference_dispatch;
    let handle = join_set.spawn(async move {
        #[cfg(test)]
        if chunk_idx
            == SEMANTIC_TASK_PANIC_CHUNK_FOR_TESTS.load(std::sync::atomic::Ordering::SeqCst)
        {
            panic!("injected semantic task panic for chunk {chunk_idx}");
        }

        let overlay = infer_chunk_semantics(
            providers_c,
            true,
            &prompt_c,
            semantic_timeout_ms,
            semantic_frames_per_chunk,
            &chunk_jpegs_c,
            frame_offset,
            pts_start_ms,
            pts_end_ms,
            tiered_config_c,
            guided_json_c,
            None,
            clip_spec_c,
            observer_c,
            inference_dispatch_c,
        )
        .await;
        (chunk_idx, overlay, Instant::now())
    });
    (chunk_idx, handle.id())
}

fn semantic_join_failure_result(err: JoinError) -> ChunkSemanticResult {
    ChunkSemanticResult {
        attempted: true,
        used_fallback: true,
        error: Some(format!("join_error:{err}")),
        ..ChunkSemanticResult::default()
    }
}

#[cfg(test)]
async fn bounded_task_spawn_probe_for_tests(total: usize, limit: usize) -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let active = Arc::new(AtomicUsize::new(0));
    let max_live = Arc::new(AtomicUsize::new(0));
    let mut join_set = JoinSet::new();
    let mut next = 0usize;
    let max_in_flight = limit.max(1);

    while next < total && join_set.len() < max_in_flight {
        spawn_probe_task(&mut join_set, Arc::clone(&active), Arc::clone(&max_live));
        next += 1;
    }
    while join_set.join_next().await.is_some() {
        if next < total {
            spawn_probe_task(&mut join_set, Arc::clone(&active), Arc::clone(&max_live));
            next += 1;
        }
    }

    max_live.load(Ordering::SeqCst)
}

#[cfg(test)]
fn spawn_probe_task(
    join_set: &mut JoinSet<()>,
    active: Arc<std::sync::atomic::AtomicUsize>,
    max_live: Arc<std::sync::atomic::AtomicUsize>,
) {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    join_set.spawn(async move {
        let live = active.fetch_add(1, Ordering::SeqCst) + 1;
        max_live.fetch_max(live, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(5)).await;
        active.fetch_sub(1, Ordering::SeqCst);
    });
}

#[allow(clippy::too_many_arguments)]
pub async fn infer_chunk_semantics(
    providers: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    semantic_available: bool,
    semantic_prompt: &str,
    timeout_ms: u64,
    semantic_frames_per_chunk: usize,
    chunk_jpegs: &[DecodedJpegFrame],
    frame_start_index: u64,
    pts_start_ms: u64,
    pts_end_ms: u64,
    tiered_config: TieredVlmConfig,
    guided_json: Option<Arc<str>>,
    prev_jpeg: Option<&[u8]>,
    clip_spec: Option<ClipSpec>,
    observer: Option<Arc<dyn InferenceObserver>>,
    inference_dispatch: Option<Arc<tokio::sync::Semaphore>>,
) -> ChunkSemanticResult {
    if !semantic_available {
        return ChunkSemanticResult::default();
    }

    let mut result = ChunkSemanticResult {
        attempted: true,
        used_fallback: false,
        ..ChunkSemanticResult::default()
    };

    let requested_media_mode = clip_spec.as_ref().map(|spec| spec.mode);
    let requested_audio_duration_ms = clip_spec.as_ref().map_or(0, |spec| spec.duration_ms);
    let audio_chunk_index = clip_spec.as_ref().map_or(0, |spec| spec.chunk_index);
    let local_audio_config = clip_spec.as_ref().and_then(|spec| spec.local_audio.clone());
    let use_video_clip = clip_spec.is_some();
    let selected = if use_video_clip {
        Vec::new()
    } else {
        let sel = select_semantic_images(chunk_jpegs, semantic_frames_per_chunk);
        if sel.is_empty() {
            result.used_fallback = true;
            result.error = Some("chunk_has_no_jpeg_frames".to_string());
            return result;
        }
        sel
    };

    let (extracted_media, local_audio_wav) = if let Some(spec) = clip_spec {
        let source_start_ms = spec.source_start_ms;
        let duration_ms = spec.duration_ms;
        let extraction = tokio::task::spawn_blocking(move || {
            let start_s = source_start_ms as f32 / 1_000.0;
            let duration_s = duration_ms as f32 / 1_000.0;
            let local_audio = spec.local_audio.clone();
            let media = match spec.mode {
                SemanticMediaMode::Video => {
                    let started = Instant::now();
                    spec.decode_pipeline
                        .extract_clip(spec.source.source(), start_s, duration_s, spec.crop)
                        .map(|bytes| SemanticMediaEvidence {
                            bytes: Arc::from(bytes),
                            media_type: "video/mp4",
                            mode: spec.mode,
                            source_start_ms,
                            source_end_ms: source_start_ms.saturating_add(duration_ms),
                            audio_streams: 0,
                            audio_channels: 0,
                            audio_mixed: false,
                            extraction_ms: started.elapsed().as_millis() as u64,
                            resolution: spec.resolution,
                            persist_evidence: spec.persist_evidence,
                        })
                }
                SemanticMediaMode::AudioVideo => {
                    let info: MediaInfo = spec.source.media_info()?.clone();
                    extract_audio_video_clip(
                        spec.source.source(),
                        start_s,
                        duration_s,
                        spec.crop,
                        &info,
                    )
                    .map(|clip| SemanticMediaEvidence {
                        bytes: Arc::from(clip.bytes),
                        media_type: "video/mp4",
                        mode: spec.mode,
                        source_start_ms,
                        source_end_ms: source_start_ms.saturating_add(duration_ms),
                        audio_streams: clip.audio_streams,
                        audio_channels: clip.audio_channels,
                        audio_mixed: clip.audio_mixed,
                        extraction_ms: clip.extraction_ms,
                        resolution: spec.resolution,
                        persist_evidence: spec.persist_evidence,
                    })
                }
                SemanticMediaMode::Frames => unreachable!("frame mode has no clip spec"),
            }?;
            let wav = if local_audio.is_some() {
                let info: MediaInfo = spec.source.media_info()?.clone();
                let started = Instant::now();
                let wav = extract_audio_wav(spec.source.source(), start_s, duration_s, &info)?;
                Some(ExtractedLocalAudio {
                    wav,
                    extraction_ms: started.elapsed().as_millis() as u64,
                })
            } else {
                None
            };
            Ok::<_, String>((media, wav))
        })
        .await;
        match extraction {
            Ok(Ok((media, wav))) => (Some(media), wav),
            Ok(Err(error)) => {
                result.used_fallback = true;
                result.error = Some(format!("media_extraction_failed:{error}"));
                return result;
            }
            Err(error) => {
                result.used_fallback = true;
                result.error = Some(format!("media_extraction_join_error:{error}"));
                return result;
            }
        }
    } else {
        (None, None)
    };
    result.media = extracted_media.clone();
    let audio_source_end_ms = extracted_media
        .as_ref()
        .map(|media| media.source_end_ms)
        .unwrap_or_else(|| pts_end_ms.max(pts_start_ms));

    if let (Some(config), Some(extracted)) = (local_audio_config.as_ref(), local_audio_wav) {
        result.local_audio_telemetry.wav_extraction_ms = extracted.extraction_ms;
        result.local_audio_telemetry.wav_bytes = extracted.wav.len() as u64;
        result.local_audio_telemetry.requested_duration_ms = requested_audio_duration_ms;
        let config = config.clone();
        let trace = config.trace.clone();
        let wav = extracted.wav;
        let round_trip_started = Instant::now();
        let audio_span = tracing::info_span!(
            "audio.sidecar.analyze",
            profile = config.profile.as_str(),
            speech_engine = config.speech_engine.as_str(),
            wav_bytes = wav.len(),
            run_id = %trace.run_id,
            request_id = %trace.request_id,
            stream_id = %trace.stream_id,
            chunk_id = audio_chunk_index,
        );
        let audio_outcome = tokio::task::spawn_blocking(move || {
            let _entered = audio_span.enter();
            let _active = config.metrics.begin_local_audio_sidecar_request();
            let mut client = AudioSidecarClient::new(&config.sidecar_addr, timeout_ms)?;
            client.analyze(AudioAnalysisRequest {
                profile: config.profile,
                speech_engine: config.speech_engine,
                source_start_ms: pts_start_ms,
                min_confidence: config.min_confidence,
                max_events: config.max_events,
                wav: &wav,
            })
        })
        .await;
        result.local_audio_telemetry.round_trip_ms =
            round_trip_started.elapsed().as_millis() as u64;
        match audio_outcome {
            Ok(Ok(analysis)) => {
                result
                    .moments
                    .extend(audio_moments(&analysis, pts_start_ms, audio_source_end_ms));
                result.local_audio = Some(analysis);
            }
            Ok(Err(error)) => {
                result.local_audio_telemetry.failure_reason = Some(error.reason());
                result.local_audio_error = Some(error.to_string());
            }
            Err(error) => {
                result.local_audio_telemetry.failure_reason = Some(AudioFailureReason::Inference);
                result.local_audio_error = Some(format!("audio_sidecar_join_error:{error}"))
            }
        }
    }

    let multimodal_instruction = if requested_media_mode == Some(SemanticMediaMode::AudioVideo) {
        "\nTreat sound as evidence, not as a transcript request. When local transcript evidence is present, it is the only allowed source for spoken words. Never invent, extend, or paraphrase speech beyond that evidence. Call the source \"the speaker\" unless identity is directly visible and stated. Do not classify speech as character dialogue, commentary, narration, or UI audio without direct evidence. An audio-only claim must overlap a matching local audio observation. If no local observation supports it, omit it. Analyze non-speech sounds and their relationship to visible actions only when the evidence overlaps in time. Return at most 32 moments. Moment offsets are milliseconds from the beginning of this clip. Each modalities array may contain only audio and video. Each kind must be exactly speech, sound_effect, music, ambient, interaction, mechanical, or other. Confidence must be between 0 and 1. Use positive-duration intervals inside the clip. Do not attribute a moment to an original audio track because input tracks may have been mixed."
    } else {
        ""
    };
    let audio_context = result
        .local_audio
        .as_ref()
        .map(local_audio_prompt_context)
        .unwrap_or_default();
    let prompt = if use_video_clip {
        format!(
            "{semantic_prompt}{multimodal_instruction}{audio_context}\nchunk_pts_start_ms={pts_start_ms}\nchunk_pts_end_ms={pts_end_ms}"
        )
    } else {
        format!(
            "{semantic_prompt}{multimodal_instruction}{audio_context}\nchunk_frame_start={frame_start_index}\nchunk_frame_end={}\nchunk_pts_start_ms={pts_start_ms}\nchunk_pts_end_ms={pts_end_ms}",
            frame_start_index
                .saturating_add(chunk_jpegs.len() as u64)
                .saturating_sub(1)
        )
    };

    let Some(provider) = providers else {
        finish_local_audio_result(&mut result);
        maybe_generate_feedback(
            &mut result,
            local_audio_config.as_ref(),
            timeout_ms,
            audio_chunk_index,
        )
        .await;
        return result;
    };

    let (images, videos) = if let Some(media) = extracted_media.as_ref() {
        let vids = vec![InferenceVideo {
            media_type: media.media_type,
            raw_bytes: Some(Arc::clone(&media.bytes)),
            data_base64: String::new(),
            media_resolution: Some(media.resolution),
        }];
        (Vec::new(), vids)
    } else {
        let mut imgs: Vec<InferenceImage> = Vec::with_capacity(selected.len() + 1);
        if let Some(prev) = prev_jpeg {
            imgs.push(InferenceImage {
                media_type: "image/jpeg",
                data_base64: BASE64_STANDARD.encode(prev),
            });
        }
        imgs.extend(selected.iter().map(|frame| InferenceImage {
            media_type: "image/jpeg",
            data_base64: BASE64_STANDARD.encode(frame.jpeg_bytes.as_ref()),
        }));
        (imgs, Vec::new())
    };

    let has_custom_output_schema = guided_json.is_some();
    let multimodal = extracted_media
        .as_ref()
        .is_some_and(|media| media.mode == SemanticMediaMode::AudioVideo);
    let first_pass_max_tokens = if has_custom_output_schema {
        CUSTOM_SCHEMA_MAX_TOKENS
    } else if multimodal {
        MULTIMODAL_MAX_TOKENS
    } else {
        DEFAULT_SEMANTIC_MAX_TOKENS
    };
    let request_guided_json = guided_json.as_ref().map(Arc::clone).or_else(|| {
        Some(Arc::from(if multimodal {
            MULTIMODAL_OVERLAY_SCHEMA
        } else {
            SEMANTIC_OVERLAY_SCHEMA
        }))
    });
    let second_pass_guided_json = (!has_custom_output_schema).then(|| {
        Arc::from(if multimodal {
            MULTIMODAL_OVERLAY_SCHEMA
        } else {
            SEMANTIC_OVERLAY_SCHEMA
        })
    });
    let request = InferenceRequest {
        model: tiered_config.first_pass_model.clone(),
        prompt: Arc::from(prompt),
        input_images: images,
        input_videos: videos,
        max_tokens: first_pass_max_tokens,
        temperature: 0.0,
        timeout_ms,
        allow_fallback: true,
        guided_json: request_guided_json,
        scheduling: vidarax_core::provider::InferenceScheduling::new(
            Arc::from(format!("offline:{frame_start_index}")),
            vidarax_core::admission::LatencyClass::Offline,
            timeout_ms.saturating_mul(2),
            timeout_ms.min(1_000),
        ),
    };

    // Capture the failing model's backend kind before the closure moves
    // tiered_config. run_tiered only surfaces Err on a first-pass failure, so
    // this attributes any recorded error to the first-pass model's backend
    // instead of the router's default kind.
    let first_pass_kind = provider.kind_for_model(tiered_config.first_pass_model.as_ref());
    let call_started = Instant::now();
    let dispatch_permit = match inference_dispatch {
        Some(dispatch) => match dispatch.try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                result.used_fallback = true;
                result.error = Some("provider_saturated".to_string());
                return result;
            }
        },
        None => None,
    };
    let provider_result = match tokio::task::spawn_blocking({
        let provider = Arc::clone(&provider);
        let observer_for_call = observer.clone();
        move || {
            let _dispatch_permit = dispatch_permit;
            run_tiered_with_second_pass_schema(
                provider.as_ref(),
                &tiered_config,
                request,
                first_pass_max_tokens,
                timeout_ms,
                second_pass_guided_json,
                observer_for_call.as_deref(),
            )
        }
    })
    .await
    {
        Ok(Ok(output)) => output.result,
        Ok(Err(err)) => {
            // run_tiered already recorded any successful pass it made before
            // failing; a failed first pass records nothing internally, so the
            // caller (here) is where that error lands in /metrics.
            if let Some(o) = observer.as_deref() {
                o.record_error(first_pass_kind, call_started.elapsed().as_millis() as u64);
            }
            result.used_fallback = true;
            result.error = Some(match err.error {
                ProviderError::UnsupportedModel(_) => "unsupported_model".to_string(),
                ProviderError::HttpStatus(code) => format!("http_status_{code}"),
                ProviderError::Transport(_) => "transport_error".to_string(),
                ProviderError::InvalidResponse(_) => "invalid_response".to_string(),
                ProviderError::Saturated { .. } => "provider_saturated".to_string(),
                ProviderError::DeadlineMissed => "deadline_missed".to_string(),
                ProviderError::RequestBudget => "request_budget_exceeded".to_string(),
            });
            return result;
        }
        Err(err) => {
            if let Some(o) = observer.as_deref() {
                o.record_error(first_pass_kind, call_started.elapsed().as_millis() as u64);
            }
            result.used_fallback = true;
            result.error = Some(format!("join_error:{err}"));
            return result;
        }
    };

    result.provider = Some(provider_result.provider.name().to_string());
    result.provider_fallback_used = provider_result.fallback_used;
    result.finish_reason = provider_result.finish_reason.clone();
    result.response_chars = Some(provider_result.output_text.chars().count());
    result.usage = provider_result.usage;
    result.inference_latency_ms = provider_result.inference_latency_ms;
    result.media = extracted_media;

    if has_custom_output_schema {
        let parsed = serde_json::from_str::<Value>(&provider_result.output_text)
            .unwrap_or_else(|_| json!({"raw": provider_result.output_text}));
        result.raw_output = Some(parsed);
        result.overlay = None;
        result
    } else {
        match parse_semantic_overlay(
            &provider_result.output_text,
            multimodal
                .then(|| {
                    result.media.as_ref().map(|media| {
                        (
                            media.source_start_ms,
                            media.source_end_ms.saturating_sub(media.source_start_ms),
                        )
                    })
                })
                .flatten(),
        ) {
            Ok((overlay, moments)) => {
                let moments = ground_provider_moments(moments, result.local_audio.as_ref());
                result.moments.extend(moments);
                deduplicate_moments(&mut result.moments);
                result.overlay = if result.local_audio.is_some() {
                    overlay_from_moments(&result.moments)
                } else {
                    Some(overlay)
                };
                result.used_fallback = provider_result.fallback_used;
                maybe_generate_feedback(
                    &mut result,
                    local_audio_config.as_ref(),
                    timeout_ms,
                    audio_chunk_index,
                )
                .await;
                result
            }
            Err(parse_error) => {
                tracing::warn!(
                    error = parse_error.as_str(),
                    finish_reason = result.finish_reason.as_deref().unwrap_or("unknown"),
                    response_chars = result.response_chars.unwrap_or(0),
                    "semantic output did not match the overlay contract"
                );
                result.used_fallback = true;
                result.error = Some(format!("semantic_parse_failed:{}", parse_error.as_str()));
                result
            }
        }
    }
}

fn audio_moments(
    analysis: &AudioAnalysis,
    source_start_ms: u64,
    source_end_ms: u64,
) -> Vec<SemanticMoment> {
    let duration_ms = source_end_ms.saturating_sub(source_start_ms);
    analysis
        .observations
        .iter()
        .filter(|observation| observation.kind != "voice_activity")
        .filter_map(|observation| {
            let start_offset_ms = observation.start_offset_ms.min(duration_ms);
            let end_offset_ms = observation
                .end_offset_ms
                .max(start_offset_ms)
                .min(duration_ms);
            if end_offset_ms <= start_offset_ms {
                return None;
            }
            let description = observation
                .transcript
                .as_deref()
                .filter(|text| !text.trim().is_empty())
                .map_or_else(
                    || observation.label.clone(),
                    |transcript| format!("Speaker: {transcript}"),
                );
            let kind = audio_moment_kind(observation);
            Some(SemanticMoment {
                start_offset_ms,
                end_offset_ms,
                start_pts_ms: source_start_ms.saturating_add(start_offset_ms),
                end_pts_ms: source_start_ms.saturating_add(end_offset_ms),
                modalities: vec!["audio".to_string()],
                kind: kind.to_string(),
                description,
                intent: observation.transcript.clone(),
                audio_visual_relation: None,
                confidence: observation.confidence.clamp(0.0, 1.0),
            })
        })
        .collect()
}

fn audio_moment_kind(observation: &vidarax_core::audio_sidecar::AudioObservation) -> &'static str {
    if observation.kind == "speech" {
        return "speech";
    }
    match observation.label.as_str() {
        "music" => "music",
        "silence"
        | "inside_small_room"
        | "outside_urban_or_manmade"
        | "outside_rural_or_natural" => "ambient",
        "engine" | "engine_starting" | "idling" | "mechanisms" | "tools" | "vehicle" => {
            "mechanical"
        }
        _ => "sound_effect",
    }
}

fn ground_provider_moments(
    moments: Vec<SemanticMoment>,
    analysis: Option<&AudioAnalysis>,
) -> Vec<SemanticMoment> {
    let Some(analysis) = analysis else {
        return moments;
    };
    moments
        .into_iter()
        .filter_map(|mut moment| {
            if !moment.modalities.iter().any(|modality| modality == "audio") {
                return Some(moment);
            }
            let matching: Vec<_> = analysis
                .observations
                .iter()
                .filter(|observation| {
                    observation.kind != "voice_activity"
                        && intervals_overlap(
                            moment.start_offset_ms,
                            moment.end_offset_ms,
                            observation.start_offset_ms,
                            observation.end_offset_ms,
                            500,
                        )
                })
                .collect();
            let is_speech = moment.kind == "speech";
            let supported: Vec<_> = matching
                .into_iter()
                .filter(|observation| {
                    if is_speech {
                        observation.kind == "speech"
                            && observation
                                .transcript
                                .as_deref()
                                .is_some_and(|text| !text.trim().is_empty())
                    } else {
                        observation.kind == "audio_event"
                    }
                })
                .collect();
            if supported.is_empty() {
                if moment.modalities.len() == 1 {
                    return None;
                }
                moment.modalities.retain(|modality| modality != "audio");
                moment.audio_visual_relation = None;
                return Some(moment);
            }
            if is_speech {
                let transcript = supported
                    .iter()
                    .filter_map(|observation| observation.transcript.as_deref())
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                moment.description = format!("Speaker: {}", truncate_context(&transcript, 1_000));
                moment.intent = Some(truncate_context(&transcript, 1_000).to_string());
                if moment.modalities.len() == 1 {
                    moment.audio_visual_relation = None;
                }
            } else if moment.modalities.len() == 1 {
                let labels = supported
                    .iter()
                    .map(|observation| observation.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                moment.description = format!("Local audio event: {labels}");
                moment.intent = None;
                moment.audio_visual_relation = None;
            }
            Some(moment)
        })
        .collect()
}

fn intervals_overlap(
    left_start: u64,
    left_end: u64,
    right_start: u64,
    right_end: u64,
    tolerance_ms: u64,
) -> bool {
    left_start <= right_end.saturating_add(tolerance_ms)
        && right_start <= left_end.saturating_add(tolerance_ms)
}

fn deduplicate_moments(moments: &mut Vec<SemanticMoment>) {
    moments.sort_by_key(|moment| {
        (
            moment.start_offset_ms,
            moment.end_offset_ms,
            moment.kind.clone(),
        )
    });
    let mut deduplicated = Vec::with_capacity(moments.len());
    for moment in moments.drain(..) {
        let duplicate = deduplicated.iter().position(|previous: &SemanticMoment| {
            previous.kind == moment.kind
                && intervals_overlap(
                    previous.start_offset_ms,
                    previous.end_offset_ms,
                    moment.start_offset_ms,
                    moment.end_offset_ms,
                    250,
                )
                && (previous.description == moment.description || moment.kind == "speech")
        });
        if let Some(index) = duplicate {
            if moment.modalities.len() > deduplicated[index].modalities.len() {
                deduplicated[index] = moment;
            }
        } else {
            deduplicated.push(moment);
        }
    }
    *moments = deduplicated;
}

fn overlay_from_moments(moments: &[SemanticMoment]) -> Option<SemanticOverlay> {
    moments
        .iter()
        .max_by(|left, right| left.confidence.total_cmp(&right.confidence))
        .map(|moment| SemanticOverlay {
            event_type: "context_observation".to_string(),
            object_label: moment.kind.clone(),
            summary: moment.description.clone(),
            description: moment.description.clone(),
            confidence: moment.confidence,
        })
}

fn local_audio_prompt_context(analysis: &AudioAnalysis) -> String {
    if analysis.observations.is_empty() {
        return "\nLocal audio front-end found no event above its calibrated threshold."
            .to_string();
    }
    let mut context = String::from(
        "\nLocal audio observations follow. Transcript entries constrain spoken words. Sound labels remain classifier hypotheses:",
    );
    for (index, observation) in analysis.observations.iter().take(32).enumerate() {
        use std::fmt::Write as _;
        let _ = write!(
            context,
            "\n- audio:{} {}..{}ms kind={} label={} confidence={:.3} model={}",
            index,
            observation.start_offset_ms,
            observation.end_offset_ms,
            observation.kind,
            observation.label,
            observation.confidence,
            observation.model
        );
        if let Some(transcript) = observation.transcript.as_deref() {
            let transcript = truncate_context(transcript, 240);
            let _ = write!(context, " transcript={transcript}");
        }
        if let Some(emotion) = observation.emotion.as_deref() {
            let _ = write!(context, " emotion={emotion}");
        }
    }
    context
}

fn finish_local_audio_result(result: &mut ChunkSemanticResult) {
    let Some(moment) = result
        .moments
        .iter()
        .max_by(|left, right| left.confidence.total_cmp(&right.confidence))
    else {
        result.provider = Some("local_audio".to_string());
        result.overlay = Some(SemanticOverlay {
            event_type: "context_observation".to_string(),
            object_label: "audio".to_string(),
            summary: "No local audio event crossed the threshold".to_string(),
            description: "No local audio event crossed the configured threshold".to_string(),
            confidence: 0.0,
        });
        return;
    };
    result.provider = Some("local_audio".to_string());
    result.overlay = Some(SemanticOverlay {
        event_type: "context_observation".to_string(),
        object_label: moment.kind.clone(),
        summary: moment.description.clone(),
        description: moment.description.clone(),
        confidence: moment.confidence,
    });
}

async fn maybe_generate_feedback(
    result: &mut ChunkSemanticResult,
    config: Option<&LocalAudioConfig>,
    timeout_ms: u64,
    chunk_index: usize,
) {
    let Some(config) = config.filter(|config| config.voice_feedback) else {
        return;
    };
    let Some(text) = result
        .overlay
        .as_ref()
        .map(|overlay| overlay.description.trim())
        .filter(|text| !text.is_empty())
    else {
        return;
    };
    result.local_audio_telemetry.tts_attempted = true;
    let address = Arc::clone(&config.sidecar_addr);
    let trace = config.trace.clone();
    let metrics = Arc::clone(&config.metrics);
    let text = truncate_context(text, 2_000).to_string();
    let round_trip_started = Instant::now();
    let tts_span = tracing::info_span!(
        "audio.sidecar.synthesize",
        text_bytes = text.len(),
        run_id = %trace.run_id,
        request_id = %trace.request_id,
        stream_id = %trace.stream_id,
        chunk_id = chunk_index,
    );
    let outcome = tokio::task::spawn_blocking(move || {
        let _entered = tts_span.enter();
        let _active = metrics.begin_local_audio_sidecar_request();
        let mut client = AudioSidecarClient::new(&address, timeout_ms)?;
        client.synthesize(&text)
    })
    .await;
    result.local_audio_telemetry.tts_round_trip_ms =
        round_trip_started.elapsed().as_millis() as u64;
    match outcome {
        Ok(Ok(SynthesizedAudio {
            bytes,
            media_type,
            sample_rate_hz,
            processing_ms,
            model,
            capacity,
        })) => {
            result.feedback_audio = Some(FeedbackAudioEvidence {
                bytes: Arc::from(bytes),
                media_type,
                model,
                sample_rate_hz,
                processing_ms,
                capacity,
            });
        }
        Ok(Err(error)) => {
            result.local_audio_telemetry.tts_failure_reason = Some(error.reason());
            result.local_audio_error = Some(format!("voice_feedback_failed:{error}"));
        }
        Err(error) => {
            result.local_audio_telemetry.tts_failure_reason = Some(AudioFailureReason::Inference);
            result.local_audio_error = Some(format!("voice_feedback_join_error:{error}"));
        }
    }
}

pub fn select_semantic_images(
    chunk_jpegs: &[DecodedJpegFrame],
    semantic_frames_per_chunk: usize,
) -> Vec<&DecodedJpegFrame> {
    if chunk_jpegs.is_empty() {
        return Vec::new();
    }
    if semantic_frames_per_chunk >= chunk_jpegs.len() {
        return chunk_jpegs.iter().collect();
    }

    let mut out = Vec::with_capacity(semantic_frames_per_chunk);
    if semantic_frames_per_chunk == 1 {
        out.push(&chunk_jpegs[chunk_jpegs.len() / 2]);
        return out;
    }

    let mut last_idx = usize::MAX;
    for i in 0..semantic_frames_per_chunk {
        let idx = i * (chunk_jpegs.len() - 1) / (semantic_frames_per_chunk - 1);
        if idx != last_idx {
            out.push(&chunk_jpegs[idx]);
            last_idx = idx;
        }
    }
    out
}

#[derive(Debug, PartialEq, Eq)]
enum SemanticParseError {
    EmptyResponse,
    JsonObjectNotFound,
    InvalidJson,
    SchemaMismatch,
}

impl SemanticParseError {
    fn as_str(&self) -> &'static str {
        match self {
            Self::EmptyResponse => "empty_response",
            Self::JsonObjectNotFound => "json_object_not_found",
            Self::InvalidJson => "invalid_json",
            Self::SchemaMismatch => "schema_mismatch",
        }
    }
}

fn parse_semantic_overlay(
    raw: &str,
    media_window: Option<(u64, u64)>,
) -> Result<(SemanticOverlay, Vec<SemanticMoment>), SemanticParseError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(SemanticParseError::EmptyResponse);
    }
    let value = parse_first_json_object(trimmed)?;

    if let Some((source_start_ms, duration_ms)) = media_window {
        let moments = parse_semantic_moments(&value, source_start_ms, duration_ms)?;
        let overlay = moments.first().map_or_else(
            || SemanticOverlay {
                event_type: "context_observation".to_string(),
                object_label: "multimodal_activity".to_string(),
                summary: "No distinct audio-video moment".to_string(),
                description: "The window contained no moment above the event threshold".to_string(),
                confidence: 0.0,
            },
            |moment| SemanticOverlay {
                event_type: "context_observation".to_string(),
                object_label: moment.kind.clone(),
                summary: moment.description.clone(),
                description: moment.description.clone(),
                confidence: moment.confidence,
            },
        );
        return Ok((overlay, moments));
    }

    let event_type = normalize_semantic_event(required_string(&value, "event_type")?);
    let object_label = normalize_semantic_object(required_string(&value, "object_label")?);
    let summary = required_string(&value, "summary")?.trim();
    let summary = if summary.is_empty() {
        "semantic summary unavailable"
    } else {
        summary
    }
    .to_string();
    let description = required_string(&value, "description")?.trim();
    let description = if description.is_empty() {
        "semantic description unavailable"
    } else {
        description
    }
    .to_string();
    let confidence = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .filter(|v| (0.0..=1.0).contains(v))
        .ok_or(SemanticParseError::SchemaMismatch)? as f32;

    Ok((
        SemanticOverlay {
            event_type,
            object_label,
            summary,
            description,
            confidence,
        },
        Vec::new(),
    ))
}

fn parse_semantic_moments(
    value: &Value,
    source_start_ms: u64,
    duration_ms: u64,
) -> Result<Vec<SemanticMoment>, SemanticParseError> {
    let values = value
        .get("moments")
        .and_then(Value::as_array)
        .ok_or(SemanticParseError::SchemaMismatch)?;
    let mut moments = Vec::with_capacity(values.len().min(MAX_MULTIMODAL_MOMENTS));
    for value in values.iter().take(MAX_MULTIMODAL_MOMENTS) {
        let start_offset_ms = value
            .get("start_offset_ms")
            .and_then(Value::as_u64)
            .ok_or(SemanticParseError::SchemaMismatch)?
            .min(duration_ms);
        let end_offset_ms = value
            .get("end_offset_ms")
            .and_then(Value::as_u64)
            .ok_or(SemanticParseError::SchemaMismatch)?
            .clamp(start_offset_ms, duration_ms);
        if end_offset_ms <= start_offset_ms {
            continue;
        }
        let mut modalities = Vec::with_capacity(2);
        for modality in value
            .get("modalities")
            .and_then(Value::as_array)
            .ok_or(SemanticParseError::SchemaMismatch)?
        {
            let modality = modality
                .as_str()
                .ok_or(SemanticParseError::SchemaMismatch)?;
            let normalized = modality.trim().to_ascii_lowercase();
            let mapped = if normalized == "video"
                || normalized == "visual"
                || normalized.contains("picture")
            {
                Some("video")
            } else if normalized == "audio"
                || normalized.contains("speech")
                || normalized.contains("sound")
                || normalized.contains("voice")
                || normalized.contains("music")
            {
                Some("audio")
            } else {
                None
            };
            if let Some(mapped) = mapped {
                if !modalities.iter().any(|existing| existing == mapped) {
                    modalities.push(mapped.to_string());
                }
            }
        }
        let kind = value
            .get("kind")
            .and_then(Value::as_str)
            .map(normalize_moment_kind)
            .ok_or(SemanticParseError::SchemaMismatch)?;
        if modalities.is_empty() {
            if matches!(
                kind.as_str(),
                "speech" | "sound_effect" | "music" | "ambient" | "mechanical"
            ) {
                modalities.push("audio".to_string());
            } else {
                return Err(SemanticParseError::SchemaMismatch);
            }
        }
        let description = required_string(value, "description")?.trim();
        if description.is_empty() {
            return Err(SemanticParseError::SchemaMismatch);
        }
        let confidence = value
            .get("confidence")
            .and_then(Value::as_f64)
            .filter(|confidence| confidence.is_finite())
            .ok_or(SemanticParseError::SchemaMismatch)?
            .clamp(0.0, 1.0) as f32;
        let optional_text = |field: &str| {
            value
                .get(field)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(|text| truncate_context(text, 1_024).to_string())
        };
        moments.push(SemanticMoment {
            start_offset_ms,
            end_offset_ms,
            start_pts_ms: source_start_ms.saturating_add(start_offset_ms),
            end_pts_ms: source_start_ms.saturating_add(end_offset_ms),
            modalities,
            kind,
            description: truncate_context(description, 1_024).to_string(),
            intent: optional_text("intent"),
            audio_visual_relation: optional_text("audio_visual_relation"),
            confidence,
        });
    }
    moments.sort_by_key(|moment| (moment.start_offset_ms, moment.end_offset_ms));
    Ok(moments)
}

fn normalize_moment_kind(raw: &str) -> String {
    let kind = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    if kind == "speech"
        || kind.contains("dialog")
        || kind.contains("voice")
        || kind.contains("spoken")
    {
        "speech"
    } else if kind == "sound_effect"
        || kind.contains("effect")
        || kind.contains("beep")
        || kind.contains("tone")
        || kind.contains("impact")
    {
        "sound_effect"
    } else if kind.contains("music") {
        "music"
    } else if kind.contains("ambient") || kind.contains("background") {
        "ambient"
    } else if kind.contains("interaction") {
        "interaction"
    } else if kind.contains("mechanical") || kind.contains("machine") {
        "mechanical"
    } else {
        "other"
    }
    .to_string()
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, SemanticParseError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or(SemanticParseError::SchemaMismatch)
}

fn parse_first_json_object(raw: &str) -> Result<Value, SemanticParseError> {
    let mut saw_balanced_object = false;
    for (start, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }
        let Some(candidate) = extract_balanced_json_object(&raw[start..]) else {
            continue;
        };
        saw_balanced_object = true;
        if let Ok(value) = serde_json::from_str(candidate) {
            return Ok(value);
        }
    }
    if saw_balanced_object {
        Err(SemanticParseError::InvalidJson)
    } else {
        Err(SemanticParseError::JsonObjectNotFound)
    }
}

fn extract_balanced_json_object(raw: &str) -> Option<&str> {
    if !raw.starts_with('{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in raw.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&raw[..=offset]);
                }
            }
            _ => {}
        }
    }
    None
}

fn truncate_context(text: &str, max_chars: usize) -> &str {
    if max_chars == 0 {
        return "";
    }
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
// Helper functions below the test module are intentionally left in place.
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use vidarax_core::audio_sidecar::AudioObservation;
    use vidarax_core::provider::{InferenceResult, ProviderKind, TokenUsage};

    struct SemanticTestProvider;

    impl InferenceProvider for SemanticTestProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Vllm
        }

        fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
            assert_eq!(
                request.guided_json.as_deref(),
                Some(SEMANTIC_OVERLAY_SCHEMA)
            );
            assert_eq!(request.max_tokens, DEFAULT_SEMANTIC_MAX_TOKENS);
            Ok(InferenceResult {
                provider: ProviderKind::Vllm,
                model: Arc::clone(&request.model),
                output_text: r#"{"event_type":"context_observation","object_label":"frame_context","summary":"ok","description":"chunk completed","confidence":0.95}"#.to_string(),
                fallback_used: false,
                finish_reason: Some("stop".to_string()),
                inference_latency_ms: 1,
                usage: TokenUsage::default(),
            })
        }
    }

    struct CustomSchemaTestProvider;

    impl InferenceProvider for CustomSchemaTestProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Vllm
        }

        fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
            assert_eq!(request.guided_json.as_deref(), Some(r#"{"type":"object"}"#));
            assert_eq!(request.max_tokens, CUSTOM_SCHEMA_MAX_TOKENS);
            Ok(InferenceResult {
                provider: ProviderKind::Vllm,
                model: Arc::clone(&request.model),
                output_text: r#"{"custom":"ok"}"#.to_string(),
                fallback_used: false,
                finish_reason: Some("stop".to_string()),
                inference_latency_ms: 1,
                usage: TokenUsage::default(),
            })
        }
    }

    #[test]
    fn semantic_context_truncation_is_utf8_safe() {
        let text = "a".repeat(199) + "étail";
        assert_eq!(truncate_context(&text, 200), "a".repeat(199) + "é");

        let longer = "é".repeat(201);
        let truncated = truncate_context(&longer, 200);
        assert_eq!(truncated.chars().count(), 200);
        assert!(longer.starts_with(truncated));
    }

    #[test]
    fn chunk_semantic_event_payload_includes_overlay_text() {
        let skipped = ChunkSemanticResult {
            attempted: false,
            ..ChunkSemanticResult::default()
        };
        assert_eq!(skipped.event_payload(0, "req-1", "stream-1"), None);

        let result = ChunkSemanticResult {
            overlay: Some(SemanticOverlay {
                event_type: "context_observation".to_string(),
                object_label: "frame_context".to_string(),
                summary: "person crosses the lobby".to_string(),
                description: "A person walks past the front desk carrying a bag.".to_string(),
                confidence: 0.91,
            }),
            attempted: true,
            finish_reason: Some("stop".to_string()),
            response_chars: Some(128),
            ..ChunkSemanticResult::default()
        };

        let payload = result
            .event_payload(3, "req-1", "stream-1")
            .expect("attempted semantic result should emit payload");

        assert_eq!(
            payload.get("summary").and_then(Value::as_str),
            Some("person crosses the lobby")
        );
        assert_eq!(
            payload.get("description").and_then(Value::as_str),
            Some("A person walks past the front desk carrying a bag.")
        );
        assert_eq!(
            payload.get("finish_reason").and_then(Value::as_str),
            Some("stop")
        );
        assert_eq!(
            payload.get("response_chars").and_then(Value::as_u64),
            Some(128)
        );
    }

    #[test]
    fn semantic_overlay_parser_accepts_fenced_json_with_braces_in_strings() {
        let raw = r#"Result:
```json
{"event_type":"context_observation","object_label":"subject","summary":"turn {left}","description":"The subject oscillates before contact.","confidence":0.8}
```
Ignore this trailing {not json}."#;

        let (overlay, moments) =
            parse_semantic_overlay(raw, None).expect("fenced overlay should parse");
        assert_eq!(overlay.object_label, "subject");
        assert_eq!(overlay.summary, "turn {left}");
        assert_eq!(overlay.confidence, 0.8);
        assert!(moments.is_empty());
    }

    #[test]
    fn multimodal_parser_keeps_audio_video_timestamps_on_the_source_timeline() {
        let raw = r#"{
          "moments":[
            {
              "start_offset_ms":1200,
              "end_offset_ms":2400,
              "modalities":["video","audio","audio"],
              "kind":"interaction",
              "description":"The button is clicked as its click sound plays.",
              "intent":"The speaker is trying to run the build.",
              "audio_visual_relation":"The click sound coincides with the visible press.",
              "confidence":0.91
            },
            {
              "start_offset_ms":9000,
              "end_offset_ms":9500,
              "modalities":["audio"],
              "kind":"error tone",
              "description":"An error sound plays.",
              "confidence":1.4
            }
          ]
        }"#;

        let (overlay, moments) =
            parse_semantic_overlay(raw, Some((20_000, 8_000))).expect("multimodal overlay");
        assert_eq!(
            overlay.summary,
            "The button is clicked as its click sound plays."
        );
        assert_eq!(moments.len(), 1);
        assert_eq!(moments[0].start_pts_ms, 21_200);
        assert_eq!(moments[0].end_pts_ms, 22_400);
        assert_eq!(moments[0].modalities, ["video", "audio"]);
        assert_eq!(
            moments[0].intent.as_deref(),
            Some("The speaker is trying to run the build.")
        );
    }

    #[test]
    fn local_audio_filters_transport_activity_and_zero_duration() {
        let analysis = AudioAnalysis {
            profile: AudioProfile::Gameplay,
            speech_engine: SpeechEngine::Whisper,
            models: vec!["openai/whisper-large-v3-turbo".to_string()],
            observations: vec![
                AudioObservation {
                    start_offset_ms: 0,
                    end_offset_ms: 500,
                    kind: "voice_activity".to_string(),
                    label: "speech".to_string(),
                    confidence: 0.9,
                    model: "silero-vad-v6".to_string(),
                    transcript: None,
                    language: None,
                    emotion: None,
                },
                AudioObservation {
                    start_offset_ms: 100,
                    end_offset_ms: 100,
                    kind: "audio_event".to_string(),
                    label: "impact".to_string(),
                    confidence: 0.8,
                    model: "efficientat/mn10_as".to_string(),
                    transcript: None,
                    language: None,
                    emotion: None,
                },
                AudioObservation {
                    start_offset_ms: 200,
                    end_offset_ms: 800,
                    kind: "speech".to_string(),
                    label: "transcript".to_string(),
                    confidence: 0.95,
                    model: "openai/whisper-large-v3-turbo".to_string(),
                    transcript: Some("Fuel empty.".to_string()),
                    language: Some("en".to_string()),
                    emotion: None,
                },
            ],
            audio_duration_ms: 1_000,
            audio_bytes: 32_044,
            queue_wait_ms: 0,
            stages: Default::default(),
            capacity: Default::default(),
            processing_ms: 10,
        };

        let moments = audio_moments(&analysis, 5_000, 6_000);
        assert_eq!(moments.len(), 1);
        assert_eq!(moments[0].kind, "speech");
        assert_eq!(moments[0].description, "Speaker: Fuel empty.");
        assert_eq!(moments[0].start_pts_ms, 5_200);
        assert_eq!(moments[0].end_pts_ms, 5_800);
    }

    #[test]
    fn local_transcript_replaces_provider_speech_wording() {
        let analysis = AudioAnalysis {
            profile: AudioProfile::Gameplay,
            speech_engine: SpeechEngine::Whisper,
            models: vec!["openai/whisper-large-v3-turbo".to_string()],
            observations: vec![AudioObservation {
                start_offset_ms: 900,
                end_offset_ms: 3_800,
                kind: "speech".to_string(),
                label: "transcript".to_string(),
                confidence: 0.95,
                model: "openai/whisper-large-v3-turbo".to_string(),
                transcript: Some("Fuel empty. Alright, now I'm drifting.".to_string()),
                language: Some("en".to_string()),
                emotion: None,
            }],
            audio_duration_ms: 4_000,
            audio_bytes: 128_044,
            queue_wait_ms: 0,
            stages: Default::default(),
            capacity: Default::default(),
            processing_ms: 10,
        };
        let provider = SemanticMoment {
            start_offset_ms: 1_000,
            end_offset_ms: 3_000,
            start_pts_ms: 1_000,
            end_pts_ms: 3_000,
            modalities: vec!["audio".to_string()],
            kind: "speech".to_string(),
            description: "A character says the water is running low.".to_string(),
            intent: Some("Discussing the UI".to_string()),
            audio_visual_relation: None,
            confidence: 0.9,
        };

        let grounded = ground_provider_moments(vec![provider], Some(&analysis));
        assert_eq!(grounded.len(), 1);
        assert_eq!(
            grounded[0].description,
            "Speaker: Fuel empty. Alright, now I'm drifting."
        );
        assert_eq!(
            grounded[0].intent.as_deref(),
            Some("Fuel empty. Alright, now I'm drifting.")
        );

        let mut combined = audio_moments(&analysis, 0, 4_000);
        let mut mixed = grounded[0].clone();
        mixed.modalities = vec!["audio".to_string(), "video".to_string()];
        combined.push(mixed);
        deduplicate_moments(&mut combined);
        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0].modalities, ["audio", "video"]);
    }

    #[test]
    fn semantic_overlay_parser_reports_actionable_failure_categories() {
        assert_eq!(
            parse_semantic_overlay("   ", None).unwrap_err(),
            SemanticParseError::EmptyResponse
        );
        assert_eq!(
            parse_semantic_overlay("plain prose only", None).unwrap_err(),
            SemanticParseError::JsonObjectNotFound
        );
        assert_eq!(
            parse_semantic_overlay("prefix {not-json} suffix", None).unwrap_err(),
            SemanticParseError::InvalidJson
        );
        assert_eq!(
            parse_semantic_overlay("{\"summary\": \"truncated\"", None).unwrap_err(),
            SemanticParseError::JsonObjectNotFound
        );
        assert_eq!(
            parse_semantic_overlay("{}", None).unwrap_err(),
            SemanticParseError::SchemaMismatch
        );
        assert_eq!(
            parse_semantic_overlay(
                r#"{"event_type":"event","object_label":"object","summary":"summary","description":"description","confidence":2}"#,
                None
            )
            .unwrap_err(),
            SemanticParseError::SchemaMismatch
        );
    }

    #[test]
    fn semantic_overlay_parser_skips_invalid_balanced_braces_before_valid_json() {
        let raw = r#"The set {left, right} resolves to {"event_type":"scene_cut","object_label":"subject","summary":"left turn","description":"The subject turns left.","confidence":0.8}."#;
        let (overlay, _) =
            parse_semantic_overlay(raw, None).expect("later valid object should parse");
        assert_eq!(overlay.event_type, "scene_cut");
        assert_eq!(overlay.confidence, 0.8);
    }

    #[tokio::test]
    async fn custom_output_schema_preserves_raw_output_and_larger_token_cap() {
        let provider: Arc<dyn InferenceProvider + Send + Sync> = Arc::new(CustomSchemaTestProvider);
        let jpeg = DecodedJpegFrame {
            frame_index: 0,
            jpeg_bytes: Arc::from(vec![0xff, 0xd8, 0xff, 0xd9]),
        };
        let result = infer_chunk_semantics(
            Some(provider),
            true,
            "classify",
            1_000,
            1,
            &[jpeg],
            0,
            0,
            33,
            TieredVlmConfig::single_model("test-model"),
            Some(Arc::from(r#"{"type":"object"}"#)),
            None,
            None,
            None,
            None,
        )
        .await;

        assert_eq!(result.raw_output, Some(json!({"custom": "ok"})));
        assert!(result.overlay.is_none());
        assert_eq!(result.error, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_semantic_dispatch_bounds_live_spawned_tasks() {
        let max_live = super::bounded_task_spawn_probe_for_tests(100, 4).await;
        assert_eq!(max_live, 4);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_semantic_dispatch_continues_after_task_panic() {
        SEMANTIC_TASK_PANIC_CHUNK_FOR_TESTS.store(1, std::sync::atomic::Ordering::SeqCst);

        let chunk_preps: Vec<ChunkPrep> = (0..5).map(test_chunk_prep).collect();
        let provider: Arc<dyn InferenceProvider + Send + Sync> = Arc::new(SemanticTestProvider);
        let (completion_tx, mut completion_rx) = tokio::sync::mpsc::channel(2);
        let dispatch = run_semantic_dispatch(
            &chunk_preps,
            Some(provider),
            true,
            "classify",
            1_000,
            1,
            TieredVlmConfig::single_model("test-model"),
            None,
            false,
            false,
            2,
            None,
            None,
            Some(completion_tx),
        );
        let collect_completions = async {
            let mut completions = Vec::new();
            while let Some(completion) = completion_rx.recv().await {
                completions.push(completion);
            }
            completions
        };
        let ((results, _finished), completions) = tokio::join!(dispatch, collect_completions);

        SEMANTIC_TASK_PANIC_CHUNK_FOR_TESTS.store(usize::MAX, std::sync::atomic::Ordering::SeqCst);

        assert_eq!(results.len(), 5);
        assert_eq!(completions.len(), 5);
        let mut completed_indices = completions
            .iter()
            .map(|(chunk_idx, _)| *chunk_idx)
            .collect::<Vec<_>>();
        completed_indices.sort_unstable();
        assert_eq!(completed_indices, vec![0, 1, 2, 3, 4]);
        for idx in [0usize, 2, 3, 4] {
            let result = results[idx].as_ref().expect("chunk should complete");
            assert!(result.attempted, "chunk {idx} should be attempted");
            assert_eq!(result.error, None, "chunk {idx} should not inherit panic");
            assert_eq!(
                result
                    .overlay
                    .as_ref()
                    .map(|overlay| overlay.summary.as_str()),
                Some("ok")
            );
        }

        let failed = results[1].as_ref().expect("panic should be surfaced");
        assert!(failed.attempted);
        assert!(failed.used_fallback);
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|err| err.starts_with("join_error:") && err.contains("panicked")),
            "expected chunk 1 join panic error, got {:?}",
            failed.error
        );
    }

    #[test]
    fn chunk_prep_dispatch_clones_share_jpeg_storage() {
        let jpeg_bytes: Arc<[u8]> = Arc::from(vec![0xff, 0xd8, 0xff, 0xd9]);
        let prep = ChunkPrep {
            analyzed: Vec::new(),
            frame_offset: 0,
            chunk_jpegs: Arc::from([DecodedJpegFrame {
                frame_index: 0,
                jpeg_bytes: Arc::clone(&jpeg_bytes),
            }]),
            clip_spec: None,
            pts_start_ms: 0,
            pts_end_ms: 33,
            chunk_len: 1,
            started: Instant::now(),
        };

        let chunk_jpegs_c = Arc::clone(&prep.chunk_jpegs);
        let cloned_frame = prep.chunk_jpegs[0].clone();

        assert!(Arc::ptr_eq(&prep.chunk_jpegs, &chunk_jpegs_c));
        assert!(Arc::ptr_eq(
            &prep.chunk_jpegs[0].jpeg_bytes,
            &chunk_jpegs_c[0].jpeg_bytes
        ));
        assert!(Arc::ptr_eq(
            &prep.chunk_jpegs[0].jpeg_bytes,
            &cloned_frame.jpeg_bytes
        ));
    }

    fn test_chunk_prep(idx: usize) -> ChunkPrep {
        ChunkPrep {
            analyzed: Vec::new(),
            frame_offset: idx,
            chunk_jpegs: Arc::from([DecodedJpegFrame {
                frame_index: idx as u64,
                jpeg_bytes: Arc::from(vec![0xff, 0xd8, 0xff, idx as u8]),
            }]),
            clip_spec: None,
            pts_start_ms: idx as u64 * 33,
            pts_end_ms: idx as u64 * 33,
            chunk_len: 1,
            started: Instant::now(),
        }
    }
}

fn normalize_semantic_event(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "scene_cut" => "scene_cut".to_string(),
        "artifact_suspected" => "artifact_suspected".to_string(),
        "keyframe_keep" => "keyframe_keep".to_string(),
        "context_observation" => "context_observation".to_string(),
        _ => "context_observation".to_string(),
    }
}

fn normalize_semantic_object(raw: &str) -> String {
    let normalized = raw.trim().to_ascii_lowercase().replace(' ', "_");
    if normalized.is_empty() {
        "frame_context".to_string()
    } else {
        normalized
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compose_frame_metadata(
    state: &AppState,
    tenant_id: Option<&str>,
    run_id: &str,
    stream_id: &str,
    mode: &str,
    model: &str,
    sampling_policy: SamplingPolicy,
    sample_fps: f32,
    segment_ms: u64,
    request_id: &str,
    trace_id: &str,
    m: FrameMetadata,
    coordinates: Option<FrameCoordinates>,
    semantic: Option<&SemanticOverlay>,
    semantic_fallback: bool,
    finish_reason: Option<String>,
) -> (AnalyzeFrameMetadata, MarkerInput) {
    let (det_event_type, det_description) = match (m.scene_cut, m.suspect_artifact, m.gate_event) {
        (true, _, _) => ("scene_cut", "Hard transition detected from pass-1 gate"),
        (_, true, _) => ("artifact_suspected", "Temporal artifact signal elevated"),
        (_, _, GateEventType::KeepKeyframe) => {
            ("keyframe_keep", "Keyframe retained by deterministic gate")
        }
        _ => (
            "context_observation",
            "No hard trigger; contextual metadata only",
        ),
    };
    let det_object_label = if m.gate_event == GateEventType::KeepKeyframe {
        "keyframe_candidate"
    } else {
        "frame_context"
    };
    let event_type = semantic
        .map(|s| s.event_type.as_str())
        .unwrap_or(det_event_type);
    let object_label = semantic
        .map(|s| s.object_label.as_str())
        .unwrap_or(det_object_label);
    let description = semantic
        .map(|s| s.description.as_str())
        .unwrap_or(det_description);
    let confidence = semantic
        .map(|s| s.confidence)
        .unwrap_or(m.confidence)
        .clamp(0.0, 1.0);
    let mapped_event = state.map_event_label(tenant_id, event_type);
    let mapped_object = state.map_object_label(tenant_id, object_label);
    let summary = semantic.map(|s| s.summary.clone()).unwrap_or_else(|| {
        format!(
            "novelty={:.3}, stability={:.3}, motion={:.3}",
            m.novelty_score, m.temporal_stability, m.motion_score
        )
    });
    (
        AnalyzeFrameMetadata {
            run_id: run_id.to_string(),
            stream_id: stream_id.to_string(),
            frame_index: m.frame_index,
            pts_ms: m.pts_ms,
            coordinate_schema: coordinates.map(|_| IMAGE_COORDINATE_SCHEMA),
            coordinates,
            mode: mode.to_string(),
            model: model.to_string(),
            sampling_policy: sampling_policy.as_str().to_string(),
            sample_fps,
            window: AnalyzeWindow {
                start_ms: m.segment_start_ms,
                end_ms: m.segment_end_ms,
                segment_id: format!("seg-{:08x}", (m.segment_start_ms / segment_ms) as u32),
                source: "frame",
            },
            annotations: AnalyzeAnnotations {
                summary,
                objects: vec![AnalyzeObject {
                    label: mapped_object.label,
                    score: confidence,
                }],
                events: vec![AnalyzeEvent {
                    r#type: mapped_event.label.clone(),
                    score: confidence,
                    description: description.to_string(),
                }],
            },
            confidence,
            fallback: AnalyzeFallback {
                used: semantic_fallback
                    || mapped_event.used_fallback
                    || mapped_object.used_fallback,
            },
            trace: AnalyzeTrace {
                request_id: request_id.to_string(),
                trace_id: trace_id.to_string(),
                span_id: format!("span-{:016x}", m.frame_index),
            },
            ordering_key: format!("{}:{}:{}", run_id, m.pts_ms, m.frame_index),
            finish_reason,
        },
        MarkerInput {
            frame_index: m.frame_index,
            pts_ms: m.pts_ms,
            event_type: mapped_event.label,
            confidence,
        },
    )
}

pub fn semantic_marker_to_api_marker(marker: SemanticMarker) -> AnalyzeMarker {
    AnalyzeMarker {
        marker_id: marker.marker_id,
        run_id: marker.run_id,
        stream_id: marker.stream_id,
        event_type: marker.event_type,
        status: marker.status,
        start_frame: marker.start_frame,
        end_frame: marker.end_frame,
        start_pts_ms: marker.start_pts_ms,
        end_pts_ms: marker.end_pts_ms,
        confidence: marker.confidence,
        supersedes_marker_id: marker.supersedes_marker_id,
    }
}

pub fn percentile_ms(values: &[u64], percentile: u64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p = percentile.clamp(0, 100) as f64 / 100.0;
    let idx = ((n as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

pub fn estimate_sample_fps(signals: &[FrameSignal]) -> Option<f32> {
    if signals.len() < 2 {
        return None;
    }
    let mut delta_sum = 0u64;
    let mut delta_count = 0u64;
    for window in signals.windows(2) {
        let delta = window[1].pts_ms.saturating_sub(window[0].pts_ms);
        if delta > 0 {
            delta_sum += delta;
            delta_count += 1;
        }
    }
    if delta_count == 0 {
        return None;
    }
    let avg_ms = delta_sum as f32 / delta_count as f32;
    Some((1000.0 / avg_ms).clamp(0.2, 120.0))
}

pub fn adaptive_sample_fps(source_fps: f32) -> f32 {
    source_fps.clamp(0.2, 120.0)
}
