use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::{Id as TaskId, JoinError, JoinSet};
use vidarax_core::coordinates::{FrameCoordinates, IMAGE_COORDINATE_SCHEMA};
use vidarax_core::gate::{FrameSignal, GateEventType};
use vidarax_core::ingest::pipeline::DecodePipeline;
use vidarax_core::ingest::{DecodedJpegFrame, PreparedSource};
use vidarax_core::pipeline::{FrameMetadata, TwoPassPipeline};
use vidarax_core::provider::{
    InferenceImage, InferenceObserver, InferenceProvider, InferenceRequest, InferenceVideo,
    ProviderError, TokenUsage,
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
const DEFAULT_SEMANTIC_MAX_TOKENS: u32 = 320;
const CUSTOM_SCHEMA_MAX_TOKENS: u32 = 1024;

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

#[derive(Debug, Default)]
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
            })
        })
    }
}

pub struct ChunkPrep {
    pub analyzed: Vec<FrameMetadata>,
    pub frame_offset: usize,
    pub chunk_jpegs: Arc<[DecodedJpegFrame]>,
    pub chunk_video_clip: Option<Arc<[u8]>>,
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
    video_clip_mode: bool,
    semantic_decode_enabled: bool,
    video_clip_duration_s: f32,
    crop: Option<vidarax_core::crop::CropRegion>,
) -> Vec<ChunkPrep> {
    let mut chunk_preps: Vec<ChunkPrep> = Vec::new();
    for (chunk_idx, chunk) in signals.chunks(chunk_size).enumerate() {
        let started = Instant::now();
        let analyzed = pipeline.analyze_batch(chunk).to_vec();
        let frame_offset = chunk_idx * chunk_size;
        let chunk_jpegs: Arc<[DecodedJpegFrame]> = decoded_jpegs
            .map(|lookup| {
                let mut jpegs: Vec<DecodedJpegFrame> = (frame_offset..frame_offset + chunk.len())
                    .filter_map(|idx| lookup.get(&(idx as u64)).cloned())
                    .collect();
                jpegs.sort_by_key(|f| f.frame_index);
                Arc::from(jpegs)
            })
            .unwrap_or_else(|| Arc::from([]));

        let pts_start_ms_for_clip = chunk.first().map(|f| f.pts_ms).unwrap_or(0);
        let chunk_video_clip: Option<Arc<[u8]>> = if video_clip_mode && semantic_decode_enabled {
            let clip_start = pts_start_ms_for_clip as f32 / 1000.0;
            // Hold an owning handle to the prepared source for the whole blocking
            // task. spawn_blocking runs detached, so if the request future is
            // cancelled mid-extraction this clone keeps the prefetched temp file
            // alive until ffmpeg is done reading it.
            let clip_source = Arc::clone(prepared_source);
            let clip_pipeline = Arc::clone(decode_pipeline);
            let duration = video_clip_duration_s;
            match tokio::task::spawn_blocking(move || {
                clip_pipeline.extract_clip(clip_source.source(), clip_start, duration, crop)
            })
            .await
            {
                Ok(Ok(bytes)) => Some(Arc::from(bytes)),
                Ok(Err(err)) => {
                    tracing::warn!(
                        chunk_idx,
                        clip_start_ms = pts_start_ms_for_clip,
                        duration = video_clip_duration_s,
                        error = %err,
                        "video clip extraction failed for chunk; falling back to no-video"
                    );
                    None
                }
                Err(err) => {
                    tracing::warn!(chunk_idx, error = %err, "clip extraction task panicked");
                    None
                }
            }
        } else {
            None
        };

        chunk_preps.push(ChunkPrep {
            started,
            analyzed,
            frame_offset,
            chunk_jpegs,
            chunk_video_clip,
            pts_start_ms: chunk.first().map(|f| f.pts_ms).unwrap_or(0),
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
                prep.chunk_video_clip.as_ref().map(Arc::clone),
                observer.clone(),
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
            );
            task_chunks.insert(task_id, chunk_idx);
        }

        while let Some(joined) = join_set.join_next_with_id().await {
            match joined {
                Ok((task_id, (idx, result, finished))) => {
                    task_chunks.remove(&task_id);
                    semantic_results[idx] = Some(result);
                    task_end_times[idx] = finished;
                }
                Err(err) => {
                    let finished = Instant::now();
                    let task_id = err.id();
                    if let Some(idx) = task_chunks.remove(&task_id) {
                        semantic_results[idx] = Some(semantic_join_failure_result(err));
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
) -> (usize, TaskId) {
    let providers_c = providers;
    let prompt_c = semantic_prompt.to_string();
    let chunk_jpegs_c = Arc::clone(&prep.chunk_jpegs);
    let chunk_video_clip_c = prep.chunk_video_clip.as_ref().map(Arc::clone);
    let frame_offset = prep.frame_offset as u64;
    let pts_start_ms = prep.pts_start_ms;
    let pts_end_ms = prep.pts_end_ms;
    let tiered_config_c = tiered_config;
    let guided_json_c = guided_json_str;
    let observer_c = observer;
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
            chunk_video_clip_c,
            observer_c,
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
    video_clip: Option<Arc<[u8]>>,
    observer: Option<Arc<dyn InferenceObserver>>,
) -> ChunkSemanticResult {
    if !semantic_available {
        return ChunkSemanticResult::default();
    }

    let mut result = ChunkSemanticResult {
        attempted: true,
        used_fallback: false,
        ..ChunkSemanticResult::default()
    };
    let Some(provider) = providers else {
        result.used_fallback = true;
        result.error = Some("provider_not_configured".to_string());
        return result;
    };

    let use_video_clip = video_clip.is_some();
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

    let prompt = format!(
        "{semantic_prompt}\nchunk_frame_start={frame_start_index}\nchunk_frame_end={}\nchunk_pts_start_ms={pts_start_ms}\nchunk_pts_end_ms={pts_end_ms}",
        frame_start_index.saturating_add(chunk_jpegs.len() as u64).saturating_sub(1)
    );

    let (images, videos) = if let Some(clip_bytes) = video_clip {
        let vids = vec![InferenceVideo {
            media_type: "video/mp4",
            data_base64: BASE64_STANDARD.encode(clip_bytes.as_ref()),
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
    let first_pass_max_tokens = if has_custom_output_schema {
        CUSTOM_SCHEMA_MAX_TOKENS
    } else {
        DEFAULT_SEMANTIC_MAX_TOKENS
    };
    let request_guided_json = guided_json
        .as_ref()
        .map(Arc::clone)
        .or_else(|| Some(Arc::from(SEMANTIC_OVERLAY_SCHEMA)));
    let second_pass_guided_json =
        (!has_custom_output_schema).then(|| Arc::from(SEMANTIC_OVERLAY_SCHEMA));
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
    };

    // Capture the failing model's backend kind before the closure moves
    // tiered_config. run_tiered only surfaces Err on a first-pass failure, so
    // this attributes any recorded error to the first-pass model's backend
    // instead of the router's default kind.
    let first_pass_kind = provider.kind_for_model(tiered_config.first_pass_model.as_ref());
    let call_started = Instant::now();
    let provider_result = match tokio::task::spawn_blocking({
        let provider = Arc::clone(&provider);
        let observer_for_call = observer.clone();
        move || {
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

    if has_custom_output_schema {
        let parsed = serde_json::from_str::<Value>(&provider_result.output_text)
            .unwrap_or_else(|_| json!({"raw": provider_result.output_text}));
        result.raw_output = Some(parsed);
        result.overlay = None;
        result
    } else {
        match parse_semantic_overlay(&provider_result.output_text) {
            Ok(overlay) => {
                result.overlay = Some(overlay);
                result.used_fallback = provider_result.fallback_used;
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

fn parse_semantic_overlay(raw: &str) -> Result<SemanticOverlay, SemanticParseError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(SemanticParseError::EmptyResponse);
    }
    let value = parse_first_json_object(trimmed)?;

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

    Ok(SemanticOverlay {
        event_type,
        object_label,
        summary,
        description,
        confidence,
    })
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

        let overlay = parse_semantic_overlay(raw).expect("fenced overlay should parse");
        assert_eq!(overlay.object_label, "subject");
        assert_eq!(overlay.summary, "turn {left}");
        assert_eq!(overlay.confidence, 0.8);
    }

    #[test]
    fn semantic_overlay_parser_reports_actionable_failure_categories() {
        assert_eq!(
            parse_semantic_overlay("   ").unwrap_err(),
            SemanticParseError::EmptyResponse
        );
        assert_eq!(
            parse_semantic_overlay("plain prose only").unwrap_err(),
            SemanticParseError::JsonObjectNotFound
        );
        assert_eq!(
            parse_semantic_overlay("prefix {not-json} suffix").unwrap_err(),
            SemanticParseError::InvalidJson
        );
        assert_eq!(
            parse_semantic_overlay("{\"summary\": \"truncated\"").unwrap_err(),
            SemanticParseError::JsonObjectNotFound
        );
        assert_eq!(
            parse_semantic_overlay("{}").unwrap_err(),
            SemanticParseError::SchemaMismatch
        );
        assert_eq!(
            parse_semantic_overlay(
                r#"{"event_type":"event","object_label":"object","summary":"summary","description":"description","confidence":2}"#
            )
            .unwrap_err(),
            SemanticParseError::SchemaMismatch
        );
    }

    #[test]
    fn semantic_overlay_parser_skips_invalid_balanced_braces_before_valid_json() {
        let raw = r#"The set {left, right} resolves to {"event_type":"scene_cut","object_label":"subject","summary":"left turn","description":"The subject turns left.","confidence":0.8}."#;
        let overlay = parse_semantic_overlay(raw).expect("later valid object should parse");
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
        let (results, _finished) = run_semantic_dispatch(
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
        )
        .await;

        SEMANTIC_TASK_PANIC_CHUNK_FOR_TESTS.store(usize::MAX, std::sync::atomic::Ordering::SeqCst);

        assert_eq!(results.len(), 5);
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
    fn chunk_prep_dispatch_clones_share_heavy_payload_storage() {
        let jpeg_bytes: Arc<[u8]> = Arc::from(vec![0xff, 0xd8, 0xff, 0xd9]);
        let clip_bytes: Arc<[u8]> = Arc::from(vec![0, 0, 0, 24, b'f', b't', b'y', b'p']);
        let prep = ChunkPrep {
            analyzed: Vec::new(),
            frame_offset: 0,
            chunk_jpegs: Arc::from([DecodedJpegFrame {
                frame_index: 0,
                jpeg_bytes: Arc::clone(&jpeg_bytes),
            }]),
            chunk_video_clip: Some(Arc::clone(&clip_bytes)),
            pts_start_ms: 0,
            pts_end_ms: 33,
            chunk_len: 1,
            started: Instant::now(),
        };

        let chunk_jpegs_c = Arc::clone(&prep.chunk_jpegs);
        let chunk_video_clip_c = prep.chunk_video_clip.as_ref().map(Arc::clone);
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
        assert!(Arc::ptr_eq(
            prep.chunk_video_clip.as_ref().expect("clip present"),
            chunk_video_clip_c.as_ref().expect("clip clone present")
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
            chunk_video_clip: None,
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
